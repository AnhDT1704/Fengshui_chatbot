"""
graph.py – Wire the supervisor + all sub-agents into one runnable graph.

Public API:
    chat(user_message, session_id="default", history=None) -> dict
    chat_with_image(user_message, image_base64, image_mime, session_id, history) -> dict

Each chat() call:
  1. Loads recent turns from conversation_log for the session_id (memory).
  2. Appends the new user message and runs the supervisor graph.
  3. Logs both turns to conversation_log.
  4. Returns { response, agent_used, intent, tools_called, messages }
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import os
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import knowledge_base_agent
import order_support_agent
import skills_agent
import vision_agent
from gemini import make_llm
from logger import get_logger
from memory import load_recent_history, log_turn
from supervisor_agent import (
    SUPERVISOR_SYSTEM_PROMPT,
    SupervisorState,
    route_to_agent,
    supervisor_node,
)


MODEL_NAME = os.getenv("CHATBOT_MODEL", "gemini-2.5-flash")
MEMORY_LIMIT = int(os.getenv("CHATBOT_MEMORY_LIMIT", "20"))

log         = get_logger("graph")
_st_log     = get_logger("small_talk")


# ═══════════════════════════════════════════════════════════════════
#  SMALL TALK NODE
#  Replies in-line without dispatching to any tool-using agent.
# ═══════════════════════════════════════════════════════════════════

SMALL_TALK_PROMPT = """
Bạn là chatbot của shop phong thủy Vạn An Group. Đây là một message giao tiếp
xã giao (chào hỏi, cảm ơn, tạm biệt, emoji,...).

Hãy trả lời NGẮN GỌN (1-2 câu), thân thiện, xưng "shop", và nếu phù hợp thì gợi
ý mở: "bạn cần tư vấn sản phẩm gì giúp shop biết với nha?" hoặc "bạn cần hỗ trợ
gì thêm không?". Không dùng emoji quá đà.
"""


def small_talk_node(state: SupervisorState) -> dict:
    _st_log.info("ENTER  small_talk")
    llm = make_llm(temperature=0.7, max_tokens=120)
    response = llm.invoke(
        [SystemMessage(content=SMALL_TALK_PROMPT)] + list(state["messages"])
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    _st_log.info("EXIT   small_talk | reply=%d chars", len(text))
    return {
        "final_response": text,
        # add_messages reducer appends — only return the NEW message
        "messages":       [AIMessage(content=text)],
    }


# ═══════════════════════════════════════════════════════════════════
#  SUB-AGENT NODES (delegate to each module's run())
# ═══════════════════════════════════════════════════════════════════

def _wrap(run_fn):
    """Convert a sub-agent's run() into a SupervisorState node function."""
    def node(state: SupervisorState) -> dict:
        input_messages = list(state["messages"])
        n_input = len(input_messages)
        result = run_fn(input_messages)
        # Only return NEW messages produced by the sub-agent so the
        # add_messages reducer doesn't duplicate the conversation.
        new_messages = list(result["messages"])[n_input:]
        return {
            "final_response": result["final_response"],
            "messages":       new_messages,
        }
    return node


knowledge_base_node = _wrap(knowledge_base_agent.run)
skills_node         = _wrap(skills_agent.run)
vision_node         = _wrap(vision_agent.run)
order_support_node  = _wrap(order_support_agent.run)


# ═══════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(SupervisorState)

    g.add_node("supervisor",            supervisor_node)
    g.add_node("small_talk",            small_talk_node)
    g.add_node("knowledge_base_agent",  knowledge_base_node)
    g.add_node("skills_agent",          skills_node)
    g.add_node("vision_agent",          vision_node)
    g.add_node("order_support_agent",   order_support_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "small_talk":            "small_talk",
            "knowledge_base_agent":  "knowledge_base_agent",
            "skills_agent":          "skills_agent",
            "vision_agent":          "vision_agent",
            "order_support_agent":   "order_support_agent",
        },
    )

    for terminal in [
        "small_talk",
        "knowledge_base_agent",
        "skills_agent",
        "vision_agent",
        "order_support_agent",
    ]:
        g.add_edge(terminal, END)

    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ═══════════════════════════════════════════════════════════════════
#  PUBLIC CHAT API
# ═══════════════════════════════════════════════════════════════════

def _collect_tool_calls(messages: list[BaseMessage]) -> list[str]:
    return sorted({
        tc["name"]
        for m in messages
        for tc in getattr(m, "tool_calls", []) or []
    })


def _invoke_graph(
    user_message_obj: BaseMessage,
    session_id: str,
    history_override: Optional[list[BaseMessage]] = None,
    log_user_text: str = "",
) -> dict:
    graph = get_graph()

    if history_override is not None:
        history = list(history_override)
    else:
        history = load_recent_history(session_id, limit=MEMORY_LIMIT)

    full_messages = history + [user_message_obj]

    log.info("┌── REQUEST  session=%s  history=%d turns  user='%s'",
             session_id, len(history), (log_user_text or "")[:100].replace("\n", " "))

    state_in = {
        "messages":       full_messages,
        "next_agent":     "",
        "intent":         "",
        "final_response": "",
        "session_id":     session_id,
    }

    result = graph.invoke(state_in)

    final_response = result.get("final_response") or ""
    if not final_response and result["messages"]:
        last = result["messages"][-1]
        if hasattr(last, "content"):
            final_response = last.content if isinstance(last.content, str) else str(last.content)

    agent_used   = result.get("next_agent", "")
    tools_called = _collect_tool_calls(list(result["messages"]))

    # Persist to memory
    if log_user_text:
        try:
            log_turn(session_id, "user", log_user_text, agent_used=agent_used, intent=agent_used)
        except Exception as e:
            log.warning("log_turn(user) failed: %s", e)
    try:
        log_turn(
            session_id,
            "assistant",
            final_response,
            agent_used=agent_used,
            intent=agent_used,
            tools_called=tools_called,
        )
    except Exception as e:
        log.warning("log_turn(assistant) failed: %s", e)

    log.info("└── RESPONSE agent=%s  tools=%s  reply=%d chars",
             agent_used, tools_called, len(final_response or ""))

    return {
        "response":     final_response,
        "agent_used":   agent_used,
        "intent":       result.get("intent", ""),
        "tools_called": tools_called,
        "messages":     list(result["messages"]),
    }


def chat(
    user_message: str,
    session_id: str = "default",
    history: Optional[list[BaseMessage]] = None,
) -> dict:
    """Text-only chat."""
    return _invoke_graph(
        user_message_obj = HumanMessage(content=user_message),
        session_id       = session_id,
        history_override = history,
        log_user_text    = user_message,
    )


def chat_with_image(
    user_message: str,
    image_base64: str,
    image_mime: str = "image/jpeg",
    session_id: str = "default",
    history: Optional[list[BaseMessage]] = None,
) -> dict:
    """Chat with one image attached. Gemini sees the image natively."""
    image_message = HumanMessage(content=[
        {"type": "text", "text": user_message or "Bạn xem giúp mình ảnh này"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime};base64,{image_base64}"},
        },
    ])
    log_text = (user_message or "[ảnh]") + "  [+image]"
    return _invoke_graph(
        user_message_obj = image_message,
        session_id       = session_id,
        history_override = history,
        log_user_text    = log_text,
    )


# ═══════════════════════════════════════════════════════════════════
#  QUICK CLI TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_cases = [
        "chào shop",                                                # small_talk
        "shop có vòng tay đá tourmaline không?",                    # KB
        "cổ tay mình 16cm thì đeo size mấy ly?",                    # skills (size)
        "mình sinh năm 1990 thì hợp đá nào?",                       # skills (fengshui)
        "mình tuổi Tý thì hợp gì?",                                 # skills (ask birth year back)
        "ship về Đà Nẵng mất bao lâu, có COD không?",               # order
        "shop ở đâu vậy mình ghé qua được không?",                  # order (address policy)
        "đơn của em mã 250115001 đến đâu rồi shop ơi",              # order (escalate)
        "cảm ơn shop nhé",                                          # small_talk
    ]

    sid = "cli-test"
    for i, msg in enumerate(test_cases, 1):
        print(f"\n=== {i} ===\nUSER: {msg}")
        out = chat(msg, session_id=sid)
        print(f"AGENT: {out['agent_used']}")
        print(f"TOOLS: {out['tools_called']}")
        print(f"BOT  : {out['response'][:400]}")
