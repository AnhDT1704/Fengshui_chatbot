"""
supervisor_agent.py – Routing brain of the chatbot.

This file only owns:
  - SupervisorState
  - SUPERVISOR_SYSTEM_PROMPT
  - supervisor_node  (LLM-based routing decision)
  - route_to_agent   (graph conditional edge)

The full graph (wiring real sub-agents) is built in graph.py.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import os
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages

from gemini import make_llm
from logger import get_logger


MODEL_NAME = os.getenv("CHATBOT_MODEL", "gemini-2.5-flash")
log = get_logger("supervisor")


# ═══════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════

class SupervisorState(TypedDict):
    messages:       Annotated[Sequence[BaseMessage], add_messages]
    next_agent:     str
    intent:         str
    final_response: str
    session_id:     str


VALID_AGENTS = {
    "small_talk",
    "knowledge_base_agent",
    "skills_agent",
    "vision_agent",
    "order_support_agent",
}


# ═══════════════════════════════════════════════════════════════════
#  ROUTING PROMPT
# ═══════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM_PROMPT = """
Bạn là Supervisor của hệ thống chatbot tư vấn sản phẩm phong thủy Vạn An Group.

Nhiệm vụ duy nhất: đọc tin nhắn cuối của user (kèm context hội thoại) rồi
quyết định chuyển đến agent nào phù hợp nhất.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÁC AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

small_talk
  Chào hỏi, cảm ơn, tạm biệt, hỏi linh tinh không liên quan sản phẩm
  ("hello", "shop ơi", "cảm ơn nhé", "ok rồi", emoji thuần,...).
  CHỌN agent này cho mọi message ngắn mang tính giao tiếp xã giao.

knowledge_base_agent
  Tìm kiếm, lọc, so sánh, xem chi tiết SẢN PHẨM trong DB shop.
  Ví dụ: "shop có vòng aquamarine không", "vòng nào dưới 200k",
  "so sánh vòng tourmaline và mã não".

skills_agent
  Câu hỏi cần TÍNH TOÁN hoặc TƯ VẤN CHUYÊN MÔN:
  - Tính size vòng tay từ cm cổ tay
  - Tư vấn theo MỆNH / TUỔI / NĂM SINH (Can Chi Nạp Âm)
  - Tư vấn quà tặng theo người nhận / dịp
  - Tìm sản phẩm ngoài DB shop (web search)

vision_agent
  Mọi message CÓ KÈM ẢNH user gửi lên, HOẶC user yêu cầu XEM ẢNH sản phẩm của shop.

order_support_agent
  Bảo hành, đổi trả, hoàn tiền, giao hàng (ship/COD/phí/thời gian), kiểm tra
  tồn kho, tra cứu đơn hàng, khiếu nại, ĐỊA CHỈ CỬA HÀNG, hỏi shop ở đâu,
  thanh toán, mã giảm giá, custom mix sản phẩm.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Nếu request có ẢNH → vision_agent (gần như luôn luôn).
2. Nếu chỉ là chào / cảm ơn / emoji → small_talk.
3. Nếu nhiều intent, chọn theo intent CHÍNH (cái user thực sự muốn biết).
4. Nghi ngờ giữa knowledge_base và skills:
   - User mô tả/lọc sản phẩm cụ thể → knowledge_base
   - User cần tính toán / suy luận phong thủy / tư vấn → skills
5. Chỉ trả về DUY NHẤT 1 trong các tên:
   small_talk
   knowledge_base_agent
   skills_agent
   vision_agent
   order_support_agent

KHÔNG giải thích, KHÔNG thêm ký tự khác.
"""


# ═══════════════════════════════════════════════════════════════════
#  SUPERVISOR NODE
# ═══════════════════════════════════════════════════════════════════

def _routing_context(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Strip ToolMessages so routing LLM only sees user / assistant turns."""
    return [m for m in messages if not isinstance(m, ToolMessage)]


def supervisor_node(state: SupervisorState) -> SupervisorState:
    llm = make_llm(temperature=0, max_tokens=20)
    routing_input = (
        [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)]
        + _routing_context(state["messages"])
    )
    response = llm.invoke(routing_input)
    raw = (response.content or "").strip().lower()

    # Normalize whatever the LLM returned (it might say "knowledge base" or
    # add punctuation). Fall back to knowledge_base_agent if unparseable.
    chosen = None
    for name in VALID_AGENTS:
        if name in raw:
            chosen = name
            break
    if chosen is None:
        chosen = "knowledge_base_agent"

    # Get a snippet of latest user turn for log readability
    snippet = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            snippet = c.replace("\n", " ")[:80]
            break
    log.info("ROUTE → %-22s | user='%s'", chosen, snippet)
    if chosen != raw:
        log.debug("       (raw LLM output: %r)", raw)

    # Only return changed keys — do NOT spread state.
    # Spreading would re-send messages through add_messages reducer
    # which would duplicate the conversation.
    return {
        "next_agent": chosen,
        "intent":     chosen,
    }


def route_to_agent(
    state: SupervisorState,
) -> Literal[
    "small_talk",
    "knowledge_base_agent",
    "skills_agent",
    "vision_agent",
    "order_support_agent",
]:
    return state["next_agent"]  # type: ignore[return-value]
