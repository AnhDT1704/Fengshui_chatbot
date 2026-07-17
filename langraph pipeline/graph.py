"""
graph.py – Wire the supervisor + all sub-agents into one runnable graph.

Public API:
    chat(user_message, session_id="default", history=None) -> dict
    chat_with_image(user_message, images, image_base64, image_mime, session_id, history) -> dict

Each chat() call:
  1. Loads recent turns from conversation_log for the session_id (memory).
  2. Appends the new user message and runs the supervisor graph.
  3. Logs both turns to conversation_log.
  4. Returns { response, agent_used, intent, tools_called, messages }
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os
import re
import unicodedata
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

import knowledge_base_agent
import order_support_agent
import response_verifier
import skills_agent
from gemini import make_llm
from logger import get_logger
import memory
from memory import load_recent_history, log_turn
from supervisor_agent import (
    SUPERVISOR_SYSTEM_PROMPT,
    SupervisorState,
    route_to_agent,
    supervisor_node,
)


MODEL_NAME = os.getenv("CHATBOT_MODEL", "gemini-2.5-flash")
MEMORY_LIMIT = int(os.getenv("CHATBOT_MEMORY_LIMIT", "20"))
# Số ảnh tối đa khách được đính kèm trong một lượt.
MAX_IMAGES = int(os.getenv("CHATBOT_MAX_IMAGES", "5"))

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


# Câu trả lời CỐ ĐỊNH (nguyên văn) cho mọi yêu cầu xin liên hệ/địa chỉ shop hoặc
# rủ giao dịch ngoài Shopee — theo đúng cách nhân viên thật phản hồi.
OFF_PLATFORM_REPLY = (
    "Dạ shopee cấm giao dịch ngoài nên bạn thông cảm mua qua shopee giúp shop ạ"
)


def _advance(state: SupervisorState) -> int:
    """Step kế tiếp sau khi node hiện tại chạy xong (để route_to_agent biết đi tiếp/END)."""
    return state.get("step", 0) + 1


def _strip_images(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Bỏ phần ẢNH khỏi các HumanMessage đa phương thức, thay bằng ghi chú text.

    Dùng cho agent KHÔNG cần xem ảnh (vd skills_agent chỉ cần số đo cổ tay dạng chữ).
    Khi để nguyên ảnh, model đa phương thức hay trả lời hội thoại và BỎ QUA tool-call
    → tính số hạt sai. Gỡ ảnh giúp skills tập trung gọi size_calculator_tool.
    """
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage) and isinstance(m.content, list):
            has_img = any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in m.content
            )
            text = " ".join(
                p.get("text", "") for p in m.content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
            if has_img:
                text = (text + "\n[Khách có gửi kèm ẢNH sản phẩm vòng tay.]").strip()
            out.append(HumanMessage(content=text))
        else:
            out.append(m)
    return out


def off_platform_policy_node(state: SupervisorState) -> dict:
    _st_log.info("ENTER  off_platform_policy (canned reply)")
    return {
        "final_response": OFF_PLATFORM_REPLY,
        "messages":       [AIMessage(content=OFF_PLATFORM_REPLY)],
        "next_agent":     "off_platform_policy",
        "step":           _advance(state),
    }


# Chuyển cuộc trò chuyện cho CHỦ SHOP (dịch vụ phụ / yêu cầu đặc biệt ngoài dữ liệu).
# Đặt phiên sang 'pending_admin' → bot NGỪNG trả lời, chủ shop (admin1) trả lời trực tiếp.
OTHER_SERVICE_REPLY = (
    "Dạ yêu cầu này shop sẽ phản hồi trực tiếp cho mình nha 🙏 Bạn vui lòng chờ shop "
    "một chút, shop sẽ trả lời ngay tại đây ạ."
)


def other_service_node(state: SupervisorState) -> dict:
    sid = state.get("session_id") or ""
    try:
        if sid:
            memory.set_session_status(sid, "pending_admin")
    except Exception as e:
        _st_log.warning("set handoff status failed: %s", e)
    _st_log.info("ENTER  other_service_agent (handoff → pending_admin) session=%s", sid)
    return {
        "final_response": OTHER_SERVICE_REPLY,
        "messages":       [AIMessage(content=OTHER_SERVICE_REPLY)],
        "next_agent":     "other_service_agent",
        "step":           _advance(state),
    }


def small_talk_node(state: SupervisorState) -> dict:
    _st_log.info("ENTER  small_talk")
    # max_tokens phải đủ lớn vì Gemini 2.5 Flash "thinking" tiêu tốn token budget
    # trước khi sinh câu trả lời hiển thị; để thấp (vd 120) sẽ bị cắt giữa chừng.
    llm = make_llm(temperature=0.7, max_tokens=8192)
    response = llm.invoke(
        [SystemMessage(content=SMALL_TALK_PROMPT)] + list(state["messages"])
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    _st_log.info("EXIT   small_talk | reply=%d chars", len(text))
    return {
        "final_response": text,
        # add_messages reducer appends — only return the NEW message
        "messages":       [AIMessage(content=text)],
        "next_agent":     "small_talk",
        "step":           _advance(state),
    }


# ═══════════════════════════════════════════════════════════════════
#  SUB-AGENT NODES (delegate to each module's run())
# ═══════════════════════════════════════════════════════════════════

def _wrap(run_fn, name: str, strip_images: bool = False):
    """Convert a sub-agent's run() into a SupervisorState node function.

    Mỗi agent: chạy run() trên toàn bộ messages (gồm cả output của agent CHẠY TRƯỚC
    trong chuỗi, nếu có), trả về NEW messages + final_response, rồi tăng 'step' để
    route_to_agent chuyển sang agent kế tiếp trong plan hoặc kết thúc.
    'final_response' của agent CHẠY SAU sẽ ghi đè agent trước → câu trả lời cuối là
    của agent cuối chuỗi. 'next_agent' = agent vừa chạy (để log + lớp verify nhận biết).

    strip_images=True: gỡ ảnh khỏi input agent này (vd skills_agent không cần ảnh, để
    nó gọi tool đáng tin thay vì sa đà mô tả ảnh).
    """
    def node(state: SupervisorState) -> dict:
        input_messages = list(state["messages"])
        n_input = len(input_messages)
        run_input = _strip_images(input_messages) if strip_images else input_messages
        result = run_fn(run_input)
        # Only return NEW messages produced by the sub-agent so the
        # add_messages reducer doesn't duplicate the conversation.
        new_messages = list(result["messages"])[n_input:]

        next_step = _advance(state)
        plan      = state.get("plan") or [name]
        is_last   = next_step >= len(plan)

        if is_last:
            out_messages = new_messages
        else:
            # Agent TRUNG GIAN trong chuỗi: KHÔNG rò rỉ tool-call/tool-result thô sang
            # agent sau (agent sau không khai báo các tool đó → Gemini có thể lỗi
            # function_response). Gói kết quả thành 1 GHI CHÚ NỘI BỘ để agent sau đọc.
            summary = (result.get("final_response") or "").strip()
            out_messages = [HumanMessage(content=(
                f"[GHI CHÚ NỘI BỘ HỆ THỐNG — kết quả tính từ {name}, hãy DÙNG để soạn "
                f"câu trả lời cho khách, ĐỪNG nhắc lại cụm 'ghi chú nội bộ']\n{summary}"
            ))]

        return {
            "final_response": result["final_response"],
            "messages":       out_messages,
            "next_agent":     name,
            "step":           next_step,
        }
    return node


knowledge_base_node = _wrap(knowledge_base_agent.run, "knowledge_base_agent")
skills_node         = _wrap(skills_agent.run,         "skills_agent", strip_images=True)
order_support_node  = _wrap(order_support_agent.run,  "order_support_agent")


# ═══════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ═══════════════════════════════════════════════════════════════════

def build_graph():
    g = StateGraph(SupervisorState)

    g.add_node("supervisor",            supervisor_node)
    g.add_node("small_talk",            small_talk_node)
    g.add_node("off_platform_policy",   off_platform_policy_node)
    g.add_node("other_service_agent",   other_service_node)
    g.add_node("knowledge_base_agent",  knowledge_base_node)
    g.add_node("skills_agent",          skills_node)
    g.add_node("order_support_agent",   order_support_node)

    # route_to_agent đọc plan/step → đi tới agent kế tiếp trong chuỗi hoặc END.
    # Dùng CHUNG cho supervisor và mọi agent node (sau khi chạy, agent tự +step).
    route_map = {
        "small_talk":            "small_talk",
        "off_platform_policy":   "off_platform_policy",
        "other_service_agent":   "other_service_agent",
        "knowledge_base_agent":  "knowledge_base_agent",
        "skills_agent":          "skills_agent",
        "order_support_agent":   "order_support_agent",
        END:                     END,
    }

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_to_agent, route_map)
    for node_name in [
        "small_talk",
        "off_platform_policy",
        "other_service_agent",
        "knowledge_base_agent",
        "skills_agent",
        "order_support_agent",
    ]:
        g.add_conditional_edges(node_name, route_to_agent, route_map)

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


# ═══════════════════════════════════════════════════════════════════
#  LỜI MỜI XEM SHOPEE
# ═══════════════════════════════════════════════════════════════════
# Luật của shop: CHỈ mời khách qua Shopee khi khách hỏi (1) GIÁ hoặc (2) KHUYẾN MÃI.
#
# Chia việc theo đúng thế mạnh, sau khi thử hỏng cả 2 thái cực:
#   - Giao HẲN cho prompt  → Gemini chèn bừa cả khi khách hỏi ý nghĩa/tồn kho.
#   - Giao HẲN cho regex   → giòn: vỡ vì Unicode tổ hợp (trình duyệt gửi "ả" = a + dấu
#                            rời, regex viết dạng dựng sẵn nên không khớp), và trật với
#                            cách hỏi lạ ("mắc quá không", "tiền nong sao").
# Nên:
#   - HIỂU Ý ĐỊNH ("khách có đang hỏi giá/khuyến mãi?") → việc của NGÔN NGỮ → LLM lo.
#   - CHÈN CÂU + LINK (đúng chỗ, đúng URL, không bịa)   → việc MÁY MÓC     → code lo.

SHOPEE_CTA = ("ngoài ra bạn có thể tìm sản phẩm này trên [sàn Shopee của shop]"
              "(https://shopee.vn/vananhome?entryPoint=ShopBySearch&"
              "searchKeyword=va%CC%A3n%20an%20home) để nhận được các giảm giá ạ")

_CTA_CLASSIFY_PROMPT = (
    "Bạn là bộ phân loại ý định cho chatbot bán đồ phong thủy. Đọc TIN NHẮN của khách "
    "và cho biết khách có đang hỏi về GIÁ sản phẩm hoặc KHUYẾN MÃI/GIẢM GIÁ hay không.\n\n"
    "true — khi khách hỏi về:\n"
    "  • GIÁ: 'giá bao nhiêu', 'bao nhiêu tiền', 'có đắt không', 'mắc quá không', "
    "'giá cả sao', 'rẻ hơn không', so sánh giá giữa các mẫu...\n"
    "  • KHUYẾN MÃI: 'có giảm giá không', 'đang sale gì', 'có voucher / mã giảm không', "
    "'ưu đãi gì', 'có chương trình gì không'...\n\n"
    "false — MỌI thứ khác, đặc biệt lưu ý các ca DỄ NHẦM:\n"
    "  • TỒN KHO: 'còn bao nhiêu cái', 'còn hàng không', 'số lượng bao nhiêu' → đây là "
    "SỐ LƯỢNG, KHÔNG phải giá → false\n"
    "  • Ý NGHĨA / công dụng của đá, chất liệu → false\n"
    "  • 'đánh giá sản phẩm thế nào' (hỏi review, không phải giá) → false\n"
    "  • size, mệnh hợp, màu sắc, bảo quản, xem ảnh, giao hàng, đổi trả, chào hỏi → false\n\n"
    'CHỈ trả về JSON, không thêm chữ nào: {"price_or_promo": true} hoặc {"price_or_promo": false}'
)

# Lưới an toàn khi LLM lỗi/hết quota. Chuẩn hoá NFC trước khi khớp (xem _nfc).
_PRICE_RE = re.compile(
    r"(?<!đánh )\bgiá\b(?! trị)|bao nhiêu tiền|bao nhiêu đồng|\bmắc không|\bđắt\b|\brẻ\b|bảng giá",
    re.IGNORECASE,
)
_PROMO_RE = re.compile(
    r"khuyến mãi|khuyến mại|giảm giá|\bsale\b|voucher|mã giảm|ưu đãi|\bdeal\b|discount",
    re.IGNORECASE,
)


def _nfc(s: str) -> str:
    """Chuẩn hoá Unicode về NFC.

    Trình duyệt/bộ gõ tiếng Việt có thể gửi dạng TỔ HỢP (NFD): "ả" = "a" + dấu hỏi rời.
    Chuỗi nhìn y hệt nhưng byte khác → so khớp chuỗi trượt. Đây chính là lỗi đã khiến
    link Shopee không hiện khi hỏi từ UI, dù gọi thẳng API thì hiện.
    """
    return unicodedata.normalize("NFC", s or "")


def _wants_shopee_cta(user_text: str) -> bool:
    """Khách có đang hỏi về GIÁ hoặc KHUYẾN MÃI không?

    Dùng LLM để HIỂU ý định (bền với mọi cách diễn đạt, không phụ thuộc từ khoá).
    Nếu LLM lỗi → rơi về regex cho khỏi mất tính năng.
    """
    t = _nfc((user_text or "").strip())
    if not t:
        return False

    try:
        llm = make_llm(temperature=0.0, max_tokens=32)
        resp = llm.invoke([SystemMessage(content=_CTA_CLASSIFY_PROMPT),
                           HumanMessage(content=t)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1:
            want = bool(json.loads(raw[s:e + 1]).get("price_or_promo"))
            log.info("shopee_cta: LLM phân loại '%s' → %s", t[:50], want)
            return want
        log.warning("shopee_cta: LLM trả về không phải JSON (%r) → dùng regex", raw[:80])
    except Exception as e:
        log.warning("shopee_cta: LLM lỗi (%s) → dùng regex dự phòng", e)

    return bool(_PRICE_RE.search(t) or _PROMO_RE.search(t))


def _apply_shopee_cta(user_text: str, answer: str) -> str:
    """Đảm bảo lời mời Shopee xuất hiện ĐÚNG khi cần, và biến mất khi không cần."""
    if not answer:
        return answer
    # LLM sinh câu trả lời vẫn có thể tự chèn link dù prompt đã cấm → gỡ sạch trước,
    # rồi CODE tự quyết định có chèn lại hay không.
    lines = [ln for ln in answer.split("\n") if "shopee.vn/vananhome" not in ln]
    cleaned = "\n".join(lines).rstrip()

    if _wants_shopee_cta(user_text):
        return cleaned + "\n\n" + SHOPEE_CTA
    return cleaned


def _focused_product_refs(messages: list[BaseMessage], answer: str) -> Optional[list[dict]]:
    """Sản phẩm mà lượt này XÁC ĐỊNH/TRÌNH BÀY (để gắn metadata vào tin nhắn → nhớ ngữ cảnh).
    Ưu tiên sản phẩm có tên xuất hiện trong câu trả lời; nếu không khớp thì lấy tối đa 5 SP
    đã tra được trong lượt."""
    try:
        grounded = response_verifier.collect_grounded_products(messages)
    except Exception:
        grounded = []
    if not grounded:
        return None
    ans = (answer or "").lower()
    focused = [p for p in grounded if (p.get("name") or "")[:16].lower() in ans]
    refs = focused or grounded[:5]
    out = [{"id": p["product_id"], "name": p["name"]} for p in refs if p.get("product_id") is not None]
    return out or None


def _invoke_graph(
    user_message_obj: BaseMessage,
    session_id: str,
    history_override: Optional[list[BaseMessage]] = None,
    log_user_text: str = "",
    user_id: Optional[int] = None,
    image_urls: Optional[list[str]] = None,
) -> dict:
    graph = get_graph()

    if history_override is not None:
        history = list(history_override)
    elif MEMORY_LIMIT <= 0:
        # Memory TẮT (CHATBOT_MEMORY_LIMIT<=0): bot xử lý mỗi tin nhắn độc lập,
        # không nạp lịch sử hội thoại. Tạm thời tắt chờ tích hợp mem0.
        history = []
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
        "plan":           [],
        "step":           0,
    }

    result = graph.invoke(state_in)

    final_response = result.get("final_response") or ""
    if not final_response and result["messages"]:
        last = result["messages"][-1]
        if hasattr(last, "content"):
            final_response = last.content if isinstance(last.content, str) else str(last.content)

    agent_used   = result.get("next_agent", "")
    tools_called = _collect_tool_calls(list(result["messages"]))

    # ── LỚP KIỂM TRA CHỐNG BỊA SẢN PHẨM ─────────────────────────────
    # Chỉ áp cho câu trả lời của KB agent CÓ nhắc sản phẩm/giá. Nếu phát hiện
    # sản phẩm/giá không có trong DB (so với tập tool đã tra trong lượt) → sinh
    # lại MỘT lần với ghi chú sửa lỗi, ép agent gọi tool lấy dữ liệu thật.
    if agent_used == "knowledge_base_agent" and response_verifier.is_product_answer(final_response):
        try:
            grounded = response_verifier.collect_grounded_products(list(result["messages"]))
            verdict  = response_verifier.verify_answer(final_response, grounded)
            if not verdict.get("ok", True):
                log.warning("VERIFY fail → regenerate | issues=%s", verdict.get("issues"))
                # knowledge_base_agent.run() tự nhận diện + tiêm sản phẩm thật cho
                # lượt CÓ ẢNH; ở đây chỉ cần thêm ghi chú ép dùng dữ liệu DB thật.
                fix_note = SystemMessage(content=(
                    "KIỂM DUYỆT NỘI BỘ: câu trả lời nháp vừa rồi bị nghi BỊA sản phẩm/giá "
                    "không có trong DB. " + (verdict.get("fix_hint") or "") + " "
                    "Hãy dùng dữ liệu sản phẩm THẬT (kết quả tool/đã tra DB) và CHỈ nêu "
                    "sản phẩm + giá có trong đó. Với câu hỏi không ảnh, GỌI keyword_search_tool/"
                    "filter_search_tool. Nếu không tra được thì nói shop sẽ kiểm tra lại, "
                    "TUYỆT ĐỐI không bịa."
                ))
                # Giữ lại GHI CHÚ NỘI BỘ (số hạt skills đã tính) nếu lượt này là chuỗi
                # skills→KB, để regen KHÔNG tính lại số hạt sai.
                carry = [
                    m for m in result["messages"]
                    if isinstance(m, HumanMessage)
                    and isinstance(m.content, str)
                    and m.content.startswith("[GHI CHÚ NỘI BỘ")
                ]
                regen = knowledge_base_agent.run(full_messages + carry + [fix_note])
                new_resp = regen.get("final_response") or ""
                if new_resp:
                    final_response = new_resp
                    tools_called = sorted(set(tools_called) | set(regen.get("tools_called", [])))
                    log.info("VERIFY regenerated reply (%d chars)", len(final_response))
        except Exception as e:
            log.warning("verify/regenerate failed (%s) — giữ câu trả lời gốc.", e)

    # ── LỚP CHỐNG LỘ LỖI / THUẬT NGỮ NỘI BỘ ra cho khách ────────────
    # (vd câu trả lời lỡ chứa "product_id", "internal_error", "nhầm lẫn id").
    if agent_used == "knowledge_base_agent" and re.search(
        r"product_id|internal_error|nhầm lẫn.{0,25}id|\bid\b.{0,15}nhầm",
        final_response, re.IGNORECASE,
    ):
        try:
            log.warning("LEAK fix → regenerate | reply lộ thuật ngữ nội bộ")
            fix_note = SystemMessage(content=(
                "KIỂM DUYỆT NỘI BỘ: câu trả lời nháp vừa rồi LỘ thuật ngữ/lỗi nội bộ cho khách "
                "(vd 'product_id', lỗi hệ thống) — KHÔNG được phép. Hãy TỰ KHẮC PHỤC: gọi "
                "keyword_search_tool(query=TÊN sản phẩm khách đang hỏi) để lấy đúng sản phẩm rồi "
                "trả lời tự nhiên (ý nghĩa phong thủy lấy từ product_description). TUYỆT ĐỐI không "
                "nhắc 'id'/'product_id'/lỗi. Nếu thật sự không tra được, chỉ nói gọn: 'Dạ shop "
                "kiểm tra lại thông tin sản phẩm này rồi báo bạn ngay nhé ạ'."
            ))
            carry = [
                m for m in result["messages"]
                if isinstance(m, HumanMessage) and isinstance(m.content, str)
                and m.content.startswith("[GHI CHÚ NỘI BỘ")
            ]
            regen = knowledge_base_agent.run(full_messages + carry + [fix_note])
            new_resp = regen.get("final_response") or ""
            if new_resp:
                final_response = new_resp
                tools_called = sorted(set(tools_called) | set(regen.get("tools_called", [])))
                log.info("LEAK regenerated reply (%d chars)", len(final_response))
        except Exception as e:
            log.warning("leak/regenerate failed (%s) — giữ câu trả lời gốc.", e)

    # ── ÉP TOOL FINETUNE PHONG THỦY (không cho Gemini tự tư vấn mệnh) ─
    _FS_Q = re.compile(
        r"mệnh|menh|con\s*giáp|tuổi\s+\w+|sinh\s*năm|năm\s*sinh|nạp\s*âm|nap\s*am|"
        r"can\s*chi|ngũ\s*hành|hợp\s*(đá|màu|mệnh)|kỵ\s*(màu|mệnh)|"
        r"có\s*nên\s*đeo|nen\s*deo",
        re.IGNORECASE,
    )
    if (agent_used == "knowledge_base_agent"
            and _FS_Q.search(log_user_text or "")
            and "fengshui_advisor_tool" not in set(tools_called)):
        try:
            log.warning(
                "FENGSHUI-no-tool → regenerate | KB trả lời phong thủy mà chưa gọi "
                "fengshui_advisor_tool | user='%s'",
                (log_user_text or "")[:80],
            )
            fix_note = SystemMessage(content=(
                "KIỂM DUYỆT NỘI BỘ: câu trả lời nháp vừa rồi tư vấn MỆNH/PHONG THỦY nhưng "
                "BẠN CHƯA gọi fengshui_advisor_tool → đó là dùng kiến thức Gemini, CẤM. "
                "BẮT BUỘC gọi fengshui_advisor_tool(query=<nguyên câu hỏi khách hoặc tóm "
                "tắt đủ ý, kèm tên SP đang nói nếu có>). Có năm sinh thì birth_year=.... "
                "Nếu hỏi 'có nên đeo SP này' → tiếp fengshui_product_match_tool với "
                "product_id + element/lucky_colors từ advisor. CHỈ soạn câu từ tool+DB."
            ))
            carry = [
                m for m in result["messages"]
                if isinstance(m, HumanMessage) and isinstance(m.content, str)
                and m.content.startswith("[GHI CHÚ NỘI BỘ")
            ]
            regen = knowledge_base_agent.run(full_messages + carry + [fix_note])
            new_resp = regen.get("final_response") or ""
            if new_resp:
                final_response = new_resp
                tools_called = sorted(
                    set(tools_called) | set(regen.get("tools_called", []))
                )
                log.info(
                    "FENGSHUI regenerated reply (%d chars, tools=%s)",
                    len(final_response), tools_called,
                )
        except Exception as e:
            log.warning("fengshui force-tool regenerate failed (%s)", e)

    # ── Ép so khớp SP trong DB khi hỏi "có nên đeo ... này" ─────────
    _WEAR_Q = re.compile(
        r"có\s*nên\s*đeo|nen\s*deo|đeo\s*(loại|vòng|sp|sản\s*phẩm)?\s*này|"
        r"hợp\s*(với\s*)?(vòng|loại|sp)?\s*này",
        re.IGNORECASE,
    )
    if (agent_used == "knowledge_base_agent"
            and _FS_Q.search(log_user_text or "")
            and _WEAR_Q.search(log_user_text or "")
            and "fengshui_advisor_tool" in set(tools_called)
            and "fengshui_product_match_tool" not in set(tools_called)):
        try:
            log.warning(
                "FENGSHUI-no-match → regenerate | có mệnh+SP nhưng chưa "
                "fengshui_product_match_tool | user='%s'",
                (log_user_text or "")[:80],
            )
            fix_note = SystemMessage(content=(
                "KIỂM DUYỆT NỘI BỘ: bạn đã gọi fengshui_advisor_tool nhưng CHƯA gọi "
                "fengshui_product_match_tool trong khi khách hỏi có NÊN ĐEO SP/vòng "
                "ĐANG NÓI không. BẮT BUỘC: lấy product_id SP trong hội thoại (search/"
                "ảnh/seed), gọi fengshui_product_match_tool(product_id, element, "
                "lucky_colors, unlucky_colors từ advisor). Trả lời CHỈ theo "
                "match.verdict + product DB — CẤM tự bảo đá X thuộc hành Y."
            ))
            carry = [
                m for m in result["messages"]
                if isinstance(m, HumanMessage) and isinstance(m.content, str)
                and m.content.startswith("[GHI CHÚ NỘI BỘ")
            ]
            regen = knowledge_base_agent.run(full_messages + carry + [fix_note])
            new_resp = regen.get("final_response") or ""
            if new_resp:
                final_response = new_resp
                tools_called = sorted(
                    set(tools_called) | set(regen.get("tools_called", []))
                )
                log.info(
                    "FENGSHUI match regenerated (%d chars, tools=%s)",
                    len(final_response), tools_called,
                )
        except Exception as e:
            log.warning("fengshui match regenerate failed (%s)", e)

    # ── Ép size qua skills/FT: KB không được tự nhẩm cổ tay Xcm ─────
    _WRIST_SIZE_Q = re.compile(
        r"cổ\s*tay\s*\d|co\s*tay\s*\d|\d{1,2}(?:[.,]\d)?\s*cm|"
        r"đeo\s*size\s*gì|size\s*gì|bao\s*nhiêu\s*hạt",
        re.IGNORECASE,
    )
    if (agent_used == "knowledge_base_agent"
            and _WRIST_SIZE_Q.search(log_user_text or "")
            and re.search(r"\d{1,2}(?:[.,]\d)?\s*cm|cổ\s*tay", log_user_text or "", re.I)
            and "size_calculator_tool" not in set(tools_called)
            and "fengshui_finetune_size" not in set(tools_called)):
        try:
            log.warning(
                "SIZE-via-KB → handoff skills | user='%s'",
                (log_user_text or "")[:80],
            )
            sizing = skills_agent.run(full_messages)
            if sizing.get("final_response"):
                final_response = sizing["final_response"]
                tools_called = sorted(
                    set(tools_called) | set(sizing.get("tools_called") or [])
                )
                agent_used = "skills_agent"
                log.info(
                    "SIZE handoff skills done | tools=%s | reply=%d chars",
                    tools_called, len(final_response or ""),
                )
        except Exception as e:
            log.warning("SIZE handoff skills failed (%s)", e)

    # ── Ép search SP sau khi đã có mệnh (gợi ý "đeo vòng nào") ──────
    _RECOMMEND_Q = re.compile(
        r"nên\s*đeo\s*(loại\s*)?(vòng|đá)|đeo\s*(vòng|đá)\s*nào|hợp\s*(đá|vòng)\s*nào|"
        r"gợi\s*ý\s*(vòng|đá)|tư\s*vấn\s*(vòng|đá)|mua\s*vòng\s*gì",
        re.IGNORECASE,
    )
    _SEARCH_AFTER_FS = {
        "filter_search_tool", "keyword_search_tool", "semantic_search_tool",
    }
    if (agent_used == "knowledge_base_agent"
            and _RECOMMEND_Q.search(log_user_text or "")
            and "fengshui_advisor_tool" in set(tools_called)
            and not (set(tools_called) & _SEARCH_AFTER_FS)
            and "fengshui_product_match_tool" not in set(tools_called)):
        try:
            log.warning(
                "FENGSHUI-no-search → regenerate | đã có mệnh nhưng chưa filter/search SP | user='%s'",
                (log_user_text or "")[:80],
            )
            fix_note = SystemMessage(content=(
                "KIỂM DUYỆT NỘI BỘ: bạn đã gọi fengshui_advisor_tool nhưng CHƯA "
                "filter_search/keyword_search/semantic_search để lấy vòng/SP trong DB. "
                "BẮT BUỘC: dùng element + lucky_colors + suggested_filter_elements từ "
                "advisor → filter_search_tool(category='vòng tay', "
                "compatible_elements=mệnh và/hoặc colors=màu hợp, top_k=10). "
                "Rồi giới thiệu 5–6 SP THẬT (tên+giá+ảnh). CẤM chỉ nói mệnh mà không list SP DB."
            ))
            carry = [
                m for m in result["messages"]
                if isinstance(m, HumanMessage) and isinstance(m.content, str)
                and m.content.startswith("[GHI CHÚ NỘI BỘ")
            ]
            regen = knowledge_base_agent.run(full_messages + carry + [fix_note])
            new_resp = regen.get("final_response") or ""
            if new_resp:
                final_response = new_resp
                tools_called = sorted(
                    set(tools_called) | set(regen.get("tools_called", []))
                )
                log.info(
                    "FENGSHUI search regenerated (%d chars, tools=%s)",
                    len(final_response), tools_called,
                )
        except Exception as e:
            log.warning("fengshui search regenerate failed (%s)", e)

    # ── LỚP CHỐNG PHỦ NHẬN SẢN PHẨM MÀ CHƯA TRA DB ─────────────────
    # KB nói "shop không có / không bán / chưa kinh doanh X" NHƯNG lượt này KHÔNG gọi
    # tool search nào → là PHÁN ĐOÁN (dễ sai, vd "dầu gió" có trong DB). Ép tra lại.
    _SEARCH_TOOLS = {"keyword_search_tool", "semantic_search_tool", "filter_search_tool",
                     "image_search_tool", "get_product_detail_tool"}
    if (agent_used == "knowledge_base_agent"
            and re.search(r"(không|ko|chưa)\s*(có|bán|kinh\s*doanh)", final_response, re.IGNORECASE)
            and not (set(tools_called) & _SEARCH_TOOLS)):
        try:
            log.warning("DENY-no-search → regenerate | KB phủ nhận SP mà chưa search")
            fix_note = SystemMessage(content=(
                "KIỂM DUYỆT NỘI BỘ: câu trả lời nháp vừa rồi nói shop KHÔNG có/không bán một sản "
                "phẩm NHƯNG bạn CHƯA gọi tool search nào → đó là phán đoán, KHÔNG được phép. BẮT "
                "BUỘC gọi keyword_search_tool(query=<tên/loại sản phẩm khách vừa hỏi>) (kể cả tên "
                "nghe lạ như 'dầu gió', 'tinh dầu', 'than xông'...). Nếu tool có kết quả khớp → "
                "giới thiệu sản phẩm đó. CHỈ khi tool trả về RỖNG/không khớp mới được nói shop "
                "chưa có. TUYỆT ĐỐI không tự phán đoán."
            ))
            carry = [
                m for m in result["messages"]
                if isinstance(m, HumanMessage) and isinstance(m.content, str)
                and m.content.startswith("[GHI CHÚ NỘI BỘ")
            ]
            regen = knowledge_base_agent.run(full_messages + carry + [fix_note])
            new_resp = regen.get("final_response") or ""
            if new_resp:
                final_response = new_resp
                tools_called = sorted(set(tools_called) | set(regen.get("tools_called", [])))
                log.info("DENY regenerated reply (%d chars, tools=%s)", len(final_response), tools_called)
        except Exception as e:
            log.warning("deny/regenerate failed (%s) — giữ câu trả lời gốc.", e)

    # Chèn/gỡ lời mời Shopee bằng CODE (không giao cho LLM tự giác — nó chèn bừa).
    final_response = _apply_shopee_cta(log_user_text, final_response)

    # Persist to memory
    if log_user_text:
        try:
            log_turn(session_id, "user", log_user_text, agent_used=agent_used,
                     intent=agent_used, user_id=user_id, images=image_urls)
        except Exception as e:
            log.warning("log_turn(user) failed: %s", e)
    try:
        product_ref = _focused_product_refs(list(result["messages"]), final_response)
        log_turn(
            session_id,
            "assistant",
            final_response,
            agent_used=agent_used,
            intent=agent_used,
            tools_called=tools_called,
            user_id=user_id,
            product_ref=product_ref,
        )
    except Exception as e:
        log.warning("log_turn(assistant) failed: %s", e)

    # Audit trail: tool phong thủy / search đã đưa gì vào context trước khi Gemini soạn câu.
    try:
        from langchain_core.messages import ToolMessage as _TM
        _ft_tools = {
            "fengshui_advisor_tool", "fengshui_product_match_tool",
            "size_calculator_tool",
            "filter_search_tool", "keyword_search_tool", "semantic_search_tool",
            "get_product_detail_tool",
        }
        for m in result.get("messages") or []:
            if not isinstance(m, _TM):
                continue
            name = getattr(m, "name", "") or ""
            if name not in _ft_tools:
                continue
            content = m.content if isinstance(m.content, str) else str(m.content)
            # Ưu tiên log think + source nếu tool fengshui
            think_snip = ""
            source = ""
            try:
                payload = json.loads(content) if content.strip().startswith("{") else {}
                source = payload.get("source") or ""
                think_snip = (payload.get("ft_think") or "")[:1500]
            except Exception:
                payload = {}
            log.info(
                "┌─ TOOL → CÂU TRẢ LỜI ─ %s source=%s\n"
                "│ content (%d chars):\n%s\n"
                "│ ft_think:\n%s\n"
                "└─",
                name,
                source,
                len(content),
                content[:3500],
                think_snip if think_snip else "(không có / không phải FT tool)",
            )
    except Exception as e:
        log.warning("audit tool trail failed: %s", e)

    log.info(
        "└── RESPONSE agent=%s  tools=%s  reply=%d chars\n│ FINAL REPLY:\n%s",
        agent_used,
        tools_called,
        len(final_response or ""),
        (final_response or "")[:2500],
    )

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
    user_id: Optional[int] = None,
) -> dict:
    """Text-only chat."""
    return _invoke_graph(
        user_message_obj = HumanMessage(content=user_message),
        session_id       = session_id,
        history_override = history,
        log_user_text    = user_message,
        user_id          = user_id,
    )


def chat_with_image(
    user_message: str,
    images: Optional[list[dict]] = None,
    image_base64: Optional[str] = None,
    image_mime: str = "image/jpeg",
    session_id: str = "default",
    history: Optional[list[BaseMessage]] = None,
    user_id: Optional[int] = None,
    image_urls: Optional[list[str]] = None,
) -> dict:
    """Chat với một HOẶC nhiều ảnh đính kèm (tối đa MAX_IMAGES). Gemini xem ảnh natively.

    images: list các dict {"base64": str, "mime": str}. Tham số single
    image_base64/image_mime được giữ để tương thích ngược với caller cũ.
    """
    imgs = list(images or [])
    if not imgs and image_base64:
        imgs = [{"base64": image_base64, "mime": image_mime}]
    if not imgs:
        raise ValueError("chat_with_image cần ít nhất 1 ảnh")
    if len(imgs) > MAX_IMAGES:
        imgs = imgs[:MAX_IMAGES]
        log.warning("Nhận >%d ảnh, chỉ giữ %d ảnh đầu.", MAX_IMAGES, MAX_IMAGES)

    content: list[dict] = [
        {"type": "text", "text": user_message or "Bạn xem giúp mình các ảnh này"}
    ]
    for im in imgs:
        mime = im.get("mime") or "image/jpeg"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{im['base64']}"},
        })
    image_message = HumanMessage(content=content)
    # Lưu text gọn cho lịch sử; ẢNH được lưu riêng (image_urls) để render lại sau reload.
    log_text = (user_message or "").strip() or "(đã gửi ảnh)"
    return _invoke_graph(
        user_message_obj = image_message,
        session_id       = session_id,
        history_override = history,
        log_user_text    = log_text,
        user_id          = user_id,
        image_urls       = image_urls,
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
