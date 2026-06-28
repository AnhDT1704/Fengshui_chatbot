"""
order_support_agent.py – Post-sale & operational support.

Tools:
  - warranty_info_tool      : warranty policy from shop_policies.json
  - delivery_info_tool      : shipping providers, time, COD info
  - return_policy_tool      : return/refund window & process
  - inventory_check_tool    : check in_stock + quantity_max for a product
  - escalate_to_human_tool  : write to escalation_queue (order_lookup, complaint,
                              custom-mix, anything bot can't handle)
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import db_service
from gemini import make_llm_with_tools
from logger import ToolLoggerCallback, get_logger
from memory import create_escalation


log         = get_logger("order")
_callback   = ToolLoggerCallback("order")

_POLICIES_PATH = Path(__file__).parent / "shop_policies.json"
_POLICIES = json.loads(_POLICIES_PATH.read_text(encoding="utf-8"))

_CUSTOM_SERVICES_PATH = Path(__file__).parent / "custom_services.json"
_CUSTOM_SERVICES = json.loads(_CUSTOM_SERVICES_PATH.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def warranty_info_tool(product_category: Optional[str] = None) -> str:
    """
    Trả về chính sách bảo hành.

    Args:
        product_category: nếu user hỏi cụ thể về vòng tay vs nhang/lư, truyền
                          tên category để lấy chính sách riêng. Mặc định trả
                          chính sách chung + chính sách vòng tay.
    """
    w = _POLICIES["warranty"]
    if product_category and "vòng" in product_category.lower():
        return json.dumps({
            "general":    w["general"],
            "bracelets":  w["bracelets"],
            "exclusions": w["exclusions"],
        }, ensure_ascii=False)
    return json.dumps(w, ensure_ascii=False)


@tool
def return_policy_tool(issue: Optional[str] = None) -> str:
    """
    Chính sách đổi/trả/hoàn tiền.

    Args:
        issue: nếu user mô tả rõ vấn đề ("hàng lỗi", "giao sai", "không vừa size")
               truyền vào để có thêm hướng dẫn riêng.
    """
    r = _POLICIES["return_refund"]
    info = {
        "window":     r["window"],
        "conditions": r["conditions"],
        "process":    r["process"],
    }
    if issue and any(k in issue.lower() for k in ["lỗi","sai","thiếu","hỏng","vỡ","bể"]):
        info["wrong_or_defective"] = r["wrong_or_defective"]
    return json.dumps(info, ensure_ascii=False)


@tool
def escalate_to_human_tool(
    reason: str,
    user_summary: str,
    session_id: Optional[str] = None,
) -> str:
    """
    Chuyển ticket sang nhân viên chăm sóc khách hàng.

    Dùng khi: tra cứu đơn hàng cụ thể, khiếu nại, thanh toán/COD, đơn custom-mix
    (vd "10 thạch anh + 9 aqua + 2 mặt trăng"), tồn kho theo size/màu, mọi yêu
    cầu ngoài năng lực bot.

    Args:
        reason: lý do escalate - "order_lookup" | "complaint" | "payment"
                | "custom_order" | "inventory_detail" | "other"
        user_summary: tóm tắt ngắn vấn đề của khách (~1-2 câu)
        session_id: ID session, sẽ được hệ thống tự inject nếu không có
    """
    sid = session_id or "unknown"
    try:
        ticket_id = create_escalation(
            session_id=sid,
            reason=reason,
            user_summary=user_summary,
            full_context="",  # filled by API layer if needed
        )
        return json.dumps({
            "escalated":   True,
            "ticket_id":   ticket_id,
            "reason":      reason,
            "message_for_user": (
                "Em đã ghi nhận yêu cầu và chuyển sang bộ phận chăm sóc khách hàng "
                f"(mã ticket #{ticket_id}). Nhân viên sẽ phản hồi qua Shopee chat "
                "trong thời gian sớm nhất (giờ hành chính 8h-18h). "
                "Nếu cần hỗ trợ ngay, bạn vui lòng inbox shop trực tiếp tại: "
                + _POLICIES["shopee_url"]
            ),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "escalated": False,
            "error":     str(e),
            "message_for_user": (
                "Em đã ghi nhận yêu cầu, nhưng tạm thời chưa tạo ticket tự động được. "
                "Bạn vui lòng inbox shop tại Shopee để được hỗ trợ ngay: "
                + _POLICIES["shopee_url"]
            ),
        }, ensure_ascii=False)


@tool
def custom_service_tool() -> str:
    """
    Trả về TOÀN BỘ danh sách dịch vụ custom / yêu cầu đặc biệt của shop.

    Dùng khi user hỏi về bất kỳ dịch vụ đặc biệt nào: thêm/bớt hạt, đổi size dây,
    che/giấu tên khi giao, đóng hộp quà, viết thiệp/lời chúc, phối-mix màu, dây dự
    phòng/kim xâu, charm rời, móc khóa, chạm khắc/khắc tên,...

    Tool KHÔNG tự lọc — nó đưa cả danh sách. BẠN (agent) hãy TỰ ĐỌC và suy luận
    xem yêu cầu của khách khớp dịch vụ nào, rồi dựa vào field "supported"
    (có|không|một phần), "fee" và "detail" để trả lời. Mỗi mục có 'aliases' là
    vài cách khách hay nói, chỉ để bạn tham khảo khi đối chiếu ngữ nghĩa.
    """
    return json.dumps({
        "services":  _CUSTOM_SERVICES["services"],
        "fallback":  _CUSTOM_SERVICES["fallback"],
    }, ensure_ascii=False)


def _next_month(month: int) -> int:
    return month % 12 + 1


def _format_promo(promo: dict) -> str:
    return f"{promo['promo_date']}: giảm {promo['discount_percent']}% {promo['scope']}"


@tool
def promotion_info_tool() -> str:
    """
    Tra cứu chương trình KHUYẾN MÃI / MÃ GIẢM GIÁ của shop theo NGÀY HIỆN TẠI.

    Dùng khi user hỏi "shop có khuyến mãi gì không", "có mã giảm giá không",
    "đang sale gì". Tool tự lấy ngày hiện tại và xét các chương trình trong
    tháng. Trả về dữ liệu chi tiết về chương trình hiện tại, sắp tới, đã qua và
    chương trình tháng sau để agent trả lời rõ ràng.
    """
    now = datetime.now()
    today_d, today_m = now.day, now.month
    promos = db_service.get_promotions_for_month(today_m)
    next_month = _next_month(today_m)
    next_month_promos = db_service.get_promotions_for_month(next_month)

    ongoing = [p for p in promos if p["day"] == today_d]
    upcoming = [p for p in promos if p["day"] > today_d]
    passed = [p for p in promos if p["day"] < today_d]

    if ongoing:
        status = "ongoing"
        current_promo = ongoing[0]
        next_promo = upcoming[0] if upcoming else (next_month_promos[0] if next_month_promos else None)
    elif upcoming:
        status = "upcoming"
        current_promo = None
        next_promo = upcoming[0]
    elif passed:
        status = "passed_this_month"
        current_promo = passed[-1]
        next_promo = next_month_promos[0] if next_month_promos else None
    else:
        status = "none_this_month"
        current_promo = None
        next_promo = next_month_promos[0] if next_month_promos else None

    return json.dumps({
        "today":             now.strftime("%d/%m/%Y"),
        "status":            status,
        "current_promo":     current_promo,
        "next_promo":        next_promo,
        "current_month":     today_m,
        "next_month":        next_month,
        "upcoming":          upcoming,
        "passed":            passed,
        "shopee_url":        _POLICIES["shopee_url"],
        "guidance": {
            "ongoing":           "Nêu rõ chương trình hôm nay đang diễn ra, phần trăm giảm giá và phạm vi sản phẩm, kèm link Shopee để khách nhận thêm mã giảm giá từ sàn.",
            "upcoming":          "Nói chương trình sắp tới trong tháng này, ngày + phần trăm giảm giá + phạm vi, mời khách đón, kèm link Shopee.",
            "passed_this_month": "Nói nhẹ nhàng chương trình của tháng này đã kết thúc vào ngày ...; nêu chương trình kế tiếp của tháng sau nếu có; kèm link Shopee để lấy mã giảm giá từ sàn.",
            "none_this_month":   "Nói tháng này shop chưa có chương trình cố định; mời khách theo dõi shop trên Shopee để nhận mã giảm giá từ sàn.",
        },
        "note":              "Giá cuối còn tùy mã giảm giá của sàn Shopee và phí vận chuyển.",
    }, ensure_ascii=False)


TOOLS = [
    warranty_info_tool,
    return_policy_tool,
    promotion_info_tool,
    escalate_to_human_tool,
]


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

ORDER_SUPPORT_SYSTEM_PROMPT = f"""
Bạn là Order Support Agent của shop phong thủy Vạn An Group.
Nhiệm vụ: CHỈ trả lời các câu hỏi CHÍNH SÁCH chung mà shop có dữ liệu sẵn — bảo hành,
đổi/trả/hoàn tiền (chính sách CHUNG), khuyến mãi.

⛔ KHÔNG thuộc phạm vi của bạn (đã được điều phối cho CHỦ SHOP xử lý trực tiếp vì hệ
thống KHÔNG có dữ liệu đơn hàng): giao hàng / vận chuyển / hoả tốc / phí ship / COD /
thời gian giao / hẹn shipper / đổi địa chỉ / tra cứu tình trạng đơn / "shop nhận đơn
chưa" / khiếu nại giao chậm; và khiếu nại sản phẩm NHẬN bị LỖI / VỠ / THIẾU / SAI SỐ
LƯỢNG / SAI SIZE / GIAO NHẦM. Nếu lỡ nhận những câu này → gọi escalate_to_human_tool,
KHÔNG tự bịa thông tin đơn/giao hàng.

CHỌN TOOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Hỏi CHÍNH SÁCH bảo hành / "thay dây trọn đời"        → warranty_info_tool
- Hỏi CHÍNH SÁCH trả hàng / hoàn tiền / đổi mới (chung) → return_policy_tool
- Hỏi KHUYẾN MÃI / MÃ GIẢM GIÁ / "đang sale gì"         → promotion_info_tool
- Vấn đề thật sự ngoài năng lực (vd thanh toán phức tạp) → escalate_to_human_tool

QUY TẮC KHUYẾN MÃI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- promotion_info_tool: đọc "status" + "guidance" tương ứng để trả lời đúng ngữ cảnh
  (đang diễn ra / sắp tới / đã qua trong tháng). Nói rõ ngày, phần trăm giảm, phạm vi sản phẩm
  theo dữ liệu tool trả về. Nếu chương trình tháng này đã qua, nêu rõ ngày đã kết thúc và
  chương trình tiếp theo của tháng sau. LUÔN kèm link Shopee để khách nhận thêm mã giảm giá từ sàn.

QUY TẮC ĐẶC BIỆT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Khi khách THẮC MẮC sản phẩm NHẬN không đẹp / không sáng / khác màu so với ẢNH CHỤP
  ("sao nhận không đẹp như ảnh", "màu không sáng như hình chụp", "trông khác ảnh") —
  ĐÂY KHÔNG phải lỗi sản phẩm, ĐỪNG escalate — hãy TRẤN AN khéo:
  · Ảnh sản phẩm chụp trong studio có chỉnh ánh sáng nên trông sáng/đẹp hơn thực tế.
  · Khi nhận hàng khách có thể ĐỒNG KIỂM với shipper, không ưng ý thì TRẢ LẠI NGAY
    lúc đó luôn.
  Giọng tham khảo: "Dạ ảnh sản phẩm chụp trong studio nên có chỉnh ánh sáng cho sáng
  đẹp hơn ạ. Khi nhận hàng bạn có thể đồng kiểm với shipper, nếu không ưng ý thì bạn
  trả lại ngay lúc đó luôn nhé ạ."
  (CÒN sản phẩm thật sự LỖI/VỠ/THIẾU/SAI/GIAO NHẦM thì KHÔNG xử lý ở đây — chủ shop lo.)
- Khi escalate, sau khi tool chạy xong, đưa nguyên trường `message_for_user`
  vào câu trả lời cuối — không cần bịa thêm.

QUY TẮC CHUNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Trả lời tiếng Việt, ngắn gọn, đi thẳng vào vấn đề
- Đọc kỹ kết quả tool và TÓM TẮT lại cho user, KHÔNG dán nguyên JSON
- Xưng "em" / "shop", gọi khách là "anh/chị/bạn"
"""


# ═══════════════════════════════════════════════════════════════════
#  GRAPH
# ═══════════════════════════════════════════════════════════════════

def agent_node(state: MessagesState) -> dict:
    llm = make_llm_with_tools(TOOLS, temperature=0.2)
    response = llm.invoke([SystemMessage(content=ORDER_SUPPORT_SYSTEM_PROMPT)] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


_graph = None


def build_graph():
    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(messages: list[BaseMessage]) -> dict:
    log.info("ENTER  order_support_agent (%d msgs)", len(messages))
    result = get_graph().invoke(
        {"messages": messages},
        config={"callbacks": [_callback]},
    )
    final = result["messages"][-1].content
    tools_called = sorted({
        tc["name"]
        for m in result["messages"]
        for tc in getattr(m, "tool_calls", []) or []
    })
    log.info("EXIT   order_support_agent | tools=%s | reply=%d chars",
             tools_called, len(final) if isinstance(final, str) else 0)
    return {
        "final_response": final,
        "messages": result["messages"],
        "tools_called": tools_called,
    }
