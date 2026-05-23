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
def delivery_info_tool(area: Optional[str] = None) -> str:
    """
    Trả về thông tin giao hàng - đơn vị vận chuyển, thời gian, COD, phí ship.

    Args:
        area: tỉnh/thành phố nếu user hỏi cụ thể (vd "Đà Nẵng", "Hà Nội", "HCM")
    """
    d = _POLICIES["delivery"]
    info = {
        "providers": d["providers"],
        "cod":       d["cod"],
        "fee":       d["fee"],
        "fast_ship": d["fast_ship"],
    }
    if area:
        a = area.lower()
        if "hcm" in a or "hồ chí minh" in a or "sài gòn" in a:
            info["time"] = d["time_hcm"]
        elif "hà nội" in a or "ha noi" in a or "hanoi" in a:
            info["time"] = d["time_hanoi"]
        else:
            info["time"] = d["time_other"]
            info["area_note"] = f"Khu vực {area} thuộc nhóm 'các tỉnh khác'"
    else:
        info["time_all"] = {
            "hcm":   d["time_hcm"],
            "hanoi": d["time_hanoi"],
            "other": d["time_other"],
        }
    return json.dumps(info, ensure_ascii=False)


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
def inventory_check_tool(product_id: int) -> str:
    """
    Kiểm tra tình trạng tồn kho của một sản phẩm cụ thể.

    Lưu ý: hệ thống chỉ biết "còn / hết" ở mức sản phẩm tổng thể + số lượng đã
    nhập kho gần nhất (quantity_max). KHÔNG biết tồn kho theo size/màu cụ thể.
    Nếu user hỏi tồn kho size/màu chi tiết → trả lời "shop cần kiểm tra với
    nhân viên kho" và gọi escalate_to_human_tool.

    Args:
        product_id: Mã sản phẩm cần check
    """
    product = db_service.get_product_by_id(product_id)
    if product is None:
        return json.dumps({"error": f"Không tìm thấy product_id={product_id}"}, ensure_ascii=False)

    qmax = getattr(product, "quantity_max", None) or 0
    qmin = getattr(product, "quantity_min", None) or 0
    in_stock = bool(product.in_stock)

    status_text = "Hết hàng" if not in_stock else (
        "Còn nhiều (đã nhập >99K)" if qmax >= 99999 else (
            f"Còn hàng (khoảng {qmin:,} - {qmax:,})" if qmin != qmax
            else f"Còn hàng (~{qmax:,})"
        )
    )

    return json.dumps({
        "product_id":   product_id,
        "name":         product.name,
        "in_stock":     in_stock,
        "quantity_min": qmin,
        "quantity_max": qmax,
        "status_text":  status_text,
        "note":         "Hệ thống không có breakdown theo size/color. Nếu user cần thông tin chi tiết, escalate.",
    }, ensure_ascii=False)


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


TOOLS = [
    warranty_info_tool,
    delivery_info_tool,
    return_policy_tool,
    inventory_check_tool,
    escalate_to_human_tool,
]


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

ORDER_SUPPORT_SYSTEM_PROMPT = f"""
Bạn là Order Support Agent của shop phong thủy Vạn An Group.
Nhiệm vụ: trả lời câu hỏi sau bán hàng và vận hành đơn hàng.

CHỌN TOOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Hỏi bảo hành / "thay dây trọn đời" / lỗi sản phẩm   → warranty_info_tool
- Hỏi giao hàng / ship / phí ship / COD / hoả tốc     → delivery_info_tool
- Hỏi trả hàng / hoàn tiền / đổi mới                  → return_policy_tool
- Hỏi "sản phẩm X còn hàng không"                     → inventory_check_tool
- Các case BẮT BUỘC escalate → escalate_to_human_tool:
   • Tra cứu trạng thái đơn hàng cụ thể ("đơn của em đến đâu rồi")
   • Khiếu nại sản phẩm lỗi / sai mẫu / thiếu
   • Custom mix theo yêu cầu (vd "10 thạch anh + 9 aqua")
   • Hỏi tồn kho theo size/màu CỤ THỂ (hệ thống không có data đó)
   • Vấn đề thanh toán, mã giảm giá, hoàn tiền chi tiết
   • Bất kỳ yêu cầu ngoài năng lực bot

QUY TẮC ĐẶC BIỆT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Khi user hỏi địa chỉ cửa hàng / ghé trực tiếp → KHÔNG cung cấp địa chỉ vật lý.
  Trả lời: "Vạn An Group chỉ bán online qua Shopee, mời bạn truy cập gian hàng:
  {_POLICIES['shopee_url']}"
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
