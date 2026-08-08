"""
skills_agent.py – Calc / advisory / external-knowledge agent.

Tools:
  - size_calculator_tool : wrist_cm → bead size + bead count
  - web_search_tool : SerpAPI fallback for items the shop does not sell
  - gift_advisor_tool : structured gift suggestions by recipient + occasion

NOTE: feng-shui-by-birth-year advice (Can Chi → Nạp âm → mệnh + lucky colors)
lives in knowledge_base_agent.fengshui_advisor_tool, since it always chains into
product filtering. Routing of mệnh/tuổi/năm-sinh questions goes to KB agent.
"""

from __future__ import annotations

import _bootstrap # noqa: F401

import json
import math
import os
import re
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# Re-export filter_search from KB so Skills can chain into product lookup
import fengshui_finetune_client as fengshui_ft
import runtime_settings
from gemini import make_llm_with_tools
from knowledge_base_agent import filter_search_tool, semantic_search_tool
from logger import ToolLoggerCallback, get_logger


log = get_logger("skills")
_callback = ToolLoggerCallback("skills")


# Đường kính 1 hạt theo size li/mm (cm). 1 li = 1 mm.
BEAD_DIAM_CM = {6: 0.6, 8: 0.8, 10: 1.0}
# Số hạt MẶC ĐỊNH shop xâu cho cổ tay phổ thông (đều rơi Sinh/Lão).
DEFAULT_COUNT = {6: 26, 8: 21, 10: 18}
# Độ dư thoải mái mục tiêu so với cổ tay (cm) — shop thường nhắm ~+0.4.
TARGET_SLACK = 0.4
# Biên (chiều dài vòng − cổ tay) khi sinh danh sách ứng viên, cm.
GEN_MIN, GEN_MAX = -0.6, 2.0
# Số hạt KHUYẾN NGHỊ: chiều dài phải ≥ cổ tay (cho rounding chừa -0.1) và ≤ +2.0cm.
REC_MIN, REC_MAX = -0.1, 2.0
# Phong thủy Sinh/Lão chỉ ưu tiên nếu vòng KHÔNG quá rộng (≤ +1.5cm so với cổ tay).
FS_MAX_OVER = 1.5


def _phong_thuy(count: int) -> tuple[str, bool]:
    """count → (tên cung Sinh-Lão-Bệnh-Tử, có_tốt). Đếm chia 4: 1=Sinh,2=Lão,3=Bệnh,0=Tử."""
    label = ["Tử", "Sinh", "Lão", "Bệnh"][count % 4]
    return label, (count % 4) in (1, 2)


def recommend_li(wrist_cm: float) -> int:
    """Chọn size hạt (li) tự nhiên theo cổ tay: ≤15.9→6, <18→8, ≥18→10."""
    if wrist_cm <= 15.9:
        return 6
    if wrist_cm < 18:
        return 8
    return 10


def _candidate(count: int, li: int, wrist_cm: float) -> dict:
    d = BEAD_DIAM_CM[li]
    length = round(count * d, 1)
    label, good = _phong_thuy(count)
    return {
        "count": count,
        "length_cm": length,
        "diff_cm": round(length - wrist_cm, 1),
        "fengshui": label,
        "is_fengshui": good,
        "needs_cut": count < DEFAULT_COUNT[li], # bớt hạt phải cắt dây
    }


def compute_bracelet(wrist_cm: float, li: int) -> dict:
    """
    Tính số hạt cho 1 size li sao cho VỪA cổ tay, ưu tiên Sinh/Lão khi vẫn vừa.

    Quy tắc (rút từ cách shop tư vấn thực tế):
      1. Chiều dài vòng phải ≥ cổ tay (vòng ngắn hơn cổ tay là chật, không đeo được),
         mục tiêu dư ~+0.4cm, tối đa +2.0cm.
      2. Nếu trong khoảng đó có số hạt KHÔNG quá rộng (≤ +1.5cm) MÀ trúng Sinh/Lão
         → ưu tiên nó (vd cổ tay 17 / 6 li → 29 hạt = 17,4cm cung Sinh).
      3. Nếu phải vượt mới trúng Sinh/Lão → BỎ phong thủy, chọn số hạt vừa tay nhất
         (vd cổ tay 18 / 8 li → 23 hạt = 18,4cm (Bệnh) thay vì 25 hạt = 20cm (Sinh, rộng)).
    """
    d = BEAD_DIAM_CM[li]
    ideal = wrist_cm + TARGET_SLACK
    lo = max(1, math.ceil((wrist_cm + GEN_MIN) / d))
    hi = max(lo, math.floor((wrist_cm + GEN_MAX) / d))
    cands = [_candidate(n, li, wrist_cm) for n in range(lo, hi + 1)]

    def closeness(c: dict) -> float:
        return abs(c["length_cm"] - ideal)

    # Số hạt vừa tay (chiều dài ≥ cổ tay, không quá +2cm)
    rec_pool = [c for c in cands if REC_MIN <= c["diff_cm"] <= REC_MAX]
    # Trong đó, số trúng Sinh/Lão mà không quá rộng
    fengshui = [c for c in rec_pool if c["is_fengshui"] and c["diff_cm"] <= FS_MAX_OVER]

    if fengshui:
        recommended = min(fengshui, key=closeness)
    elif rec_pool:
        recommended = min(rec_pool, key=closeness) # ưu tiên vừa tay, bỏ phong thủy
    else:
        recommended = min(cands, key=closeness) if cands else _candidate(
            max(1, round(ideal / d)), li, wrist_cm
        )

    # Lựa chọn thay thế trúng Sinh/Lão gần nhất (1 chật hơn / 1 rộng hơn để khách chọn,
    # vd "thêm 1 hạt là Lão đeo thoải mái hơn", hoặc "bớt 1 hạt cho ôm tay").
    alternatives = sorted(
        (c for c in cands
         if c["is_fengshui"] and c["count"] != recommended["count"]),
        key=closeness,
    )[:2]

    return {
        "li": li,
        "bead_diam_cm": d,
        "default_count": DEFAULT_COUNT[li],
        "recommended": recommended,
        "alternatives": alternatives,
        "fengshui_fits": bool(fengshui), # False = đã hy sinh phong thủy để vừa tay
    }


def _size_result_from_code(wrist_cm: float, li: Optional[int]) -> dict:
    natural_li = recommend_li(wrist_cm)
    chosen = natural_li if li is None else li
    if chosen not in BEAD_DIAM_CM:
        return {"error": f"Size hạt {chosen} li không có. Shop có 6 / 8 / 10 li."}
    result = compute_bracelet(wrist_cm, chosen)
    result.update({
        "wrist_cm": wrist_cm,
        "chosen_li": chosen,
        "natural_li": natural_li,
        "li_matches_wrist": chosen == natural_li,
        "spare_bead_note": "Mỗi đơn shop tặng kèm 1 hạt dự phòng + dây thay + kim "
                             "xâu; khách đeo thấy chật/rộng có thể tự xâu thêm/bớt.",
        "fee_note": "Thêm hạt cho vừa tay shop KHÔNG tính thêm phí.",
        "guidance": "Nếu cần GIẢM hạt (tay nhỏ) thì phải cắt dây xâu lại → "
                             "HỎI khách muốn giảm mấy hạt rồi mới chốt.",
        "source": "code",
    })
    return result


def _size_result_from_ft(wrist_cm: float, li: Optional[int], data: dict) -> dict:
    """Map JSON model (task=size) → shape size_calculator_tool (recommended/...)."""
    natural_li = recommend_li(wrist_cm)
    bead_li = data.get("bead_size_li") or data.get("chosen_li") or li or natural_li
    try:
        bead_li = int(bead_li)
    except Exception:
        bead_li = natural_li
    if bead_li not in BEAD_DIAM_CM:
        bead_li = natural_li

    count = data.get("bead_count") or data.get("count")
    length = data.get("length_cm")
    slack = data.get("slack_cm")
    fengshui = data.get("fengshui")
    is_good = data.get("is_fengshui_good")
    if is_good is None:
        is_good = data.get("is_fengshui")
    fits = data.get("fengshui_fits")

    if count is None:
        # Model thiếu số hạt → fallback code
        out = _size_result_from_code(wrist_cm, li)
        out["source"] = "code_fallback"
        out["ft_raw"] = data
        return out

    try:
        count = int(count)
    except Exception:
        out = _size_result_from_code(wrist_cm, li)
        out["source"] = "code_fallback"
        return out

    d = BEAD_DIAM_CM[bead_li]
    if length is None:
        length = round(count * d, 1)
    if slack is None:
        slack = round(float(length) - wrist_cm, 1)
    if fengshui is None:
        fengshui, is_good = _phong_thuy(count)
    if is_good is None:
        is_good = (count % 4) in (1, 2)
    if fits is None:
        fits = bool(is_good) and float(slack) <= FS_MAX_OVER

    recommended = {
        "count": count,
        "length_cm": float(length),
        "diff_cm": float(slack),
        "fengshui": fengshui,
        "is_fengshui": bool(is_good),
        "needs_cut": count < DEFAULT_COUNT[bead_li],
    }
    # Bổ sung alternatives từ code (model CoT thường không trả list alternatives)
    code_full = compute_bracelet(wrist_cm, bead_li)

    return {
        "li": bead_li,
        "bead_diam_cm": d,
        "default_count": DEFAULT_COUNT[bead_li],
        "recommended": recommended,
        "alternatives": code_full.get("alternatives", []),
        "fengshui_fits": bool(fits),
        "wrist_cm": wrist_cm,
        "chosen_li": bead_li,
        "natural_li": data.get("natural_li") or natural_li,
        "li_matches_wrist": bead_li == natural_li,
        "spare_bead_note": "Mỗi đơn shop tặng kèm 1 hạt dự phòng + dây thay + kim "
                           "xâu; khách đeo thấy chật/rộng có thể tự xâu thêm/bớt.",
        "fee_note": "Thêm hạt cho vừa tay shop KHÔNG tính thêm phí.",
        "guidance": "Nếu cần GIẢM hạt (tay nhỏ) thì phải cắt dây xâu lại → "
                           "HỎI khách muốn giảm mấy hạt rồi mới chốt.",
        "source": "fengshui_finetune",
        "ft_think": data.get("_think") or "",
        "ft_model_json": {
            k: data[k] for k in (
                "task", "bead_count", "bead_size_li", "natural_li", "fengshui",
                "is_fengshui_good", "fengshui_fits", "length_cm", "slack_cm",
                "wrist_cm",
            ) if k in data
        },
        "ft_raw": {
            k: data[k] for k in (
                "task", "bead_count", "bead_size_li", "fengshui",
                "is_fengshui_good", "fengshui_fits", "length_cm", "slack_cm",
            ) if k in data
        },
    }


@tool
def size_calculator_tool(wrist_cm: float, li: Optional[int] = None) -> str:
    """
    Tính SỐ HẠT vòng tay theo chu vi cổ tay (cm), cân bằng giữa VỪA TAY và phong
    thủy Sinh-Lão-Bệnh-Tử. Dùng cho cả 2 tình huống:
      - Khách chỉ cho cổ tay → để li=None, tool tự đề xuất size hạt phù hợp.
      - Khách muốn 1 size hạt cụ thể (vd "mình muốn 8 li") dù không khớp cổ tay
        → truyền li=8, tool tính lại số hạt cho vừa.

    Trả về JSON: size hạt chọn (và size tự nhiên nếu khác), số hạt khuyến nghị
    kèm chiều dài + cung phong thủy + có phải cắt dây không, và các lựa chọn thay thế.

   QUY TẮC ưu tiên: số hạt phải VỪA cổ tay (lệch ≤ ~2cm); chỉ chọn số trúng
    Sinh/Lão khi vẫn vừa, nếu không thì ưu tiên vừa tay.

    Chế độ admin UI (runtime_settings.size_mode):
      - code (mặc định): compute_bracelet
      - finetune: model FT; lỗi → code_fallback

    Args:
        wrist_cm: chu vi cổ tay đo bằng dây mềm, cm (vd 14, 16.5, 18)
        li: size hạt khách muốn — 6, 8 hoặc 10 (li = mm). Bỏ trống để tool tự chọn.
    """
    if wrist_cm <= 0:
        return json.dumps({"error": "Chu vi cổ tay phải > 0 cm"}, ensure_ascii=False)

    if li is not None and li not in BEAD_DIAM_CM:
        return json.dumps(
            {"error": f"Size hạt {li} li không có. Shop có 6 / 8 / 10 li."},
            ensure_ascii=False,
        )

    chosen = recommend_li(wrist_cm) if li is None else li
    result = _size_one(wrist_cm, chosen)
    rec = result.get("recommended") or {}
    log.info(
        "size_calculator_tool mode=%s source=%s wrist=%.2f li=%s → count=%s",
        runtime_settings.get_size_mode(),
        result.get("source"),
        wrist_cm,
        result.get("chosen_li"),
        rec.get("count"),
    )
    return json.dumps(result, ensure_ascii=False)


@tool
def web_search_tool(query: str, top_k: int = 5) -> str:
    """
    Tìm thông tin trên Google qua SerpAPI. Dùng cho:
      - Sản phẩm shop KHÔNG bán (vd: "đá mặt trăng" mà DB không có)
      - Câu hỏi kiến thức chung ngoài phạm vi sản phẩm
      - Tin tức / xu hướng phong thủy
    Lưu ý: khi dùng, PHẢI nói rõ với user rằng đây là thông tin tham khảo từ web,
    không phải sản phẩm của shop.

    Args:
        query: Câu truy vấn tiếng Việt hoặc tiếng Anh
        top_k: Số kết quả tối đa (mặc định 5)
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return json.dumps({
            "error": "Tool web_search chưa được cấu hình SERPAPI_KEY",
            "fallback": "Hãy trả lời dựa trên kiến thức chung và nói rõ shop sẽ kiểm tra lại.",
        }, ensure_ascii=False)

    try:
        from serpapi import GoogleSearch # type: ignore
        params = {
            "engine": "google",
            "q": query,
            "hl": "vi",
            "gl": "vn",
            "num": top_k,
            "api_key": api_key,
        }
        results = GoogleSearch(params).get_dict()
        organic = results.get("organic_results", [])[:top_k]
        compact = [
            {
                "title": r.get("title"),
                "snippet": r.get("snippet"),
                "link": r.get("link"),
            }
            for r in organic
        ]
        return json.dumps({"results": compact}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"web_search failed: {e}"}, ensure_ascii=False)


TOOLS = [
    size_calculator_tool,
    web_search_tool,
    # Chained from KB so Skills can finalize a recommendation:
    filter_search_tool,
    semantic_search_tool,
]


SKILLS_SYSTEM_PROMPT = """
Bạn là Skills Agent của shop phong thủy Vạn An Group, chuyên xử lý câu hỏi cần
TÍNH TOÁN hoặc TƯ VẤN CHUYÊN MÔN.

CÁC TÌNH HUỐNG THƯỜNG GẶP & TOOLS

1) HỎI SIZE VÒNG / SỐ HẠT THEO CỔ TAY
   CÓ SỐ ĐO CỔ TAY (cm): hệ thống TỰ gọi model finetune phong thủy (size_calculator /
   pipeline DIRECT) — BẠN (Gemini) CẤM tự nhẩm số hạt / cung Sinh-Lão-Bệnh-Tử / bảng
   26-21-18 hạt. Chỉ trình bày số từ tool nếu được gọi.
   - size_calculator_tool(wrist_cm) hoặc (wrist_cm, li=6|8|10) → nguồn số liệu duy nhất.
   - Cần đủ 3 size → tool 3 lần li=6,8,10 (hoặc để pipeline direct lo).

   KHÔNG CÓ cm (chỉ cao/cân/tay to-nhỏ):
   - Chỉ GỢI Ý size li (6/8/10) theo vóc dáng, nói rõ là ÁNG CHỪNG.
   - CẤM bịa số hạt / chiều dài / cung Sinh-Lão. Mời đo cổ tay (cm) để shop tính chuẩn
     bằng model/tool.
   - Gợi ý li tham khảo (không chốt số hạt): nữ nhỏ ~6, nữ TB ~8; nam ~8, nam to ~10.
   - Kết: hạt dự phòng + dây + kim; đo cm để tính chính xác.

   CÁCH ĐỌC KẾT QUẢ TOOL (khi có):
   - Chỉ dùng field recommended / alternatives / source từ tool.
   - source=fengshui_finetune → số từ model FT; code_fallback → công thức shop.
   - needs_cut → hỏi khách giảm hạt; luôn nhắc hạt dự phòng.

2) TƯ VẤN QUÀ TẶNG
   - Gọi gift_advisor_tool với info user cung cấp (recipient, occasion,...)
   - BẮT BUỘC chain filter_search_tool / semantic_search_tool để lấy SẢN PHẨM THẬT trong kho
     rồi mới giới thiệu. TUYỆT ĐỐI KHÔNG tự liệt kê loại vật phẩm/đá từ kiến thức (vd "tỳ hưu,
     tượng Phật, thiềm thừ, gỗ huyết long..") khi CHƯA search — có thể shop không bán.
   - KHÔNG nói "shop chưa có chức năng tư vấn quà tặng" (có gift_advisor_tool + search).

3) HỎI SẢN PHẨM NGOÀI PHẠM VI SHOP
   - Đầu tiên thử semantic_search_tool xem shop có không
   - Nếu không có → web_search_tool, NHƯNG phải nói rõ "đây là thông tin tham
     khảo từ web, sản phẩm này hiện shop chưa bán"

4) HỎI CÁCH THẮT LẠI VÒNG / ĐAN DÂY
   - Khách hỏi cách thắt lại vòng, thắt nút dây, đan/xâu lại dây, buộc dây thế nào...
     → GỬI link video hướng dẫn, KHÔNG cần mô tả dài dòng. Trả lời đúng dạng:
       Bạn xem video ở đây nhé ạ:
       https://down-zl-vn.vod.susercontent.com/api/v4/11110105/mms/vn-11110105-6khw3-m4u7yfv856c751.16000081736504660.mp4
QUY TẮC CHUNG
- CẤM BỊA danh sách vòng/SP/vật phẩm (tên, giá, URL ảnh, loại đá) khi CHƯA có kết quả search
  tool. TUYỆT ĐỐI không liệt kê "tỳ hưu, thiềm thừ, gỗ huyết long, thạch anh xanh, mã não
  xanh..." từ kiến thức của bạn — có thể shop KHÔNG bán. Muốn gợi ý SP → BẮT BUỘC gọi
  filter_search_tool / semantic_search_tool trước, chỉ nêu SP CÓ trong kết quả tool.
  Câu "thích trồng cây / thích màu X nên đeo gì" → search rồi mới gợi ý (hoặc để KB hỏi mệnh);
  Skills KHÔNG tự nghĩ ra 1 SP rồi chốt.
- Shop CHỈ bán ONLINE (Shopee), KHÔNG có cửa hàng vật lý. TUYỆT ĐỐI KHÔNG mời khách "ghé
  thăm cửa hàng / ghé qua shop / đến cửa hàng / tới xem trực tiếp / ghé website". Muốn xem thêm
  → mời xem trên Shopee của shop.
- Sau khi CÓ tool search trả SP: tên + giá + ảnh (![tên](image_cover) URL thật từ tool).
- Trả lời tiếng Việt, thân thiện, xưng "shop"
- Nếu khách hỏi tư vấn theo MỆNH / TUỔI / NĂM SINH → việc của KB agent
"""


# LLM (qua OpenRouter) gọi tool không ổn định, nhất là khi có ảnh → dễ "quên" tính
# rồi hỏi ngược khách. Phép tính số hạt là HÌNH HỌC THUẦN nên ta tính sẵn bằng code
# và TIÊM vào prompt; LLM chỉ việc trình bày → luôn có số đúng, khỏi phụ thuộc tool.
_WRIST_CM_RE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*cm", re.IGNORECASE)
_LI_RE = re.compile(r"(\d{1,2})\s*l[iy]\b", re.IGNORECASE)


def _latest_human_text(messages: list) -> str:
    from langchain_core.messages import HumanMessage
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    p.get("text", "") for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return ""
    return ""


def _extract_wrist_cm(text: str) -> Optional[float]:
    """Lấy số đo CỔ TAY (cm) từ câu khách. Chỉ nhận giá trị hợp lý 8–25cm."""
    for m in _WRIST_CM_RE.finditer(text or ""):
        v = float(m.group(1).replace(",", "."))
        if 8 <= v <= 25:
            return v
    return None


def _size_one(wrist: float, li: int) -> dict:
    """Một size theo runtime_settings.size_mode:
      - code (mặc định): compute_bracelet
      - finetune: model FT phong thủy; lỗi → code_fallback
    """
    mode = runtime_settings.get_size_mode()
    if mode == "finetune" and fengshui_ft.USE_FENGSHUI_FT:
        ft = fengshui_ft.ask_size(wrist, li)
        if ft.get("ok") and isinstance(ft.get("data"), dict) and ft["data"]:
            data = dict(ft["data"])
            data["_think"] = ft.get("think") or ""
            result = _size_result_from_ft(wrist, li, data)
            if result.get("recommended") and result.get("source") != "code_fallback":
                return result
            log.warning("FT size li=%s parse kém → code fallback", li)
        else:
            log.warning("FT size li=%s API lỗi → code fallback: %s", li, ft.get("error"))
        out = _size_result_from_code(wrist, li)
        out["source"] = "code_fallback"
        return out
    if mode == "finetune" and not fengshui_ft.USE_FENGSHUI_FT:
        log.warning("size_mode=finetune nhưng FENGSHUI_API_URL trống → dùng code")
    out = _size_result_from_code(wrist, li)
    out["source"] = "code"
    return out


def agent_node(state: MessagesState) -> dict:
    messages = list(state["messages"])
    system = SKILLS_SYSTEM_PROMPT
    # KHÔNG tiêm số hạt CODE vào prompt nữa — khi có cm, run() dùng FT/direct.
    # temperature=0 để hành vi ổn định hơn (đỡ lúc tính lúc hỏi ngược).
    llm = make_llm_with_tools(TOOLS, temperature=0)
    response = llm.invoke([SystemMessage(content=system)] + messages)
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


def _direct_sizing_answer(text: str) -> Optional[tuple[str, list[str], list[dict]]]:
    """Soạn câu tư vấn size khi có số đo cổ tay — ƯU TIÊN model finetune phong thủy.

    Không để LLM Gemini tự nhẩm. Trả (reply, tools_called, per_li_results) hoặc None
    nếu không có cm (ước lượng vóc dáng → để LLM, vẫn không nhẩm công thức cứng).
    """
    wrist = _extract_wrist_cm(text)
    if wrist is None:
        return None
    m = _LI_RE.search(text or "")
    lis = [int(m.group(1))] if (m and int(m.group(1)) in BEAD_DIAM_CM) else [6, 8, 10]

    lines, needs_cut_any = [], False
    sources = []
    results = []
    for li in lis:
        r = _size_one(wrist, li)
        results.append(r)
        sources.append(r.get("source") or "code")
        rec = r.get("recommended") or {}
        feng = rec.get("fengshui") or "?"
        cnt = rec.get("count")
        length = rec.get("length_cm")
        # Không dùng dấu ~ (Markdown/UI hay hiểu thành gạch ngang strikethrough).
        line = f'• Vòng {li} li: {cnt} hạt (khoảng {length}cm)'
        if rec.get("is_fengshui"):
            line += f', trúng cung {feng}'
        elif feng and feng != "?":
            line += f', cung {feng}'
        alts = r.get("alternatives") or []
        if alts:
            a = alts[0]
            line += (
                f' (hoặc {a.get("count")} hạt khoảng {a.get("length_cm")}cm, '
                f'cung {a.get("fengshui")})'
            )
        needs_cut_any = needs_cut_any or bool(rec.get("needs_cut"))
        lines.append(line)

    src_set = set(sources)
    if src_set == {"fengshui_finetune"}:
        src_note = "model finetune phong thủy"
        tools = ["size_calculator_tool", "fengshui_finetune_size"]
    elif "fengshui_finetune" in src_set:
        src_note = "model finetune phong thủy (+ fallback code một phần)"
        tools = ["size_calculator_tool", "fengshui_finetune_size"]
    else:
        src_note = "công thức shop (code)"
        tools = ["size_calculator_tool"]

    log.info(
        "[TIMING] SIZE_PIPELINE | mode=%s | wrist_cm=%s | lis=%s | sources=%s | note=%s",
        runtime_settings.get_size_mode(), wrist, lis, sources, src_note,
    )

    intro = f"Dạ với cổ tay {wrist}cm, shop tư vấn số hạt theo từng size như sau ạ:"
    tail = "Bạn thích size nào thì shop xâu theo đúng size đó cho mình nhé ạ."
    if needs_cut_any:
        tail += (" Cổ tay bạn khá nhỏ nên một số size shop sẽ cắt bớt hạt cho vừa; "
                 "bạn muốn tăng/giảm thêm mấy hạt cứ báo shop ạ.")
    spare = ("Mỗi đơn shop tặng kèm 1 hạt dự phòng + dây thay + kim xâu, đeo thấy "
             "chật/rộng bạn có thể tự xâu thêm/bớt tại nhà ạ.")
    reply = f"{intro}\n" + "\n".join(lines) + f"\n{tail}\n{spare}"
    return reply, tools, results


def run(messages: list[BaseMessage]) -> dict:
    log.info("ENTER skills_agent (%d msgs)", len(messages))

    # Có số đo cổ tay → size qua FINETUNE phong thủy (hoặc code nếu tắt/lỗi FT),
    # KHÔNG để Gemini tự nhẩm / không inject bảng 26-21-18 vào prompt.
    direct = _direct_sizing_answer(_latest_human_text(list(messages)))
    if direct is not None:
        reply, tools, _results = direct
        log.info(
            "EXIT skills_agent | DIRECT sizing (source tools=%s) | reply=%d chars",
            tools, len(reply),
        )
        return {
            "final_response": reply,
            "messages": list(messages) + [AIMessage(content=reply)],
            "tools_called": tools,
        }

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
    log.info("EXIT skills_agent | tools=%s | reply=%d chars",
             tools_called, len(final) if isinstance(final, str) else 0)
    return {
        "final_response": final,
        "messages": result["messages"],
        "tools_called": tools_called,
    }
