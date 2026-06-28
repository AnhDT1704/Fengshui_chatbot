"""
skills_agent.py – Calc / advisory / external-knowledge agent.

Tools:
  - size_calculator_tool   : wrist_cm → bead size + bead count
  - web_search_tool        : SerpAPI fallback for items the shop does not sell
  - gift_advisor_tool      : structured gift suggestions by recipient + occasion

NOTE: feng-shui-by-birth-year advice (Can Chi → Nạp âm → mệnh + lucky colors)
lives in knowledge_base_agent.fengshui_advisor_tool, since it always chains into
product filtering. Routing of mệnh/tuổi/năm-sinh questions goes to KB agent.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

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
from gemini import make_llm_with_tools
from knowledge_base_agent import filter_search_tool, semantic_search_tool
from logger import ToolLoggerCallback, get_logger


log         = get_logger("skills")
_callback   = ToolLoggerCallback("skills")


# ═══════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════

# ─── Bracelet sizing constants ──────────────────────────────────────
# Đường kính 1 hạt theo size li/mm (cm). 1 li = 1 mm.
BEAD_DIAM_CM  = {6: 0.6, 8: 0.8, 10: 1.0}
# Số hạt MẶC ĐỊNH shop xâu cho cổ tay phổ thông (đều rơi Sinh/Lão).
DEFAULT_COUNT = {6: 26, 8: 21, 10: 18}
# Độ dư thoải mái mục tiêu so với cổ tay (cm) — shop thường nhắm ~+0.4.
TARGET_SLACK  = 0.4
# Biên (chiều dài vòng − cổ tay) khi sinh danh sách ứng viên, cm.
GEN_MIN, GEN_MAX = -0.6, 2.0
# Số hạt KHUYẾN NGHỊ: chiều dài phải ≥ cổ tay (cho rounding chừa -0.1) và ≤ +2.0cm.
REC_MIN, REC_MAX = -0.1, 2.0
# Phong thủy Sinh/Lão chỉ ưu tiên nếu vòng KHÔNG quá rộng (≤ +1.5cm so với cổ tay).
FS_MAX_OVER   = 1.5


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
        "count":       count,
        "length_cm":   length,
        "diff_cm":     round(length - wrist_cm, 1),
        "fengshui":    label,
        "is_fengshui": good,
        "needs_cut":   count < DEFAULT_COUNT[li],  # bớt hạt phải cắt dây
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
    d      = BEAD_DIAM_CM[li]
    ideal  = wrist_cm + TARGET_SLACK
    lo     = max(1, math.ceil((wrist_cm + GEN_MIN) / d))
    hi     = max(lo, math.floor((wrist_cm + GEN_MAX) / d))
    cands  = [_candidate(n, li, wrist_cm) for n in range(lo, hi + 1)]

    def closeness(c: dict) -> float:
        return abs(c["length_cm"] - ideal)

    # Số hạt vừa tay (chiều dài ≥ cổ tay, không quá +2cm)
    rec_pool = [c for c in cands if REC_MIN <= c["diff_cm"] <= REC_MAX]
    # Trong đó, số trúng Sinh/Lão mà không quá rộng
    fengshui = [c for c in rec_pool if c["is_fengshui"] and c["diff_cm"] <= FS_MAX_OVER]

    if fengshui:
        recommended = min(fengshui, key=closeness)
    elif rec_pool:
        recommended = min(rec_pool, key=closeness)   # ưu tiên vừa tay, bỏ phong thủy
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
        "li":             li,
        "bead_diam_cm":   d,
        "default_count":  DEFAULT_COUNT[li],
        "recommended":    recommended,
        "alternatives":   alternatives,
        "fengshui_fits":  bool(fengshui),  # False = đã hy sinh phong thủy để vừa tay
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

    Args:
        wrist_cm: chu vi cổ tay đo bằng dây mềm, cm (vd 14, 16.5, 18)
        li:       size hạt khách muốn — 6, 8 hoặc 10 (li = mm). Bỏ trống để tool tự chọn.
    """
    if wrist_cm <= 0:
        return json.dumps({"error": "Chu vi cổ tay phải > 0 cm"}, ensure_ascii=False)

    natural_li = recommend_li(wrist_cm)
    if li is None:
        li = natural_li
    elif li not in BEAD_DIAM_CM:
        return json.dumps(
            {"error": f"Size hạt {li} li không có. Shop có 6 / 8 / 10 li."},
            ensure_ascii=False,
        )

    result = compute_bracelet(wrist_cm, li)
    result.update({
        "wrist_cm":          wrist_cm,
        "chosen_li":         li,
        "natural_li":        natural_li,
        "li_matches_wrist":  li == natural_li,
        "spare_bead_note":   "Mỗi đơn shop tặng kèm 1 hạt dự phòng + dây thay + kim "
                             "xâu; khách đeo thấy chật/rộng có thể tự xâu thêm/bớt.",
        "fee_note":          "Thêm hạt cho vừa tay shop KHÔNG tính thêm phí.",
        "guidance":          "Nếu cần GIẢM hạt (tay nhỏ) thì phải cắt dây xâu lại → "
                             "HỎI khách muốn giảm mấy hạt rồi mới chốt.",
    })
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
        from serpapi import GoogleSearch  # type: ignore
        params = {
            "engine": "google",
            "q":      query,
            "hl":     "vi",
            "gl":     "vn",
            "num":    top_k,
            "api_key": api_key,
        }
        results = GoogleSearch(params).get_dict()
        organic = results.get("organic_results", [])[:top_k]
        compact = [
            {
                "title":   r.get("title"),
                "snippet": r.get("snippet"),
                "link":    r.get("link"),
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


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

SKILLS_SYSTEM_PROMPT = """
Bạn là Skills Agent của shop phong thủy Vạn An Group, chuyên xử lý câu hỏi cần
TÍNH TOÁN hoặc TƯ VẤN CHUYÊN MÔN.

CÁC TÌNH HUỐNG THƯỜNG GẶP & TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) HỎI SIZE VÒNG / SỐ HẠT THEO CỔ TAY
   ⚠️ BẮT BUỘC: hễ có SỐ ĐO CỔ TAY (cm) thì PHẢI gọi size_calculator_tool để tính.
   TUYỆT ĐỐI KHÔNG tự nhẩm số hạt hay tự gán cung Sinh-Lão-Bệnh-Tử (LLM nhẩm rất dễ sai).
   - Có chu vi cổ tay (cm) → gọi size_calculator_tool(wrist_cm)
   - Khách muốn 1 size hạt cụ thể (vd "mình muốn vòng 8 li") DÙ không khớp cổ tay
     → gọi size_calculator_tool(wrist_cm, li=8) để tính lại số hạt cho vừa
   - Hỏi đeo size nào / số hạt cho 1 SẢN PHẨM vòng tay (vòng shop luôn có 3 size 6/8/10
     li) → GỌI size_calculator_tool ĐỦ 3 LẦN: li=6, li=8, li=10 với cổ tay đã cho, để
     liệt kê đủ số hạt từng size cho khách so sánh. ĐỪNG vì ẢNH trông giống 1 size mà
     chỉ tính mỗi size đó — khách đang cần chọn. (Trong luồng có ẢNH bạn CHỈ cần TÍNH
     cho cả 3 size; KHÔNG cần nhận diện sản phẩm — bước sau sẽ trình bày card.)
   - Khách KHÔNG đo được cổ tay, chỉ cho CHIỀU CAO / CÂN NẶNG (và/hoặc "tay to/nhỏ")
     → ƯỚC LƯỢNG size li bằng SUY LUẬN (xem mục "ƯỚC LƯỢNG SIZE THEO VÓC DÁNG" bên dưới),
     KHÔNG bắt khách phải đo nếu họ không muốn.

ƯỚC LƯỢNG SIZE THEO VÓC DÁNG (khi không có số đo cổ tay)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Đây là cách shop làm thật: chỉ ÁNG CHỪNG, không có công thức cứng. Bạn hãy tự SUY LUẬN
size li hợp lý nhất dựa trên các yếu tố sau, rồi nói rõ đây là ước lượng:
  • GIỚI TÍNH là yếu tố mạnh: nam khung tay/cổ tay lớn hơn nữ cùng vóc → nghiêng size
    lớn hơn. Nếu CHƯA biết giới tính → HỎI "bạn là nam hay nữ ạ?" trước (shop luôn hỏi).
  • CHIỀU CAO + CÂN NẶNG: càng cao / càng nặng → cổ tay càng to → li lớn hơn; nhỏ nhắn,
    nhẹ cân → li nhỏ. Khách tự nói "tay nhỏ/ốm" hay "tay to" thì tin theo và điều chỉnh.
  • Đối chiếu các CA THẬT của shop để canh (3 size: 6 / 8 / 10 li):
      nữ 1m47/43kg → 6 li   ·  nữ 1m53/51kg → 6 li (thích to thì 8)
      nữ 1m66/54kg → 8 li   ·  nữ 1m70/50kg → 8 li (thích to thì 10)
      nam 1m70/50kg → 8 li  ·  nam 1m78/69kg → 10 li
    (Nhìn chung: nữ nhỏ nhắn ~6 li, nữ tầm trung ~8 li; nam mặc định ~8 li, nam cao to ~10 li.)

CÁCH TRẢ LỜI khi ước lượng theo vóc dáng:
  - Chốt 1 size li khuyến nghị + nói RÕ là ÁNG CHỪNG theo vóc dáng.
  - Báo số hạt MẶC ĐỊNH của size đó (6 li = 26 hạt ~15,6cm; 8 li = 21 hạt ~16,8cm;
    10 li = 18 hạt ~18cm) — vì chưa có số đo cổ tay nên dùng số hạt phổ thông.
  - LUÔN để khách chọn: "thích hạt nhỏ thì size kề dưới, thích to thì size kề trên".
  - GỢI Ý nhẹ: nếu muốn vừa khít nhất, khách đo cổ tay bằng dây mềm (cm) báo lại,
    shop tính số hạt cho chuẩn (gọi size_calculator_tool). KHÔNG ép.
  - Kết câu: shop tặng kèm 1 hạt dự phòng + dây thay + kim xâu, về tự chỉnh được nếu
    chật/rộng. (KHÔNG dùng size_calculator_tool cho bước ước lượng này — tool đó cần cm.)

   CÁCH ĐỌC KẾT QUẢ TOOL & TRẢ LỜI (rất quan trọng):
   - Chốt theo "recommended": nêu SỐ HẠT + size li + chiều dài (vd "29 hạt 6 li là
     17,4cm"). KHÔNG cần giải thích thuật toán.
   - "recommended.is_fengshui" = true → có thể nói trúng cung "fengshui" (Sinh/Lão).
     Nếu "fengshui_fits" = false → ĐỪNG nhắc phong thủy, chỉ nói số hạt này cho VỪA
     tay nhất (đã ưu tiên vừa tay hơn phong thủy, đúng tinh thần shop).
   - "alternatives": nếu có, gợi ý thêm 1 lựa chọn (vd "muốn đeo thoải mái hơn thì
     thêm 1 hạt là 30 hạt, trúng chữ Lão ạ").
   - "li_matches_wrist" = false (khách chọn size không khớp cổ tay) → vẫn chiều khách,
     tính theo li họ muốn, có thể nhẹ nhàng nói size tự nhiên là "natural_li" li.
   - GIẢM hạt ("recommended.needs_cut" = true / tay nhỏ): phải cắt dây xâu lại → ĐỪNG
     tự chốt, HỎI khách muốn giảm mấy hạt (đưa 1-2 phương án số hạt + chiều dài).
   - THÊM hạt (tay to): nói rõ shop xâu thêm cho vừa, KHÔNG tính thêm phí.
   - LUÔN kết câu bằng lưu ý: shop tặng kèm 1 hạt dự phòng (+ dây thay, kim xâu),
     đeo thấy chật/rộng có thể tự xâu thêm/bớt.

2) TƯ VẤN QUÀ TẶNG
   - Gọi gift_advisor_tool với info user cung cấp (recipient, occasion,...)
   - Đọc gợi ý → chain với filter_search_tool / semantic_search_tool

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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Sau khi tư vấn xong nhớ trình bày sản phẩm gợi ý kèm:
  - Tên + giá + ảnh (![tên](image_cover))
  - Lý do vì sao hợp (tương sinh / màu / chất liệu)
- Trả lời tiếng Việt, thân thiện, xưng "shop"
- Nếu khách hỏi tư vấn theo MỆNH / TUỔI / NĂM SINH → đó là việc của KB agent
  (không thuộc phạm vi Skills), trả lời ngắn gọn rồi để hệ thống điều phối lại
"""


# ═══════════════════════════════════════════════════════════════════
#  GRAPH
# ═══════════════════════════════════════════════════════════════════

# ─── Tính sẵn số hạt (deterministic) khi có số đo cổ tay ─────────────
# LLM (qua OpenRouter) gọi tool không ổn định, nhất là khi có ảnh → dễ "quên" tính
# rồi hỏi ngược khách. Phép tính số hạt là HÌNH HỌC THUẦN nên ta tính sẵn bằng code
# và TIÊM vào prompt; LLM chỉ việc trình bày → luôn có số đúng, khỏi phụ thuộc tool.
_WRIST_CM_RE = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*cm", re.IGNORECASE)
_LI_RE       = re.compile(r"(\d{1,2})\s*l[iy]\b", re.IGNORECASE)


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


def _precompute_sizing_note(text: str) -> Optional[str]:
    """Nếu câu khách có số đo cổ tay → tính sẵn số hạt cho 3 size (hoặc size chỉ định)."""
    wrist = _extract_wrist_cm(text)
    if wrist is None:
        return None
    m   = _LI_RE.search(text or "")
    lis = [int(m.group(1))] if (m and int(m.group(1)) in BEAD_DIAM_CM) else [6, 8, 10]

    lines = []
    for li in lis:
        r   = compute_bracelet(wrist, li)
        rec = r["recommended"]
        alt = "; ".join(
            f'{a["count"]} hạt={a["length_cm"]}cm ({a["fengshui"]})' for a in r["alternatives"]
        )
        lines.append(
            f'- {li} li: {rec["count"]} hạt = {rec["length_cm"]}cm, cung {rec["fengshui"]}'
            + (" (tay nhỏ → cần cắt dây bớt hạt)" if rec["needs_cut"] else "")
            + (f". Lựa chọn khác: {alt}" if alt else "")
        )
    body = "\n".join(lines)
    return (
        f"\n\n[SỐ HẠT ĐÃ TÍNH SẴN CHO CỔ TAY {wrist}cm — DÙNG ĐÚNG CÁC SỐ NÀY, KHÔNG tự "
        f"tính lại, KHÔNG sửa số, KHÔNG hỏi ngược khách 'muốn size mấy li']:\n{body}\n"
        "→ Trình bày ĐỦ các size ở trên cho khách so sánh (số hạt + chiều dài + cung Sinh/Lão). "
        "Nếu cần GIẢM hạt cho vừa thì hỏi khách muốn giảm mấy hạt. Kết bằng lưu ý shop tặng "
        "kèm 1 hạt dự phòng + dây + kim, khách tự chỉnh được."
    )


def agent_node(state: MessagesState) -> dict:
    messages = list(state["messages"])
    system   = SKILLS_SYSTEM_PROMPT
    note     = _precompute_sizing_note(_latest_human_text(messages))
    if note:
        system += note
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


def _direct_sizing_answer(text: str) -> Optional[str]:
    """Soạn THẲNG câu tư vấn số hạt bằng CODE khi có số đo cổ tay — không qua LLM.

    Lý do: LLM (Gemini Flash qua OpenRouter) hay phớt lờ chỉ thị/số liệu tính sẵn rồi
    hỏi ngược; mà phép tính là hình học thuần (deterministic) nên soạn trong code cho
    chắc. Trả None nếu câu không có số đo cổ tay (vd ước lượng theo chiều cao/cân nặng
    → để LLM reasoning).
    """
    wrist = _extract_wrist_cm(text)
    if wrist is None:
        return None
    m   = _LI_RE.search(text or "")
    lis = [int(m.group(1))] if (m and int(m.group(1)) in BEAD_DIAM_CM) else [6, 8, 10]

    lines, needs_cut_any = [], False
    for li in lis:
        r   = compute_bracelet(wrist, li)
        rec = r["recommended"]
        line = f'• Vòng {li} li: {rec["count"]} hạt (~{rec["length_cm"]}cm), trúng cung {rec["fengshui"]}'
        if r["alternatives"]:
            a = r["alternatives"][0]
            line += f' (hoặc {a["count"]} hạt ~{a["length_cm"]}cm, cung {a["fengshui"]})'
        needs_cut_any = needs_cut_any or rec["needs_cut"]
        lines.append(line)

    intro = f"Dạ với cổ tay {wrist}cm, shop tư vấn số hạt theo từng size như sau ạ:"
    tail  = "Bạn thích size nào thì shop xâu theo đúng size đó cho mình nhé ạ."
    if needs_cut_any:
        tail += (" Cổ tay bạn khá nhỏ nên một số size shop sẽ cắt bớt hạt cho vừa; "
                 "bạn muốn tăng/giảm thêm mấy hạt cứ báo shop ạ.")
    spare = ("Mỗi đơn shop tặng kèm 1 hạt dự phòng + dây thay + kim xâu, đeo thấy "
             "chật/rộng bạn có thể tự xâu thêm/bớt tại nhà ạ.")
    return f"{intro}\n" + "\n".join(lines) + f"\n{tail}\n{spare}"


def run(messages: list[BaseMessage]) -> dict:
    log.info("ENTER  skills_agent (%d msgs)", len(messages))

    # Có số đo cổ tay → soạn câu sizing bằng CODE (deterministic), bỏ qua LLM cho chắc.
    direct = _direct_sizing_answer(_latest_human_text(list(messages)))
    if direct is not None:
        log.info("EXIT   skills_agent | DIRECT sizing answer (deterministic) | reply=%d chars", len(direct))
        return {
            "final_response": direct,
            "messages": list(messages) + [AIMessage(content=direct)],
            "tools_called": ["size_calculator_tool"],
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
    log.info("EXIT   skills_agent | tools=%s | reply=%d chars",
             tools_called, len(final) if isinstance(final, str) else 0)
    return {
        "final_response": final,
        "messages": result["messages"],
        "tools_called": tools_called,
    }
