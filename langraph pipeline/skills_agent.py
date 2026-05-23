"""
skills_agent.py – Calc / advisory / external-knowledge agent.

Tools:
  - size_calculator_tool   : wrist_cm → bead size + bead count
  - fengshui_advisor_tool  : birth_year → Can Chi → Nạp âm → mệnh + lucky/unlucky
  - web_search_tool        : SerpAPI fallback for items the shop does not sell
  - gift_advisor_tool      : structured gift suggestions by recipient + occasion

fengshui logic is hardcoded against the 60-year Sexagenary cycle so the LLM
never has to guess Nạp âm itself. The LLM still drives the conversation
(asking for birth_year if missing, chaining with filter_search_tool from KB).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os
from typing import Optional

from langchain_core.messages import BaseMessage, SystemMessage
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
#  CAN CHI NẠP ÂM (60-year cycle starting at 1924 = Giáp Tý)
# ═══════════════════════════════════════════════════════════════════

CAN = ["Giáp","Ất","Bính","Đinh","Mậu","Kỷ","Canh","Tân","Nhâm","Quý"]
CHI = ["Tý","Sửu","Dần","Mão","Thìn","Tỵ","Ngọ","Mùi","Thân","Dậu","Tuất","Hợi"]

# 30 Nạp âm — each covers 2 consecutive years
NAPAM: list[tuple[str, str]] = [
    ("Hải Trung Kim", "Kim"),    ("Lư Trung Hỏa", "Hỏa"),
    ("Đại Lâm Mộc",  "Mộc"),     ("Lộ Bàng Thổ",  "Thổ"),
    ("Kiếm Phong Kim","Kim"),    ("Sơn Đầu Hỏa",  "Hỏa"),
    ("Giản Hạ Thủy", "Thủy"),    ("Thành Đầu Thổ","Thổ"),
    ("Bạch Lạp Kim", "Kim"),     ("Dương Liễu Mộc","Mộc"),
    ("Tỉnh Tuyền Thủy","Thủy"),  ("Ốc Thượng Thổ","Thổ"),
    ("Tích Lịch Hỏa","Hỏa"),     ("Tùng Bách Mộc","Mộc"),
    ("Trường Lưu Thủy","Thủy"),  ("Sa Trung Kim",  "Kim"),
    ("Sơn Hạ Hỏa",   "Hỏa"),     ("Bình Địa Mộc",  "Mộc"),
    ("Bích Thượng Thổ","Thổ"),   ("Kim Bạc Kim",   "Kim"),
    ("Phú Đăng Hỏa", "Hỏa"),     ("Thiên Hà Thủy", "Thủy"),
    ("Đại Trạch Thổ","Thổ"),     ("Thoa Xuyến Kim","Kim"),
    ("Tang Đố Mộc",  "Mộc"),     ("Đại Khê Thủy",  "Thủy"),
    ("Sa Trung Thổ", "Thổ"),     ("Thiên Thượng Hỏa","Hỏa"),
    ("Thạch Lựu Mộc","Mộc"),     ("Đại Hải Thủy",  "Thủy"),
]

ELEMENT_INFO = {
    "Kim": {
        "tuong_sinh_voi_minh": "Thổ",   # Thổ sinh Kim (đại cát)
        "minh_sinh_ra":         "Thủy",  # Kim sinh Thủy (good)
        "tuong_khac_voi_minh":  "Hỏa",   # Hỏa khắc Kim (đại kỵ)
        "minh_khac":            "Mộc",
        "lucky_colors":   ["trắng","ghi","xám","ánh kim","vàng kim"],
        "unlucky_colors": ["đỏ","hồng","tím","cam"],
        "lucky_stones":   ["thạch anh trắng","đá mặt trăng","mã não trắng","pha lê trong"],
    },
    "Mộc": {
        "tuong_sinh_voi_minh": "Thủy",
        "minh_sinh_ra":         "Hỏa",
        "tuong_khac_voi_minh":  "Kim",
        "minh_khac":            "Thổ",
        "lucky_colors":   ["xanh lá","xanh đen","đen","xanh dương"],
        "unlucky_colors": ["trắng","ghi","vàng kim","ánh kim"],
        "lucky_stones":   ["ngọc bích","jade xanh","malachite","aventurine xanh","mắt mèo xanh"],
    },
    "Thủy": {
        "tuong_sinh_voi_minh": "Kim",
        "minh_sinh_ra":         "Mộc",
        "tuong_khac_voi_minh":  "Thổ",
        "minh_khac":            "Hỏa",
        "lucky_colors":   ["đen","xanh dương","xanh nước biển","trắng"],
        "unlucky_colors": ["vàng","nâu","đỏ đất"],
        "lucky_stones":   ["aquamarine","sapphire xanh","obsidian đen","mã não đen","thạch anh xanh"],
    },
    "Hỏa": {
        "tuong_sinh_voi_minh": "Mộc",
        "minh_sinh_ra":         "Thổ",
        "tuong_khac_voi_minh":  "Thủy",
        "minh_khac":            "Kim",
        "lucky_colors":   ["đỏ","hồng","cam","tím","xanh lá đậm"],
        "unlucky_colors": ["đen","xanh dương đậm"],
        "lucky_stones":   ["ruby","garnet","mã não đỏ","tourmaline đỏ","agate hồng"],
    },
    "Thổ": {
        "tuong_sinh_voi_minh": "Hỏa",
        "minh_sinh_ra":         "Kim",
        "tuong_khac_voi_minh":  "Mộc",
        "minh_khac":            "Thủy",
        "lucky_colors":   ["vàng","nâu","đỏ đất","cam đất"],
        "unlucky_colors": ["xanh lá","xanh đen"],
        "lucky_stones":   ["citrine","hổ phách","mã não vàng","mắt hổ","tiger eye"],
    },
}


def _year_to_can_chi(year: int) -> dict:
    """Map a birth year to Can-Chi, Nạp âm and Ngũ hành element."""
    offset = (year - 1924) % 60
    if offset < 0:
        offset += 60
    stem  = CAN[offset % 10]
    chi   = CHI[offset % 12]
    napam, element = NAPAM[offset // 2]
    return {
        "year":     year,
        "can":      stem,
        "chi":      chi,
        "can_chi":  f"{stem} {chi}",
        "napam":    napam,
        "element":  element,
    }


# ═══════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def size_calculator_tool(wrist_cm: float) -> str:
    """
    Tính size vòng tay phù hợp dựa trên chu vi cổ tay (cm).

    Trả về size dây, số hạt gợi ý theo từng size hạt phổ biến (6mm / 8mm / 10mm),
    và lời khuyên chừa thêm cho thoải mái.

    Args:
        wrist_cm: chu vi cổ tay đo bằng cm (vd 14.5, 16, 17)
    """
    if wrist_cm <= 0:
        return json.dumps({"error": "Chu vi cổ tay phải > 0 cm"}, ensure_ascii=False)

    # Standard: bracelet inner circumference ≈ wrist + 1.5 cm (slack for comfort)
    bracelet_cm = wrist_cm + 1.5

    # Bead count = bracelet_circumference / bead_diameter
    # Diameter trong cm: 6mm=0.6, 8mm=0.8, 10mm=1.0
    bead_counts = {
        "6mm":  round(bracelet_cm / 0.6),
        "8mm":  round(bracelet_cm / 0.8),
        "10mm": round(bracelet_cm / 1.0),
    }

    # Recommendation rule
    if wrist_cm < 14:
        recommended = "6mm (tay nhỏ, hạt nhỏ trông cân đối hơn)"
    elif wrist_cm < 16:
        recommended = "6mm hoặc 8mm tuỳ sở thích"
    elif wrist_cm < 18:
        recommended = "8mm (cân đối nhất, là size phổ biến)"
    else:
        recommended = "8mm hoặc 10mm cho tay to / nam giới"

    return json.dumps({
        "wrist_cm":     wrist_cm,
        "bracelet_cm":  round(bracelet_cm, 1),
        "bead_counts":  bead_counts,
        "recommended":  recommended,
        "note":         "Số hạt ước lượng, có thể ±1 hạt do dây co giãn và kích thước hạt thực tế.",
    }, ensure_ascii=False)


@tool
def fengshui_advisor_tool(birth_year: int) -> str:
    """
    Suy ra mệnh Ngũ Hành, Can Chi, Nạp âm từ NĂM SINH dương lịch.
    Trả về thêm: màu/đá hợp với bản mệnh + màu/đá hợp tương sinh (đại cát) +
    màu/đá tương khắc (đại kỵ).

    PHẢI gọi tool này TRƯỚC KHI tư vấn sản phẩm theo mệnh / tuổi.
    NẾU user chỉ nói con giáp (vd "tuổi Tý") MÀ KHÔNG nói năm sinh → KHÔNG gọi tool,
    hỏi lại năm sinh trước (vì cùng tuổi Tý có 5 mệnh khác nhau theo chu kỳ 60 năm).

    Args:
        birth_year: năm sinh dương lịch (vd 1990, 1984)
    """
    if birth_year < 1900 or birth_year > 2100:
        return json.dumps(
            {"error": f"birth_year {birth_year} ngoài phạm vi hỗ trợ (1900-2100)"},
            ensure_ascii=False,
        )

    info = _year_to_can_chi(birth_year)
    element = info["element"]
    rel = ELEMENT_INFO[element]

    result = {
        **info,
        "ban_menh":              f"Mệnh {element}",
        "tuong_sinh_dai_cat":    f"Mệnh {rel['tuong_sinh_voi_minh']} (sinh ra {element}) — đại cát, mạnh nhất",
        "tuong_khac_dai_ky":     f"Mệnh {rel['tuong_khac_voi_minh']} (khắc {element}) — nên tránh",
        "lucky_colors":          rel["lucky_colors"],
        "unlucky_colors":        rel["unlucky_colors"],
        "lucky_stones":          rel["lucky_stones"],
        "suggested_filter_elements": [
            element,                              # bản mệnh
            rel["tuong_sinh_voi_minh"],           # tương sinh (mạnh nhất)
        ],
        "explanation": (
            f"Bạn sinh năm {birth_year} - Can Chi {info['can_chi']} - "
            f"Nạp âm {info['napam']} - mệnh {element}. "
            f"Hợp nhất với sản phẩm thuộc mệnh {rel['tuong_sinh_voi_minh']} (tương sinh) "
            f"và mệnh {element} (bản mệnh). Tránh mệnh {rel['tuong_khac_voi_minh']}."
        ),
    }
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


@tool
def gift_advisor_tool(
    recipient: str,
    occasion:  Optional[str] = None,
    gender:    Optional[str] = None,
    age_range: Optional[str] = None,
) -> str:
    """
    Gợi ý hướng chọn quà phong thủy theo người nhận và dịp.
    Trả về danh sách category / màu / chất liệu phù hợp để agent CHAIN tiếp với
    filter_search_tool tìm sản phẩm cụ thể.

    Args:
        recipient: ai - vd "mẹ", "bạn gái", "sếp", "đồng nghiệp", "bố", "ng yêu"
        occasion:  dịp gì - "sinh nhật", "khai trương", "tân gia", "valentine", "8/3"
        gender:    "nam" | "nữ" (nếu biết)
        age_range: "trẻ" | "trung niên" | "lớn tuổi" (nếu biết)
    """
    r = (recipient or "").lower().strip()
    suggestion = {
        "recipient":  recipient,
        "occasion":   occasion,
        "categories": ["vòng tay"],
        "colors":     [],
        "materials":  [],
        "note":       "",
    }

    if any(k in r for k in ["mẹ","má","ba má","mom"]):
        suggestion["categories"] = ["vòng tay","lư xông trầm","nhang"]
        suggestion["colors"]     = ["vàng","nâu","đỏ"]
        suggestion["materials"]  = ["mã não","trầm hương"]
        suggestion["note"]       = "Tặng mẹ: ưu tiên vòng đá đầm/sang, hoặc lư xông trầm trang trí nhà."
    elif any(k in r for k in ["bố","ba","cha","dad"]):
        suggestion["categories"] = ["vòng tay","chuỗi hạt","treo xe"]
        suggestion["colors"]     = ["đen","nâu","xám"]
        suggestion["materials"]  = ["mã não đen","trầm hương","tourmaline"]
        suggestion["note"]       = "Tặng bố: hạt 10mm hợp tay nam; treo xe / chuỗi 108 hạt thường rất được ưa chuộng."
    elif any(k in r for k in ["bạn gái","ng yêu","người yêu","crush","gấu","vợ"]):
        suggestion["categories"] = ["vòng tay","dây chuyền"]
        suggestion["colors"]     = ["hồng","trắng","xanh dương","đa sắc"]
        suggestion["materials"]  = ["aquamarine","tourmaline","beryl","mã não hồng"]
        suggestion["note"]       = "Tặng người yêu / vợ: ưu tiên đá màu nhẹ nhàng, có ý nghĩa tình duyên (mã não hồng, aquamarine)."
    elif any(k in r for k in ["sếp","boss","giám đốc","cấp trên"]):
        suggestion["categories"] = ["vòng tay","treo xe","tượng phật"]
        suggestion["colors"]     = ["đen","vàng","đa sắc"]
        suggestion["materials"]  = ["tourmaline","mã não","trầm hương"]
        suggestion["note"]       = "Tặng sếp: chọn sản phẩm trang trọng - chuỗi trầm, vòng tourmaline, treo xe trầm hương."
    elif any(k in r for k in ["bạn","đồng nghiệp","colleague","friend"]):
        suggestion["categories"] = ["vòng tay"]
        suggestion["colors"]     = ["đa sắc"]
        suggestion["materials"]  = ["mã não","beryl","tourmaline"]
        suggestion["note"]       = "Tặng bạn / đồng nghiệp: vòng đa sắc hợp mọi mệnh là lựa chọn an toàn."
    else:
        suggestion["note"] = (
            "Không xác định cụ thể được người nhận. Hỏi user thêm: giới tính, độ tuổi, "
            "có biết mệnh / tuổi không, rồi chain với filter_search_tool."
        )

    if occasion:
        occ = occasion.lower()
        if "khai trương" in occ or "tân gia" in occ:
            suggestion["categories"].append("lư xông trầm")
            suggestion["note"] += " Dịp khai trương/tân gia nên kèm lư xông trầm để xả khí, hút tài lộc."
        elif "valentine" in occ or "8/3" in occ or "20/10" in occ:
            suggestion["materials"].append("aquamarine")
            suggestion["colors"].append("hồng")

    return json.dumps(suggestion, ensure_ascii=False)


TOOLS = [
    size_calculator_tool,
    fengshui_advisor_tool,
    web_search_tool,
    gift_advisor_tool,
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

1) HỎI SIZE VÒNG
   - Có chu vi cổ tay (cm) → gọi size_calculator_tool
   - Chỉ nói "tay to/nhỏ" / chiều cao cân nặng → hỏi cổ tay đo bằng dây mềm cm,
     hoặc gợi ý: nữ ~15cm, nam ~17cm

2) TƯ VẤN THEO MỆNH / TUỔI (quan trọng nhất)
   - User nói "mệnh Kim/Mộc/Thủy/Hỏa/Thổ" → chain ngay filter_search_tool
     với compatible_elements
   - User nói "tuổi Tý/Sửu/..." NHƯNG KHÔNG nói năm sinh
     → HỎI LẠI năm sinh dương lịch. Vì cùng tuổi Tý mỗi 60 năm có mệnh khác nhau:
       Giáp Tý (1924, 1984) = Kim
       Bính Tý (1936, 1996) = Thủy
       Mậu Tý (1948, 2008) = Hỏa
       Canh Tý (1960, 2020) = Thổ
       Nhâm Tý (1912, 1972) = Mộc
     KHÔNG ĐƯỢC ĐOÁN mệnh dựa trên con giáp.
   - User có năm sinh → gọi fengshui_advisor_tool(birth_year=...)
     → đọc kết quả → chain với filter_search_tool(compatible_elements=...) để
       lấy sản phẩm phù hợp → trình bày với lời giải thích "vì bạn mệnh X, sản
       phẩm này hợp tương sinh / bản mệnh"

3) TƯ VẤN QUÀ TẶNG
   - Gọi gift_advisor_tool với info user cung cấp (recipient, occasion,...)
   - Đọc gợi ý → chain với filter_search_tool / semantic_search_tool

4) HỎI SẢN PHẨM NGOÀI PHẠM VI SHOP
   - Đầu tiên thử semantic_search_tool xem shop có không
   - Nếu không có → web_search_tool, NHƯNG phải nói rõ "đây là thông tin tham
     khảo từ web, sản phẩm này hiện shop chưa bán"

QUY TẮC CHUNG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Sau khi tư vấn xong nhớ trình bày sản phẩm gợi ý kèm:
  - Tên + giá + ảnh (![tên](image_cover))
  - Lý do vì sao hợp (tương sinh / màu / chất liệu)
- Trả lời tiếng Việt, thân thiện, xưng "shop"
- Không bịa thông tin về mệnh — luôn dùng tool fengshui_advisor_tool
"""


# ═══════════════════════════════════════════════════════════════════
#  GRAPH
# ═══════════════════════════════════════════════════════════════════

def agent_node(state: MessagesState) -> dict:
    llm = make_llm_with_tools(TOOLS, temperature=0.3)
    response = llm.invoke([SystemMessage(content=SKILLS_SYSTEM_PROMPT)] + list(state["messages"]))
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
    log.info("ENTER  skills_agent (%d msgs)", len(messages))
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
