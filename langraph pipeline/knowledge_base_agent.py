"""
knowledge_base_agent.py – Tool-using agent for everything related to product data.

Tools (LLM picks based on docstring):
  - semantic_search_tool   : natural-language descriptive query
  - keyword_search_tool    : a specific stone / material / proper noun
  - filter_search_tool     : structured filters (category, material, color, element)
  - get_product_detail_tool: deep-dive on one product (by id)

All tools enrich OpenSearch hits with PostgreSQL rows so the LLM sees the full
product (price_range, quantity_max, image URL, full description).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import os
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import db_service
import opensearch_service
from embedding_service import embed_single
from gemini import make_llm_with_tools
from logger import ToolLoggerCallback, get_logger


log         = get_logger("kb")
_callback   = ToolLoggerCallback("kb")


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _serialize_product(product) -> dict:
    """Serialize a SQLAlchemy Product row to a JSON-friendly dict."""
    image_cover = None
    if product.image:
        if isinstance(product.image, list) and product.image:
            image_cover = product.image[0]
        elif isinstance(product.image, dict):
            image_cover = product.image.get("cover") or next(iter(product.image.values()), None)

    return {
        "product_id":          product.product_id,
        "name":                product.name,
        "category":            product.category,
        "material":            list(product.material or []),
        "compatible_elements": list(product.compatible_elements or []),
        "colors":              list(product.colors or []),
        "product_size":        list(product.product_size or []),
        "price_range":         product.price_range,
        "in_stock":            bool(product.in_stock),
        "quantity_max":        getattr(product, "quantity_max", None),
        "image_cover":         image_cover,
        "product_description": product.product_description,
    }


def _enrich_with_pg(hits: list[dict]) -> list[dict]:
    """Given OpenSearch hits, fetch PG rows and return merged product objects."""
    enriched = []
    for hit in hits:
        pid = hit.get("product_id")
        if pid is None:
            continue
        product = db_service.get_product_by_id(pid)
        if product is None:
            continue
        merged = _serialize_product(product)
        if "score" in hit:
            merged["_score"] = hit["score"]
        enriched.append(merged)
    return enriched


def _format_for_llm(products: list[dict]) -> str:
    """Compact JSON suitable to stuff into the LLM context."""
    return json.dumps(products, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def semantic_search_tool(query: str, top_k: int = 5) -> str:
    """
    Tìm sản phẩm theo mô tả tự nhiên / ý nghĩa / công dụng.

    Dùng khi user mô tả sản phẩm bằng ngôn ngữ tự nhiên, không nêu tên đá/chất liệu
    cụ thể. Ví dụ: "đeo tay cho may mắn", "vòng nhẹ nhàng dịu mắt".

    Args:
        query: Câu truy vấn của user
        top_k: Số sản phẩm trả về (mặc định 5)
    """
    embedding = embed_single(query)
    hits = opensearch_service.semantic_search(embedding, k=top_k)
    products = _enrich_with_pg(hits)
    return _format_for_llm(products)


@tool
def keyword_search_tool(query: str, top_k: int = 5) -> str:
    """
    Tìm sản phẩm theo từ khoá cụ thể trong tên / mô tả.

    Dùng khi user nhắc đích danh đá / chất liệu / thương hiệu, ví dụ:
    "tourmaline", "aquamarine", "trầm hương", "mã não đen".

    Args:
        query: Từ khoá tìm kiếm
        top_k: Số sản phẩm trả về
    """
    hits = opensearch_service.keyword_search(query, k=top_k)
    products = _enrich_with_pg(hits)
    return _format_for_llm(products)


@tool
def filter_search_tool(
    category:            Optional[str]       = None,
    material:            Optional[str]       = None,
    compatible_elements: Optional[str]       = None,
    colors:              Optional[str]       = None,
    in_stock:            Optional[bool]      = None,
    price_range:         Optional[str]       = None,
    top_k:               int                 = 10,
) -> str:
    """
    Lọc sản phẩm theo các thuộc tính có cấu trúc.

    Dùng khi user nêu rõ tiêu chí lọc: theo danh mục, chất liệu, mệnh phong thủy
    (Kim/Mộc/Thủy/Hỏa/Thổ), màu sắc. Có thể truyền nhiều tiêu chí cùng lúc.

    Args:
        category:            Vd "vòng tay", "nhang", "lư xông trầm",...
        material:            Vd "tourmaline", "mã não đen", "trầm hương"
        compatible_elements: Mệnh hợp - Kim | Mộc | Thủy | Hỏa | Thổ
        colors:              Vd "đen", "xanh dương", "đa sắc"
        in_stock:            True để chỉ lấy sản phẩm còn hàng
        price_range:         Vd "100.000 - 200.000"
        top_k:               Số sản phẩm trả về
    """
    filters = {}
    if category:            filters["category"]            = category
    if material:            filters["material"]            = material
    if compatible_elements: filters["compatible_elements"] = compatible_elements
    if colors:              filters["colors"]              = colors
    if in_stock is not None: filters["in_stock"]           = in_stock
    if price_range:         filters["price_range"]         = price_range

    hits = opensearch_service.filter_search(filters, k=top_k)
    products = _enrich_with_pg(hits)
    return _format_for_llm(products)


@tool
def get_product_detail_tool(product_id: int) -> str:
    """
    Lấy đầy đủ thông tin một sản phẩm cụ thể theo product_id.

    Dùng khi user hỏi chi tiết về một sản phẩm đã được nhắc đến (vd: "cho tôi
    biết thêm về sản phẩm số 5", "vòng aquamarine kia bảo hành thế nào").

    Args:
        product_id: Mã sản phẩm
    """
    product = db_service.get_product_by_id(product_id)
    if product is None:
        return json.dumps({"error": f"Không tìm thấy product_id={product_id}"}, ensure_ascii=False)
    return _format_for_llm([_serialize_product(product)])


TOOLS = [
    semantic_search_tool,
    keyword_search_tool,
    filter_search_tool,
    get_product_detail_tool,
]


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

KB_SYSTEM_PROMPT = """
Bạn là agent tư vấn sản phẩm của shop phong thủy Vạn An Group.

Nhiệm vụ: trả lời mọi câu hỏi liên quan đến danh mục sản phẩm bằng cách CHỦ ĐỘNG
gọi tool để lấy data thực từ DB, không bịa.

QUY TẮC CHỌN TOOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- User mô tả tự nhiên ("vòng nhẹ nhàng cho nữ", "đeo cho may mắn")
  → semantic_search_tool
- User nhắc đích danh đá / chất liệu / loại sản phẩm cụ thể ("aquamarine",
  "tourmaline", "nhang trầm")
  → keyword_search_tool
- User nêu tiêu chí lọc (mệnh, màu, category, giá)
  → filter_search_tool
- User đã biết sản phẩm cụ thể, muốn biết chi tiết
  → get_product_detail_tool

Có thể gọi NHIỀU tool nếu cần (vd: filter rồi xem detail).

QUY TẮC TRẢ LỜI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Trình bày tối đa 3-5 sản phẩm phù hợp nhất.
2. Mỗi sản phẩm trình bày:
   - Tên sản phẩm (không quá dài, có thể rút gọn)
   - Giá (price_range)
   - Tình trạng (in_stock, quantity_max nếu có)
   - Ảnh: dùng markdown ![tên](image_cover) - QUAN TRỌNG để user xem được
   - 1-2 câu ý nghĩa / công dụng (rút từ product_description)
3. Nếu không tìm thấy sản phẩm phù hợp, nói rõ và đề xuất hướng khác (đổi tiêu chí,
   gợi ý chat nhân viên, hoặc hỏi web_search nếu là sản phẩm shop không bán).
4. Trả lời bằng tiếng Việt, giọng thân thiện, xưng "shop" - gọi khách là "bạn".
5. Không bịa thông tin. Nếu DB không có field nào đó (vd: số hạt theo size,
   giấy chứng chỉ), hãy nói "shop sẽ kiểm tra lại và phản hồi sau, hoặc bạn
   inbox trực tiếp Shopee để được nhân viên hỗ trợ".
"""


# ═══════════════════════════════════════════════════════════════════
#  GRAPH
# ═══════════════════════════════════════════════════════════════════

def agent_node(state: MessagesState) -> dict:
    llm = make_llm_with_tools(TOOLS, temperature=0.3)
    response = llm.invoke([SystemMessage(content=KB_SYSTEM_PROMPT)] + list(state["messages"]))
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
    """Public entrypoint used by graph.py."""
    log.info("ENTER  knowledge_base_agent (%d msgs)", len(messages))
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
    log.info("EXIT   knowledge_base_agent | tools=%s | reply=%d chars",
             tools_called, len(final) if isinstance(final, str) else 0)
    return {
        "final_response": final,
        "messages": result["messages"],
        "tools_called": tools_called,
    }
