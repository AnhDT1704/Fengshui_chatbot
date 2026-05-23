"""
vision_agent.py – Image-aware agent.

Tools:
  - analyze_image_tool      : LLM provides a textual description of what it sees
                              in the user's photo, the tool embeds it and runs
                              semantic search to find visually similar products.
  - get_product_images_tool : Return all images stored for a given product_id.

The Gemini model itself is multimodal — when the user sends an image, the image
is part of the HumanMessage content and the LLM "sees" it natively. The agent
then writes a description and calls analyze_image_tool with that description.
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

import db_service
import opensearch_service
from embedding_service import embed_single
from gemini import make_llm_with_tools

from knowledge_base_agent import _enrich_with_pg, _format_for_llm, _serialize_product
from logger import ToolLoggerCallback, get_logger


log         = get_logger("vision")
_callback   = ToolLoggerCallback("vision")


# ═══════════════════════════════════════════════════════════════════
#  TOOLS
# ═══════════════════════════════════════════════════════════════════

@tool
def analyze_image_tool(image_description: str, top_k: int = 5) -> str:
    """
    Tìm sản phẩm trong DB giống với ảnh user gửi.

    Cách dùng đúng: SAU KHI đã quan sát ảnh user gửi, hãy mô tả thật chi tiết
    (loại sản phẩm, chất liệu, màu sắc, kiểu dáng, kích thước hạt nếu là vòng tay,
    có charm/mặt phật/đồng tiền không, v.v.) rồi truyền vào tham số
    `image_description`. Tool sẽ embedding mô tả và search semantic.

    Args:
        image_description: mô tả CHI TIẾT bằng tiếng Việt về vật trong ảnh
        top_k: số sản phẩm gợi ý (mặc định 5)
    """
    if not image_description or len(image_description.strip()) < 5:
        return json.dumps({
            "error": "image_description quá ngắn. Hãy mô tả chi tiết hơn về vật trong ảnh."
        }, ensure_ascii=False)

    embedding = embed_single(image_description)
    hits = opensearch_service.semantic_search(embedding, k=top_k)
    products = _enrich_with_pg(hits)
    return _format_for_llm(products)


@tool
def get_product_images_tool(product_id: int) -> str:
    """
    Lấy toàn bộ URL ảnh của một sản phẩm theo product_id.
    Dùng khi user muốn xem ảnh sản phẩm cụ thể.

    Args:
        product_id: Mã sản phẩm
    """
    product = db_service.get_product_by_id(product_id)
    if product is None:
        return json.dumps({"error": f"Không tìm thấy product_id={product_id}"}, ensure_ascii=False)

    images = []
    if isinstance(product.image, list):
        images = product.image
    elif isinstance(product.image, dict):
        cover = product.image.get("cover")
        if cover:
            images.append(cover)
        for k, v in product.image.items():
            if k == "cover":
                continue
            if isinstance(v, str):
                images.append(v)
            elif isinstance(v, list):
                images.extend(v)

    return json.dumps({
        "product_id":   product_id,
        "name":         product.name,
        "image_count":  len(images),
        "image_urls":   images,
    }, ensure_ascii=False)


TOOLS = [analyze_image_tool, get_product_images_tool]


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════

VISION_SYSTEM_PROMPT = """
Bạn là Vision Agent của shop phong thủy Vạn An Group, chuyên xử lý các yêu cầu
liên quan đến HÌNH ẢNH.

HAI TÌNH HUỐNG CHÍNH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1) USER GỬI ẢNH, MUỐN TÌM SẢN PHẨM TƯƠNG TỰ
   Bước 1: Quan sát ảnh user gửi (bạn là multimodal LLM, bạn nhìn được ảnh).
   Bước 2: Mô tả CHI TIẾT bằng tiếng Việt — gồm:
           - Loại sản phẩm (vòng tay / mặt dây / lư xông / nhang...)
           - Chất liệu / đá (mã não đen, tourmaline đa sắc, trầm hương,...)
           - Màu chủ đạo
           - Kích thước hạt (nhỏ ~6mm, vừa ~8mm, to ~10mm)
           - Charm / mặt phật / tiền xu nếu có
           - Kiểu dáng (bện dây, dây thun, dây kim loại,...)
   Bước 3: Gọi analyze_image_tool(image_description=mô tả vừa viết)
   Bước 4: Trình bày 3-5 sản phẩm gần nhất kèm ảnh cover và lý do match.

2) USER MUỐN XEM ẢNH SẢN PHẨM CỦA SHOP
   - User đã biết tên / id sản phẩm → gọi get_product_images_tool(product_id)
   - Nếu chưa biết id, hãy dùng tên trong câu hỏi đối chiếu lịch sử hội thoại,
     hoặc trả lời chuyển sang Knowledge Base để search trước.

QUY TẮC TRẢ LỜI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Luôn render ảnh bằng markdown: ![tên sản phẩm](url)
- Tóm tắt vì sao chọn sản phẩm này (giống ảnh ở điểm gì)
- Nếu không tìm thấy sản phẩm tương đồng → gợi ý mô tả cụ thể hơn hoặc chat
  trực tiếp shop qua Shopee
- Trả lời tiếng Việt, thân thiện
"""


# ═══════════════════════════════════════════════════════════════════
#  GRAPH
# ═══════════════════════════════════════════════════════════════════

def agent_node(state: MessagesState) -> dict:
    llm = make_llm_with_tools(TOOLS, temperature=0.3)
    response = llm.invoke([SystemMessage(content=VISION_SYSTEM_PROMPT)] + list(state["messages"]))
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
    log.info("ENTER  vision_agent (%d msgs)", len(messages))
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
    log.info("EXIT   vision_agent | tools=%s | reply=%d chars",
             tools_called, len(final) if isinstance(final, str) else 0)
    return {
        "final_response": final,
        "messages": result["messages"],
        "tools_called": tools_called,
    }
