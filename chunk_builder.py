"""
chunk_builder.py – Build enriched text chunks for embedding & OpenSearch.

Strategy:
  - KEEP:   product name, description, feng shui meaning, material, elements
  - REMOVE: "Hướng dẫn bảo quản", "Shop cam kết" (boilerplate, same across products)
  - The removed sections are stored separately by product type for lookup.
"""

import re
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════════
#  BOILERPLATE SECTION HEADERS (case-insensitive)
# ═══════════════════════════════════════════════════════════════════
BOILERPLATE_HEADERS = [
    r"HƯỚNG\s+DẪN\s+BẢO\s+QUẢN",
    r"SHOP\s+CAM\s+KẾT",
    r"VẠN\s+AN\s+(GROUP\s+)?CAM\s+KẾT",
    r"Hiện\s+nay,?\s+trên\s+thị\s+trường",  # generic warning paragraph
]

# Compile once
_BOILERPLATE_RE = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(BOILERPLATE_HEADERS) + r")",
    re.IGNORECASE | re.MULTILINE,
)


def _remove_boilerplate(raw_text: str) -> Tuple[str, str]:
    """
    Split raw_text into (enriched_text, boilerplate_text).
    Everything after the first boilerplate header is considered boilerplate.
    """
    match = _BOILERPLATE_RE.search(raw_text)
    if match:
        enriched = raw_text[: match.start()].strip()
        boilerplate = raw_text[match.start():].strip()
    else:
        enriched = raw_text.strip()
        boilerplate = ""
    return enriched, boilerplate


def _clean_whitespace(text: str) -> str:
    """Normalize excessive blank lines and whitespace."""
    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def build_chunk(product_id: int, raw_text: str, metadata: Dict) -> Dict:
    """
    Build an enriched text chunk for a single product.

    Returns:
        {
            "product_id":      int,
            "chunk_text":      str,   # for embedding → OpenSearch
            "boilerplate":     str,   # stored separately for lookup
            "metadata_filter": dict,  # denormalized metadata for OpenSearch filter
        }
    """
    enriched, boilerplate = _remove_boilerplate(raw_text)
    enriched = _clean_whitespace(enriched)

    # Append structured metadata summary at the end of chunk
    # This improves semantic search quality by making key attributes explicit
    meta_lines = []
    if metadata.get("category"):
        meta_lines.append(f"Danh mục: {metadata['category']}")
    if metadata.get("material"):
        materials = metadata["material"]
        if isinstance(materials, list):
            meta_lines.append(f"Chất liệu: {', '.join(materials)}")
    if metadata.get("compatible_elements"):
        elements = metadata["compatible_elements"]
        if isinstance(elements, list) and elements:
            meta_lines.append(f"Mệnh phù hợp: {', '.join(elements)}")
    if metadata.get("colors"):
        colors = metadata["colors"]
        if isinstance(colors, list):
            meta_lines.append(f"Màu sắc: {', '.join(colors)}")

    if meta_lines:
        enriched += "\n\n[Thuộc tính phong thủy]\n" + "\n".join(meta_lines)

    # Build metadata for OpenSearch filter fields
    metadata_filter = {
        "product_id":          product_id,
        "name":                metadata.get("name", ""),
        "category":            metadata.get("category", ""),
        "material":            metadata.get("material", []),
        "compatible_elements": metadata.get("compatible_elements", []),
        "colors":              metadata.get("colors", []),
        "product_size":        metadata.get("product_size", metadata.get("bead_sizes", [])),
        "brand":               metadata.get("brand", "Vạn An Group"),
        "in_stock":            metadata.get("in_stock", True),
        "price_range":         metadata.get("price_range"),
    }

    return {
        "product_id":      product_id,
        "chunk_text":      enriched,
        "boilerplate":     boilerplate,
        "metadata_filter": metadata_filter,
    }


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_raw = """Vòng tay đá Aquamarine cho mệnh Thủy, mệnh Mộc Vạn An Group
- Đá Aquamarine gắn liền với đại dương

THÔNG TIN SẢN PHẨM
- Chất liệu: Đá Aquamarine
- Phù hợp cho mệnh Thủy, mệnh Mộc
- Kích thước hạt: 6 ly, 8 ly

HƯỚNG DẪN BẢO QUẢN VÒNG TAY ĐÁ TỰ NHIÊN
- Nên dùng khăn giấy lau qua
- Tránh tiếp xúc nhiệt độ cao

SHOP CAM KẾT
- Sản phẩm giống mô tả
- Miễn phí đổi trả trong 7 ngày"""

    sample_meta = {
        "name": "Vòng tay đá Aquamarine",
        "category": "vòng tay",
        "material": ["aquamarine"],
        "compatible_elements": ["Thủy", "Mộc"],
        "colors": ["xanh dương"],
        "product_size": ["6mm", "8mm"],
        "brand": "Vạn An Group",
        "in_stock": True,
        "price_range": None,
    }

    result = build_chunk(99, sample_raw, sample_meta)
    print("=== CHUNK TEXT ===")
    print(result["chunk_text"])
    print("\n=== BOILERPLATE ===")
    print(result["boilerplate"][:200])
