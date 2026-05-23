"""
metadata_extractor.py – Extract structured metadata from raw product text.

Extracts: name, category, material, compatible_elements, colors,
          product_size, price_range, brand, origin, warranty, in_stock.
"""

import re
from typing import List, Optional, Dict


# ═══════════════════════════════════════════════════════════════════
#  CATEGORY DETECTION
# ═══════════════════════════════════════════════════════════════════
CATEGORY_KEYWORDS = {
    "vòng tay":      ["vòng tay", "vòng trầm", "vòng ngũ sắc", "vòng chỉ đỏ"],
    "chuỗi hạt":     ["chuỗi 108", "chuỗi trầm", "chuỗi hạt"],
    "dây chuyền":    ["dây chuyền", "mặt dây chuyền"],
    "nhang":         ["nhang nụ", "nhang sạch", "nhang bài", "nhang khoanh",
                      "nhang sen", "nhang quế", "1kg nhang"],
    "treo xe":       ["dây treo xe", "treo xe ô tô", "khánh treo xe"],
    "lư xông trầm":  ["lư xông", "lư đốt", "lư gỗ"],
    "thác khói":     ["thác khói", "thác xông trầm"],
    "tháp xông":     ["tháp xông"],
    "tượng phật":    ["tượng phật", "tượng quan âm"],
    "tiền xu":       ["đồng tiền xu", "ngũ đế"],
    "nước lau":      ["nước lau bàn thờ"],
    "phụ kiện":      ["charm", "phụ kiện"],
}


def detect_category(name: str, raw_text: str) -> str:
    """Detect product category from name and content."""
    combined = (name + " " + raw_text[:300]).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                return category
    return "khác"


# ═══════════════════════════════════════════════════════════════════
#  MATERIAL DETECTION
# ═══════════════════════════════════════════════════════════════════
MATERIAL_MAP = {
    "mã não rêu":      ["mã não rêu"],
    "mã não đen":      ["mã não đen"],
    "mã não trắng":    ["mã não trắng"],
    "mã não xanh lá":  ["mã não xanh lá"],
    "mã não đa sắc":   ["mã não đa sắc", "mã não bện dây"],
    "mã não":          ["mã não"],                       # fallback generic
    "tourmaline":      ["tourmaline"],
    "aquamarine":      ["aquamarine"],
    "thạch anh hồng":  ["thạch anh hồng"],
    "thạch anh xanh":  ["thạch anh xanh", "aventurine"],
    "thạch anh tím":   ["thạch anh tím", "amethyst"],
    "thạch anh":       ["thạch anh"],                    # fallback
    "mắt mèo hồng":   ["mắt mèo hồng"],
    "mắt mèo đỏ":     ["mắt mèo đỏ"],
    "mắt mèo vàng":   ["mắt mèo vàng"],
    "mắt mèo xanh":   ["mắt mèo xanh"],
    "mắt mèo":        ["mắt mèo"],                      # fallback
    "trầm hương":      ["trầm hương", "trầm tốc"],
    "đồng":            ["bằng đồng"],
    "gỗ":              ["lư gỗ"],
    "chỉ đỏ":         ["chỉ đỏ"],
}


def detect_material(name: str, raw_text: str) -> List[str]:
    """Detect material(s). More specific matches come first."""
    combined = (name + " " + raw_text[:500]).lower()
    found: List[str] = []
    for material, keywords in MATERIAL_MAP.items():
        for kw in keywords:
            if kw in combined:
                # Avoid adding generic if specific already found
                # e.g., don't add "mã não" if "mã não đen" is present
                is_generic = any(
                    material != m and material in m and m in found
                    for m in MATERIAL_MAP
                )
                if not is_generic and material not in found:
                    found.append(material)
                break
    # Deduplicate: remove generic if specific exists
    specific_found = list(found)
    for mat in found:
        for other in found:
            if mat != other and mat in other:
                # mat is a substring of other → mat is generic
                if mat in specific_found:
                    specific_found.remove(mat)
    return specific_found if specific_found else ["không xác định"]


# ═══════════════════════════════════════════════════════════════════
#  FIVE ELEMENTS (Ngũ Hành)
# ═══════════════════════════════════════════════════════════════════
ELEMENTS = ["Kim", "Mộc", "Thủy", "Hỏa", "Thổ"]


def detect_elements(name: str, raw_text: str) -> List[str]:
    """Extract compatible Five Elements."""
    combined = name + " " + raw_text

    # Check for "all elements" phrases
    all_patterns = [
        r"hợp\s+tất\s+cả\s+các\s+mệnh",
        r"phù\s+hợp\s+với\s+tất\s+cả",
        r"phù\s+hợp\s+tất\s+cả",
        r"tất\s+cả\s+các\s+mệnh",
        r"vua\s+phong\s+thủy",  # trầm hương = all elements
    ]
    for pat in all_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            return ELEMENTS.copy()

    # Check for specific element mentions
    found = []
    for elem in ELEMENTS:
        patterns = [
            rf"mệnh\s+{elem}",
            rf"cho\s+{elem}",
            rf"hợp\s+{elem}",
            rf"phù\s+hợp.*{elem}",
        ]
        for pat in patterns:
            if re.search(pat, combined, re.IGNORECASE):
                if elem not in found:
                    found.append(elem)
                break

    return found if found else []


# ═══════════════════════════════════════════════════════════════════
#  COLOR DETECTION
# ═══════════════════════════════════════════════════════════════════
COLOR_MAP = {
    "đen":         ["đen"],
    "trắng":       ["trắng"],
    "đỏ":          ["đỏ"],
    "hồng":        ["hồng"],
    "xanh lá":     ["xanh lá"],
    "xanh dương":  ["xanh dương"],
    "xanh rêu":    ["xanh rêu", "rêu xanh"],
    "vàng":        ["vàng"],
    "tím":         ["tím"],
    "nâu":         ["nâu"],
    "đa sắc":      ["đa sắc", "ngũ sắc", "mix màu", "tương sinh 2in1"],
}


def detect_colors(name: str, raw_text: str) -> List[str]:
    combined = (name + " " + raw_text[:500]).lower()
    found = []
    for color, keywords in COLOR_MAP.items():
        for kw in keywords:
            if kw in combined and color not in found:
                found.append(color)
                break
    return found if found else ["không xác định"]


# ═══════════════════════════════════════════════════════════════════
#  BEAD SIZE
# ═══════════════════════════════════════════════════════════════════
def detect_product_size(raw_text: str) -> List[str]:
    """Extract bead sizes like '6 ly', '8 ly', '10 ly', '8x5mm'."""
    sizes = []
    # Pattern: N ly
    ly_matches = re.findall(r"(\d+)\s*ly", raw_text, re.IGNORECASE)
    for s in ly_matches:
        val = f"{s}mm"
        if val not in sizes:
            sizes.append(val)

    # Pattern: NxMmm
    mm_matches = re.findall(r"(\d+x\d+)\s*mm", raw_text, re.IGNORECASE)
    for s in mm_matches:
        if s not in sizes:
            sizes.append(s)

    return sizes


# ═══════════════════════════════════════════════════════════════════
#  PRODUCT NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_product_name(raw_text: str) -> str:
    """First non-empty line is the product name."""
    for line in raw_text.splitlines():
        line = line.strip().strip('"').strip()
        if line:
            # Clean up: remove trailing description if it got merged
            # Some names have description merged on same line
            # Cut at known section headers
            for cut in ["THÔNG TIN", "MÔ TẢ", "Đá mắt mèo có"]:
                idx = line.find(cut)
                if idx > 20:
                    line = line[:idx].strip().rstrip(",").strip()
            return line
    return "Unknown"


# ═══════════════════════════════════════════════════════════════════
#  MAIN EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_metadata(product_id: int, raw_text: str) -> Dict:
    """
    Extract all structured metadata from raw product text.

    Returns dict ready for PostgreSQL insertion.
    """
    name = extract_product_name(raw_text)

    return {
        "product_id":          product_id,
        "name":                name,
        "category":            detect_category(name, raw_text),
        "material":            detect_material(name, raw_text),
        "compatible_elements": detect_elements(name, raw_text),
        "colors":              detect_colors(name, raw_text),
        "product_size":        detect_product_size(raw_text),
        "brand":               "Vạn An Group",
        "origin":              "Việt Nam",
        "in_stock":            True,          # default, update from admin
        "price_range":         None,          # no price range in raw txt data
        "warranty":            _extract_warranty(raw_text),
    }


def _extract_warranty(raw_text: str) -> Optional[str]:
    patterns = [
        r"[Bb]ảo hành[:\s]*(.*?)(?:\n|$)",
        r"đổi trả.*?(\d+\s*ngày)",
    ]
    for pat in patterns:
        m = re.search(pat, raw_text)
        if m:
            return m.group(1).strip() if m.group(1).strip() else None
    return None


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """Vòng tay đá Aquamarine cho mệnh Thủy, mệnh Mộc Vạn An Group
- Đá Aquamarine gắn liền với đại dương, gợi nhớ đến sự tinh khiết
THÔNG TIN SẢN PHẨM
- Chất liệu: Đá Aquamarine
- Phù hợp cho mệnh Thủy, mệnh Mộc
- Kích thước hạt: 6 ly, 8 ly
- Bảo hành: thay dây trọn đời
HƯỚNG DẪN BẢO QUẢN
- Nên dùng khăn giấy lau qua sau mỗi lần dùng
SHOP CAM KẾT
- Miễn phí đổi trả sản phẩm trong vòng 7 ngày"""

    result = extract_metadata(99, sample)
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
