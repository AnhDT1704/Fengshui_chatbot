"""
response_verifier.py – Lớp kiểm tra chống BỊA sản phẩm trước khi trả lời khách.

Ý tưởng: mỗi lượt, các tool (keyword_search / filter_search / image_search ...) trả
về JSON chứa sản phẩm THẬT từ DB. Ta thu "tập grounded" = mọi sản phẩm đã tra được
trong lượt đó. Câu trả lời cuối cùng KHÔNG được nêu sản phẩm / giá nào nằm ngoài tập
này. Nếu vi phạm → graph.py sẽ cho sinh lại câu trả lời (tối đa 1 lần).

Public API:
    is_product_answer(text)                 -> bool   (câu này có nhắc sản phẩm/giá?)
    collect_grounded_products(messages)     -> list[dict]
    verify_answer(answer_text, grounded)    -> {"ok": bool, "issues": [...], "fix_hint": str}

Chỉ câu trả lời CÓ sản phẩm mới đi qua verify_answer — câu chào hỏi / tư vấn màu
mệnh / hướng dẫn bảo quản... không chứa sản phẩm thì bỏ qua (is_product_answer=False).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
import re
from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from gemini import make_llm
from logger import get_logger

log = get_logger("verifier")


# ── Nhận biết câu trả lời có nhắc sản phẩm ──────────────────────────
# Dấu hiệu mạnh: có GIÁ (vd "250.000 VNĐ", "250.000đ") hoặc ảnh sản phẩm markdown.
_PRICE_RE = re.compile(r"\d[\d.\s,]*\s*(?:vnđ|vnd|₫|đ)\b", re.IGNORECASE)
_PRODUCT_IMG_RE = re.compile(r"!\[[^\]]*\]\(https?://", re.IGNORECASE)


def is_product_answer(text: str) -> bool:
    """True nếu câu trả lời trình bày sản phẩm cụ thể (có giá hoặc ảnh sản phẩm)."""
    if not text:
        return False
    return bool(_PRICE_RE.search(text) or _PRODUCT_IMG_RE.search(text))


# ── Thu tập sản phẩm THẬT đã tra trong lượt (từ ToolMessage) ─────────

def _walk_collect(obj: Any, out: dict[int, dict]) -> None:
    """Đệ quy gom mọi object trông như sản phẩm (có product_id + name)."""
    if isinstance(obj, dict):
        pid = obj.get("product_id")
        name = obj.get("name")
        if pid is not None and isinstance(name, str) and name.strip():
            # giữ bản đầu tiên gặp cho mỗi product_id
            out.setdefault(int(pid), {
                "product_id":  int(pid),
                "name":        name.strip(),
                "price_range": obj.get("price_range"),
            })
        for v in obj.values():
            _walk_collect(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk_collect(v, out)


def collect_grounded_products(messages: list[BaseMessage]) -> list[dict]:
    """Parse mọi ToolMessage trong lượt → danh sách sản phẩm THẬT (product_id, name, price)."""
    out: dict[int, dict] = {}
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        content = m.content
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue
        _walk_collect(data, out)
    return list(out.values())


# ── Verifier LLM ────────────────────────────────────────────────────

_VERIFIER_PROMPT = """Bạn là bộ KIỂM DUYỆT chống bịa sản phẩm cho chatbot bán hàng.

Bạn được cho:
1) DANH SÁCH SẢN PHẨM THẬT (grounded) đã tra từ database trong lượt này — đây là
   nguồn sự thật DUY NHẤT về tên + giá sản phẩm.
2) CÂU TRẢ LỜI của chatbot gửi cho khách.

Nhiệm vụ: kiểm tra MỌI sản phẩm mà câu trả lời trình bày như SẢN PHẨM CỦA SHOP
(có tên riêng và/hoặc kèm giá) có được hỗ trợ bởi danh sách grounded không:
- Tên sản phẩm khớp với một mục trong grounded (cho phép khác biệt nhỏ về cách viết).
- Giá (nếu câu trả lời có nêu) khớp với price_range của sản phẩm đó trong grounded.

ok=false NẾU: có sản phẩm nêu tên/giá mà KHÔNG có trong grounded (bịa tên, bịa giá,
giá sai), HOẶC câu trả lời nêu giá/sản phẩm trong khi grounded RỖNG.
ok=true NẾU: mọi sản phẩm + giá đều khớp grounded.

BỎ QUA (không tính là vi phạm): tên đá/màu/chất liệu nói chung (vd "đá mắt mèo",
"màu xanh lá") khi KHÔNG đi kèm như một sản phẩm có giá; lời tư vấn mệnh/phong thủy.

CHỈ trả về JSON đúng định dạng, không thêm chữ nào khác:
{"ok": true/false, "issues": ["mô tả ngắn từng vi phạm"], "fix_hint": "1 câu hướng dẫn sửa"}
"""


def _extract_json(text: str) -> dict:
    """Lấy object JSON đầu tiên trong text (model đôi khi bọc thêm chữ)."""
    if not isinstance(text, str):
        return {}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {}


def verify_answer(answer_text: str, grounded: list[dict]) -> dict:
    """Kiểm tra câu trả lời so với tập sản phẩm grounded. Không bao giờ raise."""
    # Câu có sản phẩm nhưng lượt này KHÔNG tra được sản phẩm nào → chắc chắn bịa.
    if not grounded:
        return {
            "ok": False,
            "issues": ["Câu trả lời nêu sản phẩm/giá nhưng lượt này chưa tra được sản phẩm THẬT nào từ DB."],
            "fix_hint": ("PHẢI gọi keyword_search_tool/filter_search_tool/image_search_tool để lấy "
                         "sản phẩm THẬT, chỉ nêu tên + giá có trong kết quả tool."),
        }
    try:
        llm = make_llm(temperature=0.0, max_tokens=2048)
        grounded_json = json.dumps(
            [{"name": p["name"], "price_range": p.get("price_range")} for p in grounded],
            ensure_ascii=False,
        )
        human = f"GROUNDED (sản phẩm thật):\n{grounded_json}\n\nCÂU TRẢ LỜI:\n{answer_text}"
        resp = llm.invoke([SystemMessage(content=_VERIFIER_PROMPT), SystemMessage(content=human)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        verdict = _extract_json(raw)
        if "ok" not in verdict:
            # Không parse được → fail-open (cho qua) để tránh chặn nhầm câu đúng.
            log.warning("verifier: không parse được JSON, cho qua. raw=%r", raw[:200])
            return {"ok": True, "issues": [], "fix_hint": ""}
        verdict.setdefault("issues", [])
        verdict.setdefault("fix_hint", "")
        return verdict
    except Exception as e:
        log.warning("verifier lỗi (%s) — cho qua để không chặn nhầm.", e)
        return {"ok": True, "issues": [], "fix_hint": ""}
