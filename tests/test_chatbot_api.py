"""
test_chatbot_api.py – Integration test luồng KHÁCH (chạy thật qua API).

Kiểm các lớp lỗi: câu trả lời RỖNG/cụt, lộ thuật ngữ nội bộ, đọc số kho thô,
grounding sai, khuyến mãi chỉ còn CTA, off-platform không redirect.
Assertion cố ý LỎNG (LLM không tất định) — chỉ bám INVARIANT.
"""

import re

import pytest

from conftest import requires_api

pytestmark = [pytest.mark.integration, requires_api]

# Thuật ngữ nội bộ TUYỆT ĐỐI không được lộ ra câu trả lời khách.
LEAK = re.compile(
    r'product_id|internal_error|stock_status|stock_display|"element"|"compatible_elements"'
    r'|best_product|per_image|IRRELEVANT|_SEED_|need_more_info|suggested_filter',
    re.IGNORECASE,
)


def _ok(text: str):
    assert text and text.strip(), "câu trả lời RỖNG / cụt"
    m = LEAK.search(text)
    assert not m, f"lộ thuật ngữ nội bộ ({m.group()}): {text[:160]}"


def test_health(api):
    r = api.get_public("/health")
    assert r.status_code == 200 and r.json().get("status") == "ok"


def test_chat_requires_auth(api):
    r = api.raw_post("/chat", json={"session_id": "x", "message": "hi"})
    assert r.status_code == 401


def test_small_talk(api, user_token):
    d = api.chat(user_token, api.sid(), "xin chào shop")
    _ok(d["response"])


def test_fengshui_product_advice(api, user_token):
    """Sinh năm 2004 → mệnh Thủy + liệt kê sản phẩm thật (giá)."""
    d = api.chat(user_token, api.sid(), "tôi sinh năm 2004, gợi ý cho tôi vài mẫu vòng tay hợp mệnh")
    _ok(d["response"])
    assert "Thủy" in d["response"], "không xác định đúng mệnh"
    assert re.search(r"\d{2,3}\.\d{3}|\bVN[ĐD]\b", d["response"]), "thiếu giá sản phẩm thật"


def test_stock_not_raw_number(api, user_token):
    sid = api.sid()
    api.chat(user_token, sid, "cho tôi thông tin dây xâu 5 đồng tiền xu ngũ đế")
    d = api.chat(user_token, sid, "sản phẩm này còn hàng không")
    _ok(d["response"])
    assert re.search(r"còn hàng|còn nhiều|hết hàng|sắp hết", d["response"].lower())
    assert "939235" not in d["response"], "đọc số kho thô vô nghĩa"


def test_warranty_uses_product_column(api, user_token):
    sid = api.sid()
    api.chat(user_token, sid, "cho tôi thông tin chuỗi 108 hạt trầm hương")
    d = api.chat(user_token, sid, "sản phẩm này bảo hành thế nào")
    _ok(d["response"])
    assert "bảo hành" in d["response"].lower()


def test_promotion_not_only_cta(api, user_token):
    """Regression: câu giảm giá phải có vế THÔNG TIN KM, không chỉ mỗi CTA Shopee."""
    d = api.chat(user_token, api.sid(), "shop đang có chương trình giảm giá gì không")
    _ok(d["response"])
    body = re.split(r"ngoài ra bạn có thể tìm", d["response"])[0].strip()
    assert len(body) > 20, f"chỉ còn CTA, mất vế khuyến mãi: {d['response']!r}"


def test_off_platform_redirect(api, user_token):
    d = api.chat(user_token, api.sid(), "cho mình xin số zalo shop để đặt hàng ngoài nhé")
    _ok(d["response"])
    assert re.search(r"shopee", d["response"].lower()), "không redirect về Shopee"


def test_order_escalate_triggers_handoff(api, user_token):
    """Ca ngoài dữ liệu (tra đơn) → order_support escalate → phiên pending_admin, bot im lặng lượt sau."""
    sid = api.sid()
    d = api.chat(user_token, sid, "đơn 250115001 của mình tới đâu rồi")
    _ok(d["response"])
    assert d["agent_used"] == "order_support_agent"
    assert "escalate_to_human_tool" in (d["tools_called"] or [])
    assert d["session_status"] == "pending_admin"
    d2 = api.chat(user_token, sid, "shop còn hàng không")
    assert d2["agent_used"] == "handoff", "bot chưa im lặng sau handoff"


def test_promotion_no_handoff(api, user_token):
    """Ca chính sách (khuyến mãi) → order_support trả lời, KHÔNG escalate, KHÔNG handoff."""
    d = api.chat(user_token, api.sid(), "shop đang có khuyến mãi gì không")
    _ok(d["response"])
    assert d["session_status"] == "bot"
    assert "escalate_to_human_tool" not in (d["tools_called"] or [])


def test_price_filter_lists_products(api, user_token):
    """Lọc giá thuần → liệt kê SP theo giá, KHÔNG hỏi lại mệnh/năm sinh."""
    d = api.chat(user_token, api.sid(), "shop có vòng tay nào dưới 200k không")
    _ok(d["response"])
    assert not re.search(r"cho.*xin.*năm sinh|bạn.*mệnh gì", d["response"].lower()), \
        "hỏi mệnh cho câu lọc giá"
    assert re.search(r"\d{3}\.\d{3}", d["response"]), "không liệt kê sản phẩm theo giá"


@pytest.mark.parametrize("q", [
    "quà tặng mẹ dịp 8/3 nên chọn gì",
    "tôi có bạn thích trồng cây thì nên đeo vòng nào",
])
def test_no_physical_store_invite(api, user_token, q):
    """Shop online-only: TUYỆT ĐỐI không mời khách ghé cửa hàng vật lý."""
    d = api.chat(user_token, api.sid(), q)
    _ok(d["response"])
    assert not re.search(r"ghé (thăm|qua)[^.]{0,15}(cửa hàng|shop)|đến cửa hàng|thăm cửa hàng|ghé website",
                         d["response"].lower()), f"mời ghé cửa hàng (sai online-only): {d['response'][:160]}"


def test_menh_context_carry_over(api, user_token):
    """Nhớ mệnh giữa các lượt: cho năm sinh rồi hỏi tiếp KHÔNG bị hỏi lại năm sinh."""
    sid = api.sid()
    api.chat(user_token, sid, "tôi sinh năm 1990 hợp màu gì")
    d = api.chat(user_token, sid, "shop còn mẫu vòng nào khác hợp mình không")
    _ok(d["response"])
    # không được hỏi lại năm sinh (đã có trong ngữ cảnh)
    assert not re.search(r"cho.*(biết|xin).*năm sinh|bạn sinh năm nào", d["response"].lower()), \
        f"hỏi lại năm sinh dù đã có ngữ cảnh mệnh: {d['response'][:160]}"
