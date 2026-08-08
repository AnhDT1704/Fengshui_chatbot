"""
test_admin_api.py – Integration test các tính năng TÀI KHOẢN ADMIN (chủ shop).

Bao phủ: phân quyền admin, cấu hình size_mode, danh sách handoff/users, máy trạng
thái handoff (bot → admin → bot), tải template Excel, nạp Excel (round-trip idempotent),
và từ chối file rác. Round-trip dùng template có sẵn dữ liệu THẬT → không đổi dữ liệu shop.
"""

import pytest

from conftest import XLSX_MIME, requires_api

pytestmark = [pytest.mark.integration, requires_api]


def test_admin_endpoints_require_admin(api, user_token):
    # user thường → 403
    assert api.get("/admin/settings", user_token).status_code == 403
    assert api.get("/admin/handoffs", user_token).status_code == 403
    # không token → 401
    assert api.get_public("/admin/settings").status_code == 401


def test_settings_get_put_validate(api, admin_token):
    r = api.get("/admin/settings", admin_token)
    assert r.status_code == 200
    assert r.json().get("size_mode") in ("code", "finetune")

    # đổi sang finetune rồi đọc lại
    assert api.put("/admin/settings", admin_token, json={"size_mode": "finetune"}).json()["size_mode"] == "finetune"
    assert api.get("/admin/settings", admin_token).json()["size_mode"] == "finetune"

    # giá trị sai → 400
    assert api.put("/admin/settings", admin_token, json={"size_mode": "bừa"}).status_code == 400

    # trả lại code (mặc định)
    assert api.put("/admin/settings", admin_token, json={"size_mode": "code"}).json()["size_mode"] == "code"


def test_handoffs_and_users_lists(api, admin_token):
    assert isinstance(api.get("/admin/handoffs", admin_token).json().get("sessions"), list)
    assert isinstance(api.get("/admin/users", admin_token).json().get("users"), list)


def test_templates_downloadable(api, admin_token):
    for path in (
        "/admin/template/products",
        "/admin/template/promotions",
        "/admin/template/catalog",
    ):
        r = api.get(path, admin_token)
        assert r.status_code == 200, path
        assert len(r.content) > 500
        assert r.content[:2] == b"PK", "không phải file xlsx (zip)"


def test_import_products_roundtrip(api, admin_token):
    """Tải template (dữ liệu THẬT) → nạp lại NGUYÊN → idempotent, không đổi dữ liệu."""
    tpl = api.get("/admin/template/products", admin_token).content
    r = api.post("/admin/import/products", admin_token,
                 files={"file": ("gia_tonkho.xlsx", tpl, XLSX_MIME)})
    assert r.status_code == 200, r.text


def test_import_promotions_roundtrip(api, admin_token):
    tpl = api.get("/admin/template/promotions", admin_token).content
    r = api.post("/admin/import/promotions", admin_token,
                 files={"file": ("khuyen_mai.xlsx", tpl, XLSX_MIME)})
    assert r.status_code == 200, r.text


def test_import_bad_file_rejected(api, admin_token):
    """File rác → phải từ chối (400/500), KHÔNG âm thầm nuốt."""
    r = api.post("/admin/import/products", admin_token,
                 files={"file": ("bad.xlsx", b"day khong phai excel", XLSX_MIME)})
    assert r.status_code in (400, 500)


def test_handoff_state_machine(api, user_token, admin_token):
    """bot → (admin reply) admin → bot IM LẶNG → (return-to-bot) bot trả lời lại."""
    sid = api.sid()
    api.chat(user_token, sid, "xin chào")  # tạo phiên (thuộc user)

    # admin trả lời trực tiếp → khoá phiên ở 'admin'
    r = api.post("/admin/reply", admin_token, json={"session_id": sid, "message": "Dạ chủ shop nghe ạ"})
    assert r.status_code == 200, r.text
    assert r.json()["session_status"] == "admin"

    # bot IM LẶNG: user chat tiếp → agent_used='handoff'
    d = api.chat(user_token, sid, "shop còn hàng không")
    assert d["agent_used"] == "handoff", f"bot vẫn tự trả lời khi đã handoff: {d}"

    # trả phiên về bot
    assert api.post("/admin/sessions/" + sid + "/return-to-bot", admin_token).status_code == 200
    d2 = api.chat(user_token, sid, "xin chào lại")
    assert d2["agent_used"] != "handoff", "phiên chưa trả về bot"
