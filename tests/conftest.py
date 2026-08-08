"""
conftest.py – fixtures dùng chung cho test chatbot phong thủy.

- Thêm "langraph pipeline" vào sys.path để import module (skills_agent, memory, graph...).
- Cung cấp fixture: base_url, user_token, admin_token, và helper `api` (gọi HTTP).
- Integration tests tự SKIP nếu API không chạy tại CHATBOT_TEST_URL (mặc định localhost:8000).

Chạy trong container:
    docker exec fengshui_chatbot python -m pytest /app/tests -v
"""

from __future__ import annotations

import itertools
import os
import sys
import time

import pytest
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "langraph pipeline"))

BASE_URL = os.getenv("CHATBOT_TEST_URL", "http://localhost:8000").rstrip("/")
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_counter = itertools.count()


def _api_up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


API_UP = _api_up()
requires_api = pytest.mark.skipif(not API_UP, reason=f"chatbot API không chạy tại {BASE_URL}")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: test cần API chatbot đang chạy")


class _API:
    """Helper mỏng bọc requests cho các test integration."""

    def __init__(self, base: str):
        self.base = base

    def sid(self) -> str:
        return f"pt_{int(time.time())}_{next(_counter)}"

    def _h(self, token):
        return {"Authorization": f"Bearer {token}"}

    # public (không auth)
    def get_public(self, path):
        return requests.get(self.base + path, timeout=15)

    def raw_post(self, path, **kw):
        return requests.post(self.base + path, timeout=20, **kw)

    # có auth
    def get(self, path, token, timeout=30):
        return requests.get(self.base + path, headers=self._h(token), timeout=timeout)

    def put(self, path, token, json=None, timeout=30):
        return requests.put(self.base + path, headers=self._h(token), json=json, timeout=timeout)

    def post(self, path, token, json=None, files=None, timeout=120):
        return requests.post(self.base + path, headers=self._h(token),
                             json=json, files=files, timeout=timeout)

    def chat(self, token, sid, message, timeout=220) -> dict:
        r = requests.post(self.base + "/chat",
                          json={"session_id": sid, "message": message},
                          headers=self._h(token), timeout=timeout)
        r.raise_for_status()
        return r.json()


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def api():
    return _API(BASE_URL)


@pytest.fixture(scope="session")
def user_token():
    u = f"pytest_user_{int(time.time())}"
    r = requests.post(f"{BASE_URL}/auth/register",
                      json={"username": u, "password": "pytest123"}, timeout=15)
    r.raise_for_status()
    return r.json()["token"]


@pytest.fixture(scope="session")
def admin_token():
    """Token cho tài khoản ADMIN (mặc định 'admin1'); tạo user nếu chưa có."""
    import auth
    from sqlalchemy import text
    from models import get_engine

    admin_name = sorted(auth._ADMIN_USERNAMES)[0] if auth._ADMIN_USERNAMES else "admin1"
    with get_engine().begin() as c:
        row = c.execute(text("SELECT id FROM users WHERE lower(username)=lower(:u)"),
                        {"u": admin_name}).first()
    uid = row[0] if row else auth.register_user(admin_name, "admin_pytest_pw")["id"]
    assert auth.is_admin(admin_name), "cấu hình admin sai — is_admin trả False"
    return auth.create_token(uid)
