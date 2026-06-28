"""
auth.py – Đăng ký / đăng nhập / token cho UI chatbot.

Mật khẩu băm bằng PBKDF2-HMAC-SHA256 (thư viện chuẩn, KHÔNG cần thêm dependency).
Token đăng nhập là chuỗi ngẫu nhiên, lưu trong bảng auth_tokens; client giữ token
ở localStorage và gửi qua header `Authorization: Bearer <token>`.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import hashlib
import os
import secrets
from typing import Optional

from sqlalchemy import text

from logger import get_logger
from models import get_engine

log = get_logger("auth")

_ALGO  = "pbkdf2_sha256"
_ITERS = 200_000

# Tài khoản chủ shop (admin). Đặt qua env ADMIN_USERNAMES (phân tách bằng dấu phẩy),
# mặc định "admin1". User đăng nhập trùng tên này = chủ shop, nhận các phiên chuyển tới.
_ADMIN_USERNAMES = {
    u.strip().lower() for u in os.getenv("ADMIN_USERNAMES", "admin1").split(",") if u.strip()
}


def is_admin(username: Optional[str]) -> bool:
    return (username or "").strip().lower() in _ADMIN_USERNAMES


# ═══════════════════════════════════════════════════════════════════
#  PASSWORD HASHING (stdlib)
# ═══════════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _ITERS)
    return f"{_ALGO}${_ITERS}${salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt, digest = stored.split("$")
        if algo != _ALGO:
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iters))
        return secrets.compare_digest(dk.hex(), digest)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════
#  USERS
# ═══════════════════════════════════════════════════════════════════

def register_user(username: str, password: str) -> dict:
    """Tạo user mới. Raise ValueError nếu input không hợp lệ / trùng tên."""
    username = (username or "").strip()
    if len(username) < 3:
        raise ValueError("Tên đăng nhập tối thiểu 3 ký tự")
    if len(password or "") < 6:
        raise ValueError("Mật khẩu tối thiểu 6 ký tự")

    engine = get_engine()
    with engine.begin() as conn:
        dup = conn.execute(
            text("SELECT 1 FROM users WHERE lower(username) = lower(:u)"), {"u": username}
        ).first()
        if dup:
            raise ValueError("Tên đăng nhập đã tồn tại")
        uid = conn.execute(
            text("INSERT INTO users (username, password_hash) VALUES (:u, :p) RETURNING id"),
            {"u": username, "p": hash_password(password)},
        ).scalar_one()
    log.info("Đăng ký user mới id=%d username=%s", uid, username)
    return {"id": uid, "username": username}


def login(username: str, password: str) -> Optional[dict]:
    """Trả {id, username} nếu đúng mật khẩu, None nếu sai."""
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, password_hash FROM users WHERE lower(username) = lower(:u)"),
            {"u": (username or "").strip()},
        ).first()
    if not row or not verify_password(password or "", row[2]):
        return None
    return {"id": row[0], "username": row[1]}


# ═══════════════════════════════════════════════════════════════════
#  TOKENS
# ═══════════════════════════════════════════════════════════════════

def create_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO auth_tokens (token, user_id) VALUES (:t, :u)"),
            {"t": token, "u": user_id},
        )
    return token


def user_for_token(token: str) -> Optional[dict]:
    """Trả {id, username} cho token hợp lệ, None nếu không có."""
    if not token:
        return None
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT u.id, u.username
                FROM auth_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE t.token = :t
            """),
            {"t": token},
        ).first()
    return {"id": row[0], "username": row[1]} if row else None


def revoke_token(token: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM auth_tokens WHERE token = :t"), {"t": token})
