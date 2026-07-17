"""
017_auth_and_user_history.py – Tài khoản (users) + lịch sử trò chuyện theo USER.

Thêm:
  - users        : tài khoản (username + password hash)
  - auth_tokens  : token đăng nhập (lưu ở client, gửi qua header Authorization)
  - chat_sessions: danh sách phiên trò chuyện THEO user (thay cho localStorage)
  - conversation_log: thêm cột user_id + images (URL ảnh đã lưu trên server) để
    sau khi reload vẫn hiển thị lại được ảnh khách đã gửi.

Idempotent (CREATE/ADD IF NOT EXISTS) — chạy lại nhiều lần vô hại.

Chạy:  docker exec fengshui_chatbot python /app/migrations/017_auth_and_user_history.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from models import get_engine  # noqa: E402


DDL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id            SERIAL PRIMARY KEY,
        username      VARCHAR(64) UNIQUE NOT NULL,
        password_hash TEXT        NOT NULL,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_tokens (
        token      TEXT PRIMARY KEY,
        user_id    INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens(user_id)",
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        id         TEXT PRIMARY KEY,
        user_id    INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title      TEXT        NOT NULL DEFAULT 'Cuộc trò chuyện mới',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id, updated_at DESC)",
    # conversation_log có thể đã tồn tại (tạo từ trước). Tạo nếu chưa có cho DB sạch.
    """
    CREATE TABLE IF NOT EXISTS conversation_log (
        id           SERIAL PRIMARY KEY,
        session_id   TEXT NOT NULL,
        role         TEXT NOT NULL,
        content      TEXT,
        agent_used   TEXT,
        intent       TEXT,
        tools_called TEXT[],
        created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "ALTER TABLE conversation_log ADD COLUMN IF NOT EXISTS user_id INTEGER",
    "ALTER TABLE conversation_log ADD COLUMN IF NOT EXISTS images  JSONB",
    "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_log(session_id, created_at)",
]


def main():
    engine = get_engine()
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
    print("  ✓ Schema auth + lịch sử theo user đã sẵn sàng")


if __name__ == "__main__":
    main()
