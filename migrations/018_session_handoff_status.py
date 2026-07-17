"""
018_session_handoff_status.py – Trạng thái phiên cho cơ chế chuyển cho chủ shop.

Thêm cột chat_sessions.status:
  - 'bot'           : bot tự trả lời (mặc định)
  - 'pending_admin' : đã chuyển cho chủ shop, đang chờ shop trả lời (bot NGỪNG)
  - 'admin'         : chủ shop đã/đang trả lời trực tiếp (bot NGỪNG)

Chạy:  docker exec fengshui_chatbot python /app/migrations/018_session_handoff_status.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from models import get_engine  # noqa: E402


DDL = [
    "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'bot'",
    "CREATE INDEX IF NOT EXISTS idx_chat_sessions_status ON chat_sessions(status, updated_at DESC)",
]


def main():
    engine = get_engine()
    with engine.begin() as conn:
        for stmt in DDL:
            conn.execute(text(stmt))
    print("  ✓ chat_sessions.status đã sẵn sàng (bot / pending_admin / admin)")


if __name__ == "__main__":
    main()
