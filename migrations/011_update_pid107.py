"""
011_update_pid107.py – Đổi TÊN + CATEGORY cho sản phẩm product_id=107
(Lư điện xông trầm hương) và chuyển sang nhóm "lư xông trầm".

Sau script này cần:
  - chạy lại 010 để gom nhóm (đưa sp vào dải id của 'lư xông trầm'),
  - re-index OpenSearch (name + category đổi → search cần cập nhật).

Chạy:  docker exec fengshui_chatbot python /app/migrations/011_update_pid107.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402
from models import get_engine  # noqa: E402

PID = 107
NEW_NAME = ("Lư điện xông đốt trầm hương hoa sen có hẹn giờ, điều khiển nhiệt độ, "
            "dùng xông trầm miếng, trầm bột")
NEW_CATEGORY = "lư xông trầm"


def main():
    engine = get_engine()
    with engine.begin() as conn:
        before = conn.execute(
            text("SELECT id, product_id, category, name FROM products WHERE product_id=:p"),
            {"p": PID},
        ).fetchone()
        if before is None:
            print(f"KHÔNG tìm thấy product_id={PID}")
            sys.exit(1)
        print("TRƯỚC:", before.id, "|", before.category, "|", before.name)

        conn.execute(
            text("UPDATE products SET name=:n, category=:c, updated_at=now() "
                 "WHERE product_id=:p"),
            {"n": NEW_NAME, "c": NEW_CATEGORY, "p": PID},
        )

        after = conn.execute(
            text("SELECT id, product_id, category, name FROM products WHERE product_id=:p"),
            {"p": PID},
        ).fetchone()
        print("SAU  :", after.id, "|", after.category, "|", after.name)


if __name__ == "__main__":
    main()
