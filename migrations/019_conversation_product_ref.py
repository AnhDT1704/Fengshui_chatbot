"""
019_conversation_product_ref.py – Gắn metadata SẢN PHẨM vào tin nhắn để nhớ ngữ cảnh.

Thêm cột conversation_log.product_ref (JSONB) = danh sách sản phẩm mà lượt đó đã xác
định/trình bày, dạng [{"id": <product_id>, "name": "<tên>"}, ...].
Nhờ vậy các lượt sau biết "sản phẩm này" khách đang nói tới là product_id nào → không
phải đoán id (sửa lỗi quên sản phẩm giữa phiên).

Chạy:  docker exec fengshui_chatbot python /app/migrations/019_conversation_product_ref.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from models import get_engine  # noqa: E402


def main():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE conversation_log ADD COLUMN IF NOT EXISTS product_ref JSONB"))
    print("  ✓ conversation_log.product_ref đã sẵn sàng")


if __name__ == "__main__":
    main()
