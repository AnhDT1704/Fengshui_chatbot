"""
010_reorder_by_category.py – Gán lại khóa chính `id` của bảng products để các sản
phẩm CÙNG category nằm ở các hàng liền nhau (xem bảng cho gọn).

QUAN TRỌNG:
- Chỉ đổi `id` (PK surrogate, không FK nào trỏ tới, runtime KHÔNG dùng — chatbot tra
  theo `product_id`). GIỮ NGUYÊN `product_id`, nội dung, ảnh.
- KHÔNG cần đồng bộ OpenSearch: chunks/docs khóa theo product_id (không đổi),
  embedding là của product_description (không đổi).

Thứ tự: gom theo CATEGORY_ORDER, trong mỗi nhóm sắp theo product_id tăng dần.
Category không có trong danh sách → xếp cuối.

An toàn: chạy trong 1 transaction; offset id +100000 trước để tránh đụng PK.
Idempotent: chạy lại cho ra cùng thứ tự (deterministic).

Chạy:  docker exec fengshui_chatbot python /app/migrations/010_reorder_by_category.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402
from models import get_engine  # noqa: E402


# Thứ tự nhóm (đã được shop duyệt). "khác" để cuối.
CATEGORY_ORDER = [
    "vòng tay",
    "nhang",
    "treo xe",
    "lư xông trầm",
    "chuỗi hạt",
    "thác khói",
    "tượng phật",
    "dây chuyền",
    "nước lau",
    "khác",
]

_OFFSET = 100000


def _rank(category: str) -> int:
    try:
        return CATEGORY_ORDER.index(category)
    except ValueError:
        return len(CATEGORY_ORDER)  # category lạ → cuối


def main():
    engine = get_engine()
    with engine.begin() as conn:
        total = conn.execute(text("SELECT count(*) FROM products")).scalar_one()

        # 1) Đẩy toàn bộ id ra khỏi vùng 1..N để khỏi đụng PK khi gán lại.
        conn.execute(text(f"UPDATE products SET id = id + {_OFFSET}"))

        # 2) Lấy thứ tự mong muốn (theo category rank, rồi product_id).
        rows = conn.execute(
            text("SELECT id, category, product_id FROM products")
        ).fetchall()
        rows_sorted = sorted(rows, key=lambda r: (_rank(r.category), r.product_id))

        # 3) Gán lại id = 1..N theo thứ tự đó (id hiện tại đang là id+OFFSET).
        for new_id, r in enumerate(rows_sorted, start=1):
            conn.execute(
                text("UPDATE products SET id = :nid WHERE id = :oid"),
                {"nid": new_id, "oid": r.id},
            )

        # 4) Đồng bộ lại sequence để insert sau này không trùng.
        conn.execute(
            text("SELECT setval('products_id_seq', (SELECT max(id) FROM products))")
        )

    print(f"Đã gán lại id cho {total} sản phẩm theo category.")
    # In thử vài hàng đầu mỗi nhóm để kiểm tra
    with engine.connect() as conn:
        sample = conn.execute(
            text("SELECT id, category, product_id, left(name,40) AS name "
                 "FROM products ORDER BY id")
        ).fetchall()
    last_cat = None
    for r in sample:
        if r.category != last_cat:
            print(f"  ── {r.category} ──")
            last_cat = r.category
        print(f"    id={r.id:<3} product_id={r.product_id:<3} {r.name}")


if __name__ == "__main__":
    main()
