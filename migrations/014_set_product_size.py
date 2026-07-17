"""
014_set_product_size.py – Điền cột product_size cho một số sản phẩm.

LƯU Ý: product_size KHÔNG dùng cho search/filter và dữ liệu hiển thị luôn lấy lại
từ Postgres (qua _enrich_with_pg) → KHÔNG cần re-index OpenSearch.

Chạy:  docker exec fengshui_chatbot python /app/migrations/014_set_product_size.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Product, get_session  # noqa: E402

# (danh sách product_id, giá trị product_size)
UPDATES = [
    ([23, 42],     ["4 cm"]),
    ([32, 54, 56], ["10cm x cao 12cm x rộng cả đế 12cm"]),
    ([19],         ["30cm"]),
]


def main():
    session = get_session()
    done, missing = [], []
    try:
        for pids, size in UPDATES:
            for pid in pids:
                p = session.query(Product).filter(Product.product_id == pid).first()
                if p is None:
                    missing.append(pid)
                    continue
                p.product_size = list(size)
                done.append((pid, size))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    for pid, size in done:
        print(f"  product_id={pid:<4} product_size={size}")
    if missing:
        print(f"Không tìm thấy: {missing}")


if __name__ == "__main__":
    main()
