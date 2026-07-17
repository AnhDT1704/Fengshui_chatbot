"""
016_set_product_size_treoxe.py – Điền product_size (kích thước) cho 4 sản phẩm treo xe.

product_size KHÔNG dùng cho search; dữ liệu hiển thị lấy lại từ PG → KHÔNG cần re-index.

Chạy:  docker exec fengshui_chatbot python /app/migrations/016_set_product_size_treoxe.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Product, get_session  # noqa: E402

# product_id -> product_size
UPDATES = {
    8:  ["5cm x 5cm x 0,7cm"],
    24: ["6cm x 4cm x 0,5cm"],
    52: ["7cm x 3cm x 0,5cm"],
    55: ["2.5 cm x 2 mm"],
}


def main():
    session = get_session()
    done, missing = [], []
    try:
        for pid, size in UPDATES.items():
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
