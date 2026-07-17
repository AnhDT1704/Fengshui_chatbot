"""
015_dedupe_products.py – Xóa các hàng TRÙNG (cùng 1 sản phẩm bị lặp nhiều hàng),
mỗi sản phẩm chỉ giữ 1 hàng đại diện.

Giữ → Xóa:
  - Vòng mã não bện dây:   giữ 13  → xóa 14,16,18,25,44,45,46,48
  - Vòng mix 2in1:         giữ 1   → xóa 4,12
  - Nhang nụ tháp:         giữ 42  → xóa 97
  - Nhang nụ trầm:         giữ 23  → xóa 70
  - Nhang sạch:            giữ 63  → xóa 102  (copy product_size đầy đủ 102→63 trước)

Sau script cần: 010 (dồn id), re-index semantic, xóa image index của các id đã xóa.

Chạy: docker exec fengshui_chatbot python /app/migrations/015_dedupe_products.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Product, get_session  # noqa: E402

# keeper product_id -> list product_id cần xóa
KEEP_DELETE = {
    13: [14, 16, 18, 25, 44, 45, 46, 48],
    1:  [4, 12],
    42: [97],
    23: [70],
    63: [102],
}
# (keeper, source) — copy product_size từ source sang keeper nếu source đầy đủ hơn
SIZE_COPY = (63, 102)


def main():
    session = get_session()
    try:
        # 1) Giữ product_size đầy đủ hơn cho nhang sạch (63) trước khi xóa 102.
        keep = session.query(Product).filter_by(product_id=SIZE_COPY[0]).first()
        src = session.query(Product).filter_by(product_id=SIZE_COPY[1]).first()
        if keep and src and len(src.product_size or []) > len(keep.product_size or []):
            print(f"  product_size: copy {src.product_size} (pid {SIZE_COPY[1]}) "
                  f"-> pid {SIZE_COPY[0]} (cũ: {keep.product_size})")
            keep.product_size = list(src.product_size)

        # 2) Xóa các hàng dư.
        to_delete = sorted(pid for dels in KEEP_DELETE.values() for pid in dels)
        deleted, missing = [], []
        for pid in to_delete:
            p = session.query(Product).filter_by(product_id=pid).first()
            if p is None:
                missing.append(pid)
                continue
            session.delete(p)
            deleted.append(pid)

        session.commit()
        print(f"Đã xóa {len(deleted)} hàng: {deleted}")
        if missing:
            print(f"Không tìm thấy (bỏ qua): {missing}")
        print(f"Tổng sản phẩm còn lại: {session.query(Product).count()}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
