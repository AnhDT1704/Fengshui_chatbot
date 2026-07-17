"""
012_update_pid107_name_image.py – Cập nhật TÊN + cột IMAGE cho product_id=107
(Lư điện xông trầm hương).

Sau script này:
  - name đổi → nên re-index OpenSearch semantic (pipeline --steps 6 --index-from-postgres).
  - image đổi → KHÔNG ảnh hưởng semantic index; chỉ ảnh hưởng index ẢNH (visual search)
    nếu muốn nhận diện sản phẩm qua ảnh mới → cần sync lại index ảnh (tùy chọn).

Chạy:  docker exec fengshui_chatbot python /app/migrations/012_update_pid107_name_image.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402
from models import get_engine  # noqa: E402

PID = 107
NEW_NAME = ("Lư điện xông đốt trầm hương Vạn An Group, có hẹn giờ, điều khiển nhiệt độ "
            "dùng xông trầm miếng, dăm trầm, trầm bột")
NEW_IMAGE = {
    "cover": "https://down-vn.img.susercontent.com/file/vn-11134207-7ras8-m519gwq2tmxj7d.webp",
    "images": [
        {"url": "https://down-vn.img.susercontent.com/file/vn-11134207-7r98o-lldn4261yaiwb4.webp", "color": None},
        {"url": "https://down-vn.img.susercontent.com/file/vn-11134207-7r98o-lldn4261zp3cfd.webp", "color": None},
        {"url": "https://down-vn.img.susercontent.com/file/vn-11134207-7r98o-lldn426213nsca.webp", "color": None},
        {"url": "https://down-vn.img.susercontent.com/file/vn-11134207-7r98o-lldn42622i8819.webp", "color": None},
    ],
}


def main():
    engine = get_engine()
    with engine.begin() as conn:
        before = conn.execute(
            text("SELECT id, product_id, name, image->>'cover' AS cover FROM products WHERE product_id=:p"),
            {"p": PID},
        ).fetchone()
        if before is None:
            print(f"KHÔNG tìm thấy product_id={PID}")
            sys.exit(1)
        print("TRƯỚC:", before.id, "|", before.name)
        print("       cover cũ:", before.cover)

        conn.execute(
            text("UPDATE products SET name=:n, image=CAST(:img AS jsonb), updated_at=now() "
                 "WHERE product_id=:p"),
            {"n": NEW_NAME, "img": json.dumps(NEW_IMAGE, ensure_ascii=False), "p": PID},
        )

        after = conn.execute(
            text("SELECT name, image->>'cover' AS cover, "
                 "jsonb_array_length(image->'images') AS n_imgs "
                 "FROM products WHERE product_id=:p"),
            {"p": PID},
        ).fetchone()
        print("SAU  :", after.name)
        print("       cover mới:", after.cover)
        print("       số ảnh trong 'images':", after.n_imgs)


if __name__ == "__main__":
    main()
