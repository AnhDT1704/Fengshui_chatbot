"""Populate PostgreSQL product image URLs from danh_sach_hinh_anh_san_pham.txt."""

import json
import sys

from db_service import populate_product_images_from_file


def main() -> int:
    image_file = sys.argv[1] if len(sys.argv) > 1 else r"D:\fengshui_data_pipeline\danh_sach_hinh_anh_san_pham.txt"
    result = populate_product_images_from_file(image_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())