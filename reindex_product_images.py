"""
reindex_product_images.py – Đồng bộ index ẢNH (visual search) cho MỘT sản phẩm,
KHÔNG rebuild toàn bộ index. Dùng khi vừa đổi cột `image` của 1 sản phẩm trong DB.

Việc làm:
  1) Xóa các doc ảnh CŨ của product_id đó khỏi index ảnh (delete_by_query).
  2) Đọc ảnh hiện tại của sản phẩm từ Postgres → download + embed (SigLIP) → index lại.

Chạy:
    docker exec fengshui_chatbot python /app/reindex_product_images.py --product-id 107
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "langraph pipeline"))
import image_embedding as IE          # noqa: E402
import db_service                     # noqa: E402
import opensearch_service as oss      # noqa: E402


def _extract_urls(image) -> list[tuple[str, bool]]:
    """[(url, is_cover)] từ cột image (dict {'cover',  'images':[{url}]} hoặc list)."""
    out: list[tuple[str, bool]] = []
    if isinstance(image, dict):
        cover = image.get("cover")
        if cover:
            out.append((cover, True))
        for im in image.get("images", []) or []:
            u = im.get("url") if isinstance(im, dict) else im
            if u:
                out.append((u, False))
    elif isinstance(image, list):
        for i, u in enumerate(image):
            if isinstance(u, str):
                out.append((u, i == 0))
    seen, uniq = set(), []
    for u, c in out:
        if u not in seen:
            seen.add(u)
            uniq.append((u, c))
    return uniq


def main(pid: int):
    p = db_service.get_product_by_id(pid)
    if p is None:
        print(f"KHÔNG tìm thấy product_id={pid}")
        return
    urls = _extract_urls(p.image)
    print(f"product_id={pid}: {len(urls)} ảnh hiện tại trong DB.")

    client = oss.get_client()

    # 1) Xóa doc ảnh CŨ của sản phẩm này.
    try:
        r = client.delete_by_query(
            index=oss.config.OS_IMAGE_INDEX,
            body={"query": {"term": {"product_id": pid}}},
            refresh=True,
        )
        print(f"Đã xóa {r.get('deleted', 0)} doc ảnh cũ của product_id={pid}.")
    except Exception as e:
        print(f"delete_by_query lỗi: {e}")

    # 2) Download + embed + index ảnh hiện tại.
    print(f"Embed model: {IE.model_id()}")
    docs = []
    for u, is_cover in urls:
        b = IE.download_bytes(u)
        if not b:
            print(f"  ⚠ tải lỗi: {u}")
            continue
        try:
            vec = IE.embed_image(b).tolist()
        except Exception as e:
            print(f"  ⚠ embed lỗi ({e}): {u}")
            continue
        docs.append({
            "product_id": int(pid),
            "image_url":  u,
            "is_cover":   bool(is_cover),
            "embedding":  vec,
        })

    if docs:
        oss.bulk_index_image_vectors(docs)
    print(f"Xong. Index {len(docs)}/{len(urls)} ảnh cho product_id={pid}. "
          f"Tổng index ảnh: {oss.get_image_doc_count()} vector.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--product-id", type=int, required=True)
    args = ap.parse_args()
    main(args.product_id)
