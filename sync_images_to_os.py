"""
sync_images_to_os.py – Embed TẤT CẢ ảnh sản phẩm (SigLIP 2) và đẩy vector lên
OpenSearch image index để phục vụ VISUAL SEARCH.

Nguồn product_id ↔ ảnh: lấy thẳng từ Postgres (db_service.get_all_products),
cột `image` dạng {'cover': url, 'images': [{'url': ...}, ...]}.

Tái dùng cache bytes ảnh (output/image_bytes_cache.pkl) nếu có để khỏi tải lại.
Mỗi ẢNH thành 1 doc kNN: {product_id, image_url, is_cover, image_embedding}.

Chạy:
    python sync_images_to_os.py            # build/refresh index ảnh (xoá & tạo lại)
    python sync_images_to_os.py --keep     # giữ index cũ, chỉ upsert thêm
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "langraph pipeline"))
import image_embedding as IE          # noqa: E402
import db_service                     # noqa: E402
import opensearch_service as oss      # noqa: E402

BYTES_PATH = "output/image_bytes_cache.pkl"


def _extract_urls(image) -> list[tuple[str, bool]]:
    """Lấy [(url, is_cover)] từ cột image của 1 sản phẩm (dict hoặc list)."""
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
    # dedupe theo url, giữ thứ tự (cover ưu tiên)
    seen, uniq = set(), []
    for u, c in out:
        if u not in seen:
            seen.add(u)
            uniq.append((u, c))
    return uniq


def _load_bytes() -> dict:
    if os.path.exists(BYTES_PATH):
        with open(BYTES_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def main(keep: bool = False, delay: float = 0.6):
    # 1) Gom mọi (product_id, url, is_cover) từ DB
    products = db_service.get_all_products()
    items = []
    for p in products:
        for url, is_cover in _extract_urls(p.image):
            items.append((p.product_id, url, is_cover))
    print(f"DB: {len(products)} sản phẩm | {len(items)} ảnh (đã dedupe theo SP).")

    # 2) Đảm bảo có bytes ảnh (tải cái còn thiếu, có delay tránh rate-limit)
    bcache = _load_bytes()
    need = [(pid, u, c) for (pid, u, c) in items if u not in bcache]
    print(f"Tải {len(need)} ảnh mới (đã cache {len(bcache)}), delay={delay}s.")
    fail = 0
    for i, (pid, u, c) in enumerate(need, 1):
        b = IE.download_bytes(u)
        if b:
            bcache[u] = b
        else:
            fail += 1
        if delay > 0:
            time.sleep(delay)
        if i % 20 == 0 or i == len(need):
            with open(BYTES_PATH, "wb") as f:
                pickle.dump(bcache, f)
            print(f"  tải {i}/{len(need)} (lỗi {fail})")

    # 3) Embed mỗi URL DUY NHẤT một lần (nhiều sản phẩm có thể share URL)
    uniq_urls = []
    seen = set()
    for _, u, _ in items:
        if u not in seen:
            seen.add(u)
            uniq_urls.append(u)
    print(f"Embed {len(uniq_urls)} ảnh duy nhất ({len(items)} cặp SP-ảnh). Model: {IE.model_id()}")
    vec_by_url: dict[str, list] = {}
    for i, u in enumerate(uniq_urls, 1):
        b = bcache.get(u)
        if not b:
            continue
        try:
            vec_by_url[u] = IE.embed_image(b).tolist()
        except Exception:
            pass
        if i % 50 == 0 or i == len(uniq_urls):
            print(f"  embed {i}/{len(uniq_urls)}")

    # Dựng doc cho TỪNG cặp (sản phẩm, ảnh)
    docs, emb_fail = [], 0
    for pid, u, c in items:
        v = vec_by_url.get(u)
        if v is None:
            emb_fail += 1
            continue
        docs.append({
            "product_id": int(pid),
            "image_url":  u,
            "is_cover":   bool(c),
            "embedding":  v,
        })

    # 4) Tạo index + đẩy lên OpenSearch
    oss.create_image_index(delete_existing=not keep)
    oss.bulk_index_image_vectors(docs)
    print(f"\nXong. docs đẩy lên: {len(docs)} | bỏ qua (thiếu bytes/lỗi embed): {emb_fail}")
    print(f"Index ảnh '{oss.config.OS_IMAGE_INDEX}' hiện có: {oss.get_image_doc_count()} vector.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="giữ index cũ, chỉ upsert thêm")
    ap.add_argument("--delay", type=float, default=0.6)
    args = ap.parse_args()
    main(keep=args.keep, delay=args.delay)
