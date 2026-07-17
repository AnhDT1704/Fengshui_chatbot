"""
test_image_search.py – POC kiểm chứng visual search (SigLIP 2 + cosine).

Cách dùng:
  # 1) Build cache embedding cho toàn bộ ảnh sản phẩm (chạy 1 lần, có resume)
  python test_image_search.py --build

  # 2) Hold-out: giữ lại 1 ảnh của 1 sản phẩm, query bằng nó, xem có ra đúng SP
  python test_image_search.py --holdout "mã não đa sắc hạt bánh xe"

  # 3) Query bằng ảnh thật của khách (file hoặc URL)
  python test_image_search.py --query-file test_query.jpg
  python test_image_search.py --query-url https://...

Scoring: điểm 1 sản phẩm = MAX cosine giữa ảnh query và TỪNG ảnh của sản phẩm đó.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "langraph pipeline"))
import image_embedding as IE  # noqa: E402

CATALOG_PATH = "output/product_images.json"
CACHE_PATH   = "output/image_emb_cache.pkl"      # url -> vector (embedding)
BYTES_PATH   = "output/image_bytes_cache.pkl"    # url -> raw image bytes


def _load_catalog():
    return json.load(open(CATALOG_PATH, encoding="utf-8"))


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def _load_bytes() -> dict:
    if os.path.exists(BYTES_PATH):
        with open(BYTES_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def build_index(limit_per_product: int | None = None):
    """2 pha: (1) tải bytes ảnh (cache, có retry); (2) embed từ bytes.
    Embed luôn tính lại để phản ánh thay đổi model/pooling — KHÔNG cần tải lại."""
    catalog = _load_catalog()
    urls = []
    for prod in catalog:
        us = prod["urls"][:limit_per_product] if limit_per_product else prod["urls"]
        urls.extend(us)

    # --- Pha 1: tải bytes (chỉ tải URL chưa có), có DELAY tránh Shopee rate-limit ---
    import time
    delay = float(os.getenv("IMG_DL_DELAY", "0.5"))  # giây giữa các request
    bcache = _load_bytes()
    need = [u for u in urls if u not in bcache]
    print(f"[tải] {len(need)} ảnh mới (đã có bytes {len(bcache)}), delay={delay}s.", flush=True)
    fail = 0
    for i, u in enumerate(need, 1):
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
            print(f"  tải {i}/{len(need)} (lỗi {fail})", flush=True)

    # --- Pha 2: embed từ bytes (tính lại toàn bộ) ---
    print(f"[embed] {len(bcache)} ảnh. Model: {IE.model_id()}", flush=True)
    vcache = {}
    for i, (u, b) in enumerate(bcache.items(), 1):
        try:
            vcache[u] = IE.embed_image(b)
        except Exception:
            pass
        if i % 40 == 0 or i == len(bcache):
            _save_cache(vcache)
            print(f"  embed {i}/{len(bcache)}", flush=True)
    _save_cache(vcache)
    print(f"Xong. bytes={len(bcache)} | vector={len(vcache)} | dim={next(iter(vcache.values())).shape}", flush=True)
    return vcache


def _rank(query_vec: np.ndarray, exclude_urls: set[str] | None = None, top=10):
    catalog = _load_catalog()
    cache = _load_cache()
    exclude_urls = exclude_urls or set()
    scored = []
    for prod in catalog:
        best = -1.0
        for u in prod["urls"]:
            if u in exclude_urls:
                continue
            v = cache.get(u)
            if v is None:
                continue
            s = IE.cosine(query_vec, v)
            if s > best:
                best = s
        if best > -1.0:
            scored.append((best, prod["name"]))
    scored.sort(reverse=True)
    return scored[:top]


def _print_ranking(scored, target_sub: str | None = None):
    print("\n=== TOP MATCH ===")
    for rank, (score, name) in enumerate(scored, 1):
        hit = "  <<< TARGET" if (target_sub and target_sub.lower() in name.lower()) else ""
        print(f"  #{rank}  {score:.4f}  {name[:64]}{hit}")


def holdout_test(name_sub: str):
    catalog = _load_catalog()
    cache = _load_cache()
    target = next((p for p in catalog if name_sub.lower() in p["name"].lower()), None)
    if not target:
        print(f"Không tìm thấy sản phẩm chứa '{name_sub}'"); return
    held = next((u for u in target["urls"] if u in cache), None)
    if not held:
        print("Sản phẩm target chưa có ảnh nào trong cache. Chạy --build trước."); return
    print(f"TARGET: {target['name'][:70]}")
    print(f"Giữ lại (query) ảnh: {held}")
    qv = cache[held]
    scored = _rank(qv, exclude_urls={held})
    _print_ranking(scored, target_sub=name_sub)
    top_name = scored[0][1] if scored else ""
    ok = name_sub.lower() in top_name.lower()
    print(f"\nKẾT QUẢ: {'✓ ĐÚNG - target xếp #1' if ok else '✗ target KHÔNG ở #1'}")


def query_image(vec: np.ndarray, label: str, exclude_urls: set[str] | None = None):
    print(f"Query: {label} | model: {IE.model_id()}")
    scored = _rank(vec, exclude_urls=exclude_urls)
    _print_ranking(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit-per-product", type=int, default=None)
    ap.add_argument("--holdout", type=str)
    ap.add_argument("--query-file", type=str)
    ap.add_argument("--query-url", type=str)
    args = ap.parse_args()

    if args.build:
        build_index(args.limit_per_product)
    if args.holdout:
        holdout_test(args.holdout)
    if args.query_file:
        with open(args.query_file, "rb") as f:
            vec = IE.embed_image(f.read())
        query_image(vec, args.query_file)
    if args.query_url:
        vec = IE.embed_url(args.query_url)
        if vec is None:
            print("Không tải được ảnh query."); return
        # Nếu URL query trùng 1 ảnh trong catalog -> loại ra để rank công bằng
        query_image(vec, args.query_url, exclude_urls={args.query_url})


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
