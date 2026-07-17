"""
test_finetune_api.py – Test model finetune (Vintern qua ngrok) so với DB Postgres.

Lấy N sản phẩm NGẪU NHIÊN trong bảng `products`, tải ảnh cover của từng sản phẩm,
gửi lên endpoint /predict của model finetune (Colab ngrok), rồi so sánh JSON model
trả về với dữ liệu THẬT trong DB theo từng trường.

Cách dùng:
    # URL ngrok truyền qua tham số hoặc env FINETUNE_API_URL
    python test_finetune_api.py --url https://xxxx.ngrok-free.app -n 5
    python test_finetune_api.py            # dùng $FINETUNE_API_URL, n=5

Trường so khớp: name (map keyword), category, material, colors, product_size,
compatible_elements. List so theo TẬP HỢP (không quan tâm thứ tự).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "langraph pipeline"))
import image_embedding as IE  # noqa: E402
from models import Product, get_session  # noqa: E402

EVAL_FIELDS = ["category", "material", "colors", "product_size", "compatible_elements"]


def cover_url(image) -> str | None:
    """Rút URL ảnh cover từ cột `image` (JSONB: list URL hoặc dict cover/images)."""
    if isinstance(image, dict):
        if image.get("cover"):
            return image["cover"]
        for im in image.get("images", []) or []:
            if isinstance(im, dict) and im.get("url"):
                return im["url"]
            if isinstance(im, str):
                return im
    elif isinstance(image, list):
        for u in image:
            if isinstance(u, str):
                return u
    return None


def norm(v) -> set:
    """Chuẩn hoá 1 giá trị field về set các token in-thường để so khớp."""
    if v is None:
        return set()
    if isinstance(v, (list, tuple, set)):
        return {str(x).strip().lower() for x in v if str(x).strip()}
    return {str(v).strip().lower()}


def parse_lenient(frag: str) -> dict:
    """json.loads chịu lỗi: model hay xuất newline THẬT trong chuỗi (vd
    product_description) → JSON không hợp lệ. Escape ký tự điều khiển rồi parse."""
    try:
        return json.loads(frag)
    except Exception:
        frag2 = re.sub(r"[\x00-\x1f]", lambda m: "\\u%04x" % ord(m.group()), frag)
        try:
            return json.loads(frag2)
        except Exception:
            return {}


def predict(url_base: str, img_bytes: bytes) -> dict:
    r = requests.post(
        f"{url_base}/predict",
        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
        headers={"ngrok-skip-browser-warning": "true"},
        timeout=120,
    )
    r.raise_for_status()
    res = (r.json() or {}).get("result", {}) or {}
    # Server bọc output thô vào {"raw": ...} khi nó tự parse thất bại → parse lại ở đây.
    if isinstance(res, dict) and set(res.keys()) <= {"raw"}:
        raw = res.get("raw", "")
        s, e = raw.find("{"), raw.rfind("}")
        res = parse_lenient(raw[s:e + 1]) if s != -1 and e != -1 else {}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.getenv("FINETUNE_API_URL", ""),
                    help="URL ngrok (mặc định lấy $FINETUNE_API_URL)")
    ap.add_argument("-n", type=int, default=5, help="số sản phẩm ngẫu nhiên")
    ap.add_argument("--seed", type=int, default=None, help="seed để lặp lại được")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    if not base:
        print("Thiếu --url hoặc env FINETUNE_API_URL"); return

    if args.seed is not None:
        random.seed(args.seed)

    sess = get_session()
    products = sess.query(Product).all()
    # chỉ chọn sản phẩm CÓ ảnh
    with_img = [p for p in products if cover_url(p.image)]
    sample = random.sample(with_img, min(args.n, len(with_img)))
    print(f"DB có {len(products)} sản phẩm ({len(with_img)} có ảnh). "
          f"Test {len(sample)} sản phẩm ngẫu nhiên qua {base}\n")

    field_hits = {f: 0 for f in EVAL_FIELDS}
    name_hits = 0
    tested = 0

    for p in sample:
        url = cover_url(p.image)
        b = IE.download_bytes(url)
        if not b:
            print(f"[BỎ QUA] pid={p.product_id} tải ảnh lỗi: {url}\n")
            continue

        try:
            pred = predict(base, b)
        except Exception as ex:
            print(f"[BỎ QUA] pid={p.product_id} gọi API lỗi: {ex}\n")
            continue

        tested += 1
        gold = {
            "name": p.name,
            "category": p.category,
            "material": list(p.material or []),
            "colors": list(p.colors or []),
            "product_size": list(p.product_size or []),
            "compatible_elements": list(p.compatible_elements or []),
        }

        print("=" * 70)
        print(f"pid={p.product_id}  {p.name[:60]}")
        print(f"  ảnh: {url}")
        # name: coi là khớp nếu model đọc được tên và trùng (bỏ khoảng trắng, in thường)
        pred_name = str(pred.get("name") or "").strip()
        name_ok = pred_name.lower() == str(gold["name"]).strip().lower()
        name_hits += int(name_ok)
        print(f"  name  DB   : {gold['name'][:60]}")
        print(f"  name  MODEL: {pred_name[:60]}  {'✓' if name_ok else '✗'}")

        for f in EVAL_FIELDS:
            g, pr = norm(gold[f]), norm(pred.get(f))
            ok = g == pr
            field_hits[f] += int(ok)
            print(f"  {f:20s} DB={sorted(g)}")
            print(f"  {'':20s} MD={sorted(pr)}  {'✓' if ok else '✗'}")
        print()

    if tested == 0:
        print("Không test được sản phẩm nào."); return

    print("=" * 70)
    print(f"KẾT QUẢ trên {tested} sản phẩm:")
    print(f"  {'name (khớp chính xác)':24s}: {name_hits}/{tested} = {name_hits/tested*100:5.1f}%")
    for f in EVAL_FIELDS:
        print(f"  {f:24s}: {field_hits[f]}/{tested} = {field_hits[f]/tested*100:5.1f}%")
    avg = sum(field_hits.values()) / (len(EVAL_FIELDS) * tested) * 100
    print(f"  {'TRUNG BÌNH (5 field)':24s}: {avg:5.1f}%")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
