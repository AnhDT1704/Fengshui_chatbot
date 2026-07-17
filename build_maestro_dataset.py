"""
build_maestro_dataset.py
────────────────────────
Sinh dataset JSONL cho maestro (fine-tune Qwen2.5-VL / PaliGemma-2) TỪ bảng
`products` trong PostgreSQL + cột `image` (ảnh Shopee).

Layout đầu ra (đúng chuẩn maestro JSONLDataset):

    dataset/
      train/  annotations.jsonl  +  *.jpg
      valid/  annotations.jsonl  +  *.jpg
      test/   annotations.jsonl  +  *.jpg

Mỗi dòng JSONL:
    {"image": "<file>.jpg", "prefix": "<câu lệnh>", "suffix": "<JSON đáp án>"}

CHẠY (trên host, có venv + docker postgres đang chạy):
    python build_maestro_dataset.py

Sau đó nén thư mục `dataset/` và upload lên Colab, trỏ maestro config vào nó.

Ghi chú thiết kế:
  • Chia theo ẢNH (KHÔNG theo sản phẩm): ảnh 1..N-1 của MỖI sản phẩm → TRAIN, nên
    model học ĐỦ toàn bộ 97 sản phẩm (phủ hết catalog — demo nhận đúng mọi SP). Ảnh
    CUỐI mỗi sản phẩm → held-out, chia đôi vào VALID/TEST → đánh giá trên ẢNH MỚI của
    sản phẩm ĐÃ học. train/val/test không trùng ảnh → không rò rỉ; dùng chung 2 model.
  • Ảnh được RESIZE về cạnh dài tối đa MAX_SIDE. Ảnh to → OCR chữ Việt tốt hơn (cần GPU khoẻ).
  • Field trong `suffix` chỉ gồm thuộc tính ỔN ĐỊNH, đọc/suy được từ ảnh. KHÔNG
    gồm price_range / quantity_min / quantity_max (cột hay thay đổi → để RAG lấy
    real-time lúc chatbot chạy, KHÔNG nhồi vào trọng số model).
  • Cột `colors` được CHUẨN HOÁ (tách cụm gộp, bỏ "không xác định", gom biến thể)
    trước khi đưa vào đáp án — xem normalize_colors().
"""

import io
import json
import random
import re
import shutil
import time
from pathlib import Path

import requests
from PIL import Image

from models import Product, get_session

# ── Cấu hình ────────────────────────────────────────────────────────
OUT_DIR             = Path("dataset")
MAX_SIDE            = 1024         # cạnh dài nhất ảnh sau resize (px). To → OCR tốt hơn (cần A100/GPU khoẻ). Hạ 768/512 nếu VRAM chật.
IMAGES_PER_PRODUCT  = 5            # số ảnh tối đa lấy mỗi sản phẩm (N-1 ảnh train + 1 ảnh held-out)
HOLDOUT_VAL_FRAC    = 0.5          # trong số ảnh held-out (ảnh cuối mỗi SP), tỉ lệ đi VALID (còn lại TEST)
SEED                = 42
DOWNLOAD_DELAY      = 0.3          # nghỉ giữa các lần tải (tránh bị Shopee CDN chặn)

# Nhiều cách diễn đạt câu lệnh → augment nhẹ, giúp model bớt bám 1 khuôn.
PREFIXES = [
    "Trích xuất thông tin sản phẩm phong thủy trong ảnh dưới dạng JSON.",
    "Đọc ảnh và trả về thông tin sản phẩm ở định dạng JSON.",
    "Phân tích ảnh sản phẩm phong thủy, xuất JSON gồm tên, danh mục, chất liệu, màu, size và mệnh hợp.",
]


# ── Helpers ─────────────────────────────────────────────────────────
def image_urls(image) -> list[str]:
    """Lấy danh sách URL ảnh (cover + variant) từ cột image (dict hoặc list)."""
    urls: list[str] = []
    if isinstance(image, dict):
        if image.get("cover"):
            urls.append(image["cover"])
        for im in image.get("images", []) or []:
            if isinstance(im, dict) and im.get("url"):
                urls.append(im["url"])
            elif isinstance(im, str):
                urls.append(im)
    elif isinstance(image, list):
        urls += [u for u in image if isinstance(u, str)]
    # dedupe, giữ thứ tự (cover trước)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download(url: str, retries: int = 3) -> bytes | None:
    for i in range(retries):
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.content
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def save_resized(raw: bytes, path: Path) -> bool:
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return False
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    img.save(path, "JPEG", quality=90)
    return True


# Gom biến thể màu về token chuẩn (khớp từ vựng màu của KB agent). Mở rộng tùy ý.
_COLOR_ALIASES = {
    "nâu đậm": "nâu", "nâu nhạt": "nâu", "màu nâu": "nâu",
    "xanh aqua": "xanh dương", "xanh nước biển": "xanh dương", "xanh biển": "xanh dương",
    "xanh ngọc": "xanh ngọc bích",
    "nhiều màu": "đa sắc", "ngũ sắc": "đa sắc",
}
# Giá trị coi như "không có màu xác định" → bỏ khỏi đáp án.
_COLOR_DROP = {"không xác định", ""}


def normalize_colors(colors) -> list[str]:
    """Chuẩn hoá cột colors: tách cụm gộp ("trắng + xanh dương", "đỏ, vàng, trắng"),
    lowercase, gom biến thể (nâu đậm→nâu), bỏ 'không xác định', khử trùng lặp."""
    out: list[str] = []
    for raw in (colors or []):
        for tok in re.split(r"[,+/]", str(raw)):        # tách theo dấu , + /
            c = re.sub(r"\s+", " ", tok).strip().lower()
            c = _COLOR_ALIASES.get(c, c)
            if c in _COLOR_DROP or c in out:
                continue
            out.append(c)
    return out


# Giá trị "rác" chung → loại khỏi mọi mảng.
_UNKNOWN = {"không xác định", ""}
# Gom biến thể cách viết size.
_SIZE_ALIASES = {"freesize": "free_size"}


def drop_unknown(vals) -> list[str]:
    """Bỏ 'không xác định'/rỗng, trim, khử trùng lặp — dùng cho material, compatible_elements."""
    out: list[str] = []
    for v in (vals or []):
        s = re.sub(r"\s+", " ", str(v)).strip()
        if s.lower() in _UNKNOWN or s in out:
            continue
        out.append(s)
    return out


def normalize_sizes(sizes) -> list[str]:
    """Chuẩn hoá product_size: bỏ 'không xác định', gom 'freesize'→'free_size', khử trùng lặp."""
    out: list[str] = []
    for raw in (sizes or []):
        s = re.sub(r"\s+", " ", str(raw)).strip()
        if s.lower() in _UNKNOWN:
            continue
        s = _SIZE_ALIASES.get(s.lower(), s)
        if s not in out:
            out.append(s)
    return out


def target_json(p: Product) -> str:
    """JSON đáp án (suffix). 6 field ổn định + product_description (mô tả / ý nghĩa /
    công dụng). KHÔNG có giá/tồn kho (cột hay đổi → để DB)."""
    obj = {
        "name":                p.name,
        "category":            p.category,
        "material":            drop_unknown(p.material),
        "colors":              normalize_colors(p.colors),
        "product_size":        normalize_sizes(p.product_size),
        "compatible_elements": drop_unknown(p.compatible_elements),
        "product_description": (p.product_description or "").strip(),
    }
    return json.dumps(obj, ensure_ascii=False)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    random.seed(SEED)
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)                    # dọn dataset cũ để sinh lại sạch
    sess = get_session()
    products = sess.query(Product).order_by(Product.product_id).all()

    # Chia SẢN PHẨM để quyết ảnh held-out đi VALID hay TEST.
    # (TRAIN luôn có TẤT CẢ sản phẩm — khác hẳn chia-theo-sản-phẩm cũ.)
    order = list(products)
    random.shuffle(order)
    n_val = int(len(order) * HOLDOUT_VAL_FRAC)
    holdout_split = {p.product_id: ("valid" if i < n_val else "test")
                     for i, p in enumerate(order)}

    for split in ("train", "valid", "test"):
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    lines = {"train": [], "valid": [], "test": []}
    prod_in_train: set = set()

    for p in products:
        urls = image_urls(p.image)[:IMAGES_PER_PRODUCT]
        if not urls:
            continue
        # ≥2 ảnh: ảnh CUỐI làm held-out, còn lại train. Chỉ 1 ảnh: cho hết vào train.
        train_urls, holdout_url = (urls[:-1], urls[-1]) if len(urls) >= 2 else (urls, None)

        # --- ảnh TRAIN (model học sản phẩm này) ---
        for k, url in enumerate(train_urls):
            raw = download(url)
            if not raw:
                continue
            fname = f"{p.product_id}_{k}.jpg"
            if not save_resized(raw, OUT_DIR / "train" / fname):
                continue
            lines["train"].append({"image": fname, "prefix": random.choice(PREFIXES),
                                   "suffix": target_json(p)})
            prod_in_train.add(p.product_id)
            time.sleep(DOWNLOAD_DELAY)

        # --- ảnh HELD-OUT (val hoặc test) ---
        if holdout_url:
            split = holdout_split[p.product_id]
            raw = download(holdout_url)
            if raw and save_resized(raw, OUT_DIR / split / f"{p.product_id}_h.jpg"):
                lines[split].append({"image": f"{p.product_id}_h.jpg",
                                     "prefix": random.choice(PREFIXES),
                                     "suffix": target_json(p)})
                time.sleep(DOWNLOAD_DELAY)

    for split in ("train", "valid", "test"):
        (OUT_DIR / split / "annotations.jsonl").write_text(
            "\n".join(json.dumps(l, ensure_ascii=False) for l in lines[split]),
            encoding="utf-8")

    total = sum(len(v) for v in lines.values())
    print(f"  train: {len(lines['train']):4d} ảnh  |  {len(prod_in_train)} sản phẩm (PHỦ HẾT catalog)")
    print(f"  valid: {len(lines['valid']):4d} ảnh  (ảnh MỚI của SP đã học)")
    print(f"  test : {len(lines['test']):4d} ảnh  (ảnh MỚI của SP đã học)")
    print(f"\n✓ Xong: {len(products)} sản phẩm → {total} mẫu, lưu ở '{OUT_DIR}/'.")
    print(f"  Model sẽ học ĐỦ {len(prod_in_train)} sản phẩm → demo nhận đúng mọi sản phẩm.")


if __name__ == "__main__":
    main()
