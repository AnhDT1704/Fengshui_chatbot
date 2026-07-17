"""
020_fix_product_covers.py – Sửa ảnh sản phẩm bị TRÙNG/sai (cover + gallery) từ file
export Shopee 'Hình ảnh sản phẩm.csv'. Khớp DB↔CSV theo TÊN (đã chủ shop duyệt tay).

Dry-run (chỉ in):  docker exec -w /app fengshui_chatbot python /app/migrations/020_fix_product_covers.py
ÁP DỤNG (ghi DB):  docker exec -w /app fengshui_chatbot python /app/migrations/020_fix_product_covers.py apply
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from models import Product, get_session  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, "Hình ảnh sản phẩm.csv")

# db_product_id -> csv et_title_product_id (lấy cover + gallery từ dòng CSV đó)
CSV_MAP = {
    9: "41015245542", 11: "40315245856", 59: "24389210307", 33: "27977227820",
    17: "29777274841", 20: "29577160109", 41: "27727893242", 13: "26729659472",
    29: "28531129214", 54: "25641148224", 83: "22554088177", 32: "28025364902",
    73: "23079572703", 26: "29208623774", 111: "18583929363", 28: "28623183617",
    49: "26623198886", 105: "19492854360", 24: "29364045476", 116: "14298113267",
    61: "23954060571", 85: "22479561786", 64: "23754062041", 108: "19183916552",
    66: "23654082812", 19: "29663747459", 81: "22579581495",
    # ── chủ shop duyệt ──
    23: "23554080496", 100: "20583905600", 90: "22054059884", 63: "23779575289",
    68: "23579575470", 82: "22579580678", 104: "19983913094", 117: "12099165460",
    84: "22554077272", 110: "18783914169", 98: "20983903619",
}

# db_product_id -> list link ảnh trực tiếp (ảnh ĐẦU = cover, còn lại = gallery)
DIRECT_IMG = {
    101: [
        "https://cf.shopee.vn/file/vn-11134207-7r98o-lldn4225zrgofd",
        "https://cf.shopee.vn/file/vn-11134207-7r98o-lldn42262klk80",
        "https://cf.shopee.vn/file/vn-11134207-7r98o-lldn4226161484",
        "https://cf.shopee.vn/file/vn-11134207-7r98o-lldn421w773q0f",
        "https://cf.shopee.vn/file/vn-11134207-7r98o-lldn421wctdicc",
        "https://cf.shopee.vn/file/vn-11134207-7r98o-lldn421w8lo6d0",
    ],
}

# db_product_id -> tên mới (chỉ những cái chủ shop yêu cầu đổi)
RENAME = {
    100: "Vòng tay trầm hương hoa sen Vạn An Group mang đến may mắn, hạnh phúc, bình an, tài lộc",
    63: "Nhang sạch Vạn An Group không hóa chất, hương quế, khuynh diệp, hương bài, đã kiểm định, hộp 100 cây cao 30cm",
    68: "Nhang sạch Vạn An Group không hóa chất, hương trầm, hương quế, hương khuynh diệp, hương bài, hộp 100 cây cao 38cm",
    82: "Nhang sạch hương quế Vạn An Group hoàn toàn từ thảo mộc tự nhiên, đã kiểm định, không hóa chất, an toàn sức khỏe",
    104: "Nhang sạch hương bài Vạn An Group hoàn toàn từ thảo mộc tự nhiên, đã kiểm định, không hóa chất, an toàn sức khỏe",
    117: "Nhang sạch hương khuynh diệp Vạn An Group hoàn toàn từ thảo mộc tự nhiên, đã kiểm định, không hóa chất, an toàn sức khỏe",
    101: "Lư xông đốt trầm hương Vạn An Grou bằng gốm sứ, có miếng lót chống cháy, dùng xông nhà tẩy uế, mang đến bình an, tài lộc",
    84: "Thác khói Vạn An Group dùng xông đốt trầm hương bằng gốm sứ cao cấp",
    110: "Lá bồ đề Vạn An Group cầu bình an, may mắn, tài lộc, dùng để ốp điện thoại, ví tiền",
    98: "Vòng tay trầm hương Vạn An Group mang đến may mắn, hạnh phúc, bình an, tài lộc",
}


def _cid(v) -> str:
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return str(v).strip()


def main(apply: bool):
    df = pd.read_excel(CSV_PATH, engine="openpyxl")
    img_cols = [f"ps_item_image.{i}" for i in range(1, 9)]
    csv = {}
    for _, x in df.iterrows():
        cover = x.get("ps_item_cover_image")
        if pd.isna(cover):
            continue
        gal = [str(x.get(c)).strip() for c in img_cols if pd.notna(x.get(c))]
        csv[_cid(x.get("et_title_product_id"))] = (str(cover).strip(), gal)

    missing = sorted({cid for cid in CSV_MAP.values() if cid not in csv})
    if missing:
        print("⛔ DỪNG — các CSV id KHÔNG có trong file:", missing)
        return

    s = get_session()
    db = {p.product_id: p for p in s.query(Product).all()}

    def apply_one(pid, cover, gallery):
        p = db.get(pid)
        if not p:
            print(f"  ⚠️ DB không có product_id={pid}")
            return
        old = (p.image or {}).get("cover", "")
        if apply:
            p.image = {"cover": cover, "images": [{"url": u, "color": None} for u in gallery]}
            if pid in RENAME:
                p.name = RENAME[pid]
        tag = " | RENAME" if pid in RENAME else ""
        print(f"  id {pid:<4} cover: ...{old[-14:]:<14} -> ...{cover[-14:]} | gallery={len(gallery)}{tag}")

    for pid, cid in CSV_MAP.items():
        cover, gal = csv[cid]
        apply_one(pid, cover, gal)
    for pid, links in DIRECT_IMG.items():
        apply_one(pid, links[0], links[1:])

    if apply:
        s.commit()
    print(f"\n{'✅ ĐÃ GHI DB' if apply else '🟡 DRY-RUN (chưa ghi)'} | {len(CSV_MAP) + len(DIRECT_IMG)} sản phẩm | đổi tên: {len(RENAME)}")


if __name__ == "__main__":
    main("apply" in sys.argv)
