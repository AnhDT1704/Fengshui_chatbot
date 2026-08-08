"""
admin_import.py – Chủ shop nạp file Excel để cập nhật DỮ LIỆU ĐỘNG.

Các loại dữ liệu (cố ý KHÔNG train vào model finetune vì chúng đổi liên tục):
  1. price_range – giá bán (bảng products)
  2. quantity_max – số lượng còn lại (bảng products)
  3. promotions – chương trình khuyến mãi (bảng promotions)
  4. catalog – sản phẩm MỚI / cập nhật full row (Postgres + OpenSearch text + vector ảnh SigLIP)

Vì sao có module này: trước đây muốn sửa giá/tồn/khuyến mãi phải vào pgAdmin gõ SQL.
Giờ chủ shop tự up file Excel từ giao diện admin.

QUAN TRỌNG — đồng bộ OpenSearch:
  `price_range` và `in_stock` CÓ nằm trong index OpenSearch và `filter_search` lọc theo
  chúng. Sửa Postgres mà quên OpenSearch → bộ lọc chạy trên dữ liệu CŨ. Nên mỗi lần cập
  nhật giá, ta update luôn doc tương ứng trong index. (`quantity_max` không được index
  nên không cần đồng bộ.)

  Catalog SP mới: ghi Postgres + index text (embedding) + index vector ảnh (SigLIP) từ
  URL trong cột `image` (JSON: cover + images[].url).

Triết lý xử lý lỗi: KHÔNG chấp nhận "nạp một nửa". Validate TOÀN BỘ file trước; chỉ khi
sạch lỗi mới ghi DB. File có 1 dòng sai → từ chối cả file, báo rõ sai ở dòng nào. Nạp
nửa vời khiến giá sản phẩm này mới, sản phẩm kia cũ — rất khó lần ra.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text

import db_service
import opensearch_service
from logger import get_logger
from models import get_engine

log = get_logger("admin_import")

# Cột bắt buộc của từng loại file.
PRODUCT_COLS = ["product_id"] # + ít nhất 1 trong 2 cột dữ liệu
PRODUCT_DATA_COLS = ["price_range", "quantity_max"]
PROMO_COLS = ["promo_date", "discount_percent", "scope", "promotion_info"]

# Catalog SP đầy đủ — cột khớp schema bảng products.
CATALOG_REQUIRED = ["product_id", "name", "category"]
CATALOG_OPTIONAL = [
    "material", "colors", "compatible_elements", "product_size",
    "price_range", "brand", "origin", "warranty",
    "quantity_max", "in_stock", "product_description", "image",
]


class ImportError_(Exception):
    """Lỗi nghiệp vụ khi nạp file — thông điệp hiển thị thẳng cho chủ shop."""


def _read_excel(raw: bytes, kind: str = "") -> pd.DataFrame:
    log.info("NẠP FILE EXCEL %s %.1f KB", kind, len(raw) / 1024)
    try:
        df = pd.read_excel(io.BytesIO(raw), dtype=object)
    except Exception as e:
        log.error("Không đọc được file: %s", e)
        raise ImportError_(f"Không đọc được file Excel: {e}")
    if df.empty:
        log.error("File rỗng")
        raise ImportError_("File rỗng, không có dòng dữ liệu nào.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    log.info("Đọc được %d dòng | cột: %s", len(df), list(df.columns))
    return df


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def import_products(raw: bytes) -> dict:
    """Nạp file giá / tồn kho.

    Cột: product_id (bắt buộc) + price_range và/hoặc quantity_max.
    Ô để TRỐNG = giữ nguyên giá trị cũ (cho phép chỉ cập nhật giá, không đụng tồn kho).
    """
    df = _read_excel(raw, "GIÁ / TỒN KHO")

    if "product_id" not in df.columns:
        log.error("Thiếu cột 'product_id'")
        raise ImportError_("Thiếu cột bắt buộc 'product_id'.")
    data_cols = [c for c in PRODUCT_DATA_COLS if c in df.columns]
    if not data_cols:
        log.error("Không có cột dữ liệu nào (cần price_range và/hoặc quantity_max)")
        raise ImportError_("File phải có ít nhất một trong hai cột: "
                           "'price_range' (giá) hoặc 'quantity_max' (số lượng còn lại).")
    log.info("Sẽ cập nhật cột: %s", data_cols)

    errors: list[str] = []
    rows: list[dict] = []
    seen: set[int] = set()

    for i, r in df.iterrows():
        ln = i + 2 # +2: dòng 1 là tiêu đề, index pandas bắt đầu từ 0
        pid_raw = r.get("product_id")
        if _blank(pid_raw):
            errors.append(f"Dòng {ln}: thiếu product_id.")
            continue
        try:
            pid = int(float(str(pid_raw).strip()))
        except Exception:
            errors.append(f"Dòng {ln}: product_id '{pid_raw}' không phải số.")
            continue
        if pid in seen:
            errors.append(f"Dòng {ln}: product_id {pid} bị lặp trong file.")
            continue
        old = db_service.get_product_by_id(pid)
        if old is None:
            errors.append(f"Dòng {ln}: không có sản phẩm product_id={pid} trong hệ thống.")
            continue
        seen.add(pid)

        # Giữ giá trị CŨ để log "cũ → mới" — nhìn log là biết ngay thật sự đổi gì.
        row: dict = {"product_id": pid, "_old_price": old.price_range,
                     "_old_qty": old.quantity_max, "_name": old.name}

        if "price_range" in data_cols and not _blank(r.get("price_range")):
            row["price_range"] = str(r["price_range"]).strip()

        if "quantity_max" in data_cols and not _blank(r.get("quantity_max")):
            try:
                qty = int(float(str(r["quantity_max"]).strip()))
            except Exception:
                errors.append(f"Dòng {ln}: quantity_max '{r['quantity_max']}' không phải số.")
                continue
            if qty < 0:
                errors.append(f"Dòng {ln}: quantity_max = {qty}, không được âm.")
                continue
            row["quantity_max"] = qty

        # Chỉ có product_id + các khoá _old_* → không có gì để cập nhật.
        if not any(k in row for k in PRODUCT_DATA_COLS):
            continue
        rows.append(row)

    if errors:
        log.error("VALIDATE THẤT BẠI — %d lỗi, KHÔNG ghi gì vào DB:", len(errors))
        for e in errors[:20]:
            log.error("%s", e)
        log.error("")
        raise ImportError_("File có lỗi, KHÔNG cập nhật gì cả:\n" + "\n".join(errors[:20]))
    if not rows:
        log.error("Không có dòng nào chứa dữ liệu để cập nhật")
        raise ImportError_("Không có dòng nào chứa dữ liệu để cập nhật.")

    log.info("Validate OK — %d dòng hợp lệ, bắt đầu ghi DB", len(rows))

    changed: list[str] = [] # chỉ log dòng THỰC SỰ đổi giá trị
    engine = get_engine()
    with engine.begin() as conn:
        for row in rows:
            sets, params = [], {"pid": row["product_id"]}
            diffs = []
            if "price_range" in row:
                sets.append("price_range = :price")
                params["price"] = row["price_range"]
                if str(row["_old_price"]) != str(row["price_range"]):
                    diffs.append(f"giá: {row['_old_price']} → {row['price_range']}")
            if "quantity_max" in row:
                sets.append("quantity_max = :qty")
                params["qty"] = row["quantity_max"]
                # Hết hàng khi số lượng còn lại về 0 (khớp logic _live_fields ở memory.py).
                sets.append("in_stock = :instock")
                params["instock"] = row["quantity_max"] > 0
                if str(row["_old_qty"]) != str(row["quantity_max"]):
                    diffs.append(f"tồn: {row['_old_qty']} → {row['quantity_max']}"
                                 + (" [HẾT HÀNG]" if row["quantity_max"] == 0 else ""))
            conn.execute(text(f"UPDATE products SET {', '.join(sets)} WHERE product_id = :pid"),
                         params)
            if diffs:
                changed.append(f"pid={row['product_id']:<4} {row['_name'][:34]:<34} | "
                               + " | ".join(diffs))

    # In từng thay đổi THẬT (bỏ qua dòng giữ nguyên) → biết chính xác cái gì đã đổi.
    if changed:
        log.info("ĐÃ ĐỔI %d/%d sản phẩm:", len(changed), len(rows))
        for c in changed[:30]:
            log.info("%s", c)
        if len(changed) > 30:
            log.info("... và %d sản phẩm nữa", len(changed) - 30)
    else:
        log.info("(không sản phẩm nào đổi giá trị — file giống hệt dữ liệu hiện tại)")

    # BẮT BUỘC: filter_search lọc theo price_range/in_stock trên INDEX, không phải trên
    # Postgres. Quên bước này thì bot vẫn lọc theo giá CŨ dù DB đã đổi.
    os_ok = os_miss = 0
    for row in rows:
        fields = {}
        if "price_range" in row:
            fields["price_range"] = row["price_range"]
        if "quantity_max" in row:
            fields["in_stock"] = row["quantity_max"] > 0
        if not fields:
            continue
        if opensearch_service.update_product_fields(row["product_id"], fields):
            os_ok += 1
        else:
            os_miss += 1
            log.warning("pid=%s chưa có trong index OpenSearch", row["product_id"])

    log.info("Đồng bộ OpenSearch: %d thành công, %d chưa index", os_ok, os_miss)
    log.info("HOÀN TẤT: %d sản phẩm cập nhật, %d thực sự đổi giá trị",
             len(rows), len(changed))

    return {
        "updated": len(rows),
        "changed": len(changed),
        "fields": data_cols,
        "opensearch_sync": os_ok,
        "opensearch_miss": os_miss,
    }


def import_promotions(raw: bytes, replace_all: bool = True) -> dict:
    """Nạp bảng chương trình khuyến mãi.

    Cột: promo_date ("d/m" — vd 7/7, 30/4), discount_percent, scope, promotion_info.
    day/month được SUY RA từ promo_date, admin không phải nhập tay (tránh lệch nhau).

    replace_all=True (mặc định): file là NGUỒN SỰ THẬT — xoá sạch bảng cũ rồi nạp lại.
      Chọn vậy vì khuyến mãi cần XOÁ được: nếu chỉ upsert, chương trình admin đã bỏ khỏi
      file vẫn nằm lại trong DB và bot vẫn quảng cáo cho khách.
    """
    df = _read_excel(raw, "KHUYẾN MÃI")

    missing = [c for c in PROMO_COLS if c not in df.columns]
    if missing:
        log.error("Thiếu cột: %s", missing)
        raise ImportError_(f"Thiếu cột bắt buộc: {', '.join(missing)}. "
                           f"Cần đủ: {', '.join(PROMO_COLS)}.")

    errors: list[str] = []
    rows: list[dict] = []
    seen: set[str] = set()

    for i, r in df.iterrows():
        ln = i + 2
        if all(_blank(r.get(c)) for c in PROMO_COLS):
            continue # dòng trống hoàn toàn → bỏ qua

        pdate = str(r.get("promo_date") or "").strip()
        if not pdate:
            errors.append(f"Dòng {ln}: thiếu promo_date.")
            continue
        # Excel hay tự biến "7/7" thành ngày tháng → chấp nhận cả dạng datetime.
        if hasattr(r["promo_date"], "day") and hasattr(r["promo_date"], "month"):
            day, month = r["promo_date"].day, r["promo_date"].month
            pdate = f"{day}/{month}"
        else:
            parts = pdate.replace("-", "/").split("/")
            if len(parts) < 2:
                errors.append(f"Dòng {ln}: promo_date '{pdate}' sai định dạng, cần dạng d/m (vd 7/7, 30/4).")
                continue
            try:
                day, month = int(parts[0]), int(parts[1])
            except Exception:
                errors.append(f"Dòng {ln}: promo_date '{pdate}' sai định dạng, cần dạng d/m.")
                continue
            pdate = f"{day}/{month}"

        if not (1 <= month <= 12) or not (1 <= day <= 31):
            errors.append(f"Dòng {ln}: ngày '{pdate}' không hợp lệ.")
            continue
        if pdate in seen:
            errors.append(f"Dòng {ln}: ngày {pdate} bị lặp trong file.")
            continue
        seen.add(pdate)

        try:
            disc = int(float(str(r["discount_percent"]).strip()))
        except Exception:
            errors.append(f"Dòng {ln}: discount_percent '{r['discount_percent']}' không phải số.")
            continue
        if not (0 < disc < 100):
            errors.append(f"Dòng {ln}: discount_percent = {disc}, phải trong khoảng 1-99.")
            continue

        info = str(r.get("promotion_info") or "").strip()
        if not info:
            errors.append(f"Dòng {ln}: thiếu promotion_info (câu bot đọc cho khách).")
            continue

        # CHẶN MÂU THUẪN SỐ ↔ CHỮ.
        # promotion_info là câu bot ĐỌC THẲNG cho khách; discount_percent là cột số dùng
        # để lọc/tính. Nếu admin chỉ sửa một trong hai (đã xảy ra thật: cột số = 15 nhưng
        # câu chữ ghi "giảm 22%"), bot sẽ NÓI SAI với khách. Thà từ chối file còn hơn để
        # dữ liệu tự mâu thuẫn.
        m = re.search(r"(\d+)\s*%", info)
        if m and int(m.group(1)) != disc:
            errors.append(
                f"Dòng {ln}: LỆCH SỐ — cột discount_percent = {disc}%, nhưng "
                f"promotion_info lại ghi \"{m.group(1)}%\". Sửa cho khớp nhau "
                f"(bot đọc câu chữ này cho khách nên phải đúng)."
            )
            continue

        rows.append({
            "promo_date": pdate,
            "day": day,
            "month": month,
            "discount_percent": disc,
            "scope": str(r.get("scope") or "mọi sản phẩm").strip(),
            "promotion_info": info,
        })

    if errors:
        log.error("VALIDATE THẤT BẠI — %d lỗi, KHÔNG ghi gì vào DB:", len(errors))
        for e in errors[:20]:
            log.error("%s", e)
        log.error("")
        raise ImportError_("File có lỗi, KHÔNG cập nhật gì cả:\n" + "\n".join(errors[:20]))
    if not rows:
        log.error("Không có dòng khuyến mãi nào trong file")
        raise ImportError_("Không có dòng khuyến mãi nào trong file.")

    log.info("Validate OK — %d chương trình hợp lệ", len(rows))

    engine = get_engine()
    with engine.begin() as conn:
        # Đối chiếu bảng CŨ để log rõ: MỚI / SỬA / XOÁ. Lấy ĐỦ 3 cột dữ liệu — trước đây
        # chỉ so discount_percent nên admin sửa mỗi câu chữ thì log báo "SỬA (0)", sai
        # sự thật. Log mà bảo "không đổi gì" trong khi DB đã đổi thì tệ hơn là không log.
        old = {r[0]: {"discount_percent": r[1], "scope": r[2], "promotion_info": r[3]}
               for r in conn.execute(text(
                   "SELECT promo_date, discount_percent, scope, promotion_info FROM promotions"
               )).fetchall()}

        if replace_all:
            conn.execute(text("TRUNCATE promotions RESTART IDENTITY"))
        for row in rows:
            conn.execute(text("""
                INSERT INTO promotions (promo_date, day, month, discount_percent, scope, promotion_info)
                VALUES (:promo_date, :day, :month, :discount_percent, :scope, :promotion_info)
                ON CONFLICT (promo_date) DO UPDATE SET
                    day = EXCLUDED.day,
                    month = EXCLUDED.month,
                    discount_percent = EXCLUDED.discount_percent,
                    scope = EXCLUDED.scope,
                    promotion_info = EXCLUDED.promotion_info
            """), row)

    DIFF_COLS = ["discount_percent", "scope", "promotion_info"]
    new = {r["promo_date"]: {c: r[c] for c in DIFF_COLS} for r in rows}

    added = [d for d in new if d not in old]
    removed = [d for d in old if d not in new] # bị xoá vì file là NGUỒN SỰ THẬT
    edited = [d for d in new if d in old
               and any(str(old[d][c]) != str(new[d][c]) for c in DIFF_COLS)]

    log.info("MỚI (%d): %s", len(added),
             ", ".join(f"{d}={new[d]['discount_percent']}%" for d in added) or "—")

    if edited:
        log.info("SỬA (%d):", len(edited))
        for d in edited:
            # Nêu ĐÍCH DANH cột nào đổi, cũ → mới. Không gộp chung một dòng mơ hồ.
            for c in DIFF_COLS:
                o, n = str(old[d][c]), str(new[d][c])
                if o != n:
                    log.info("%-6s %-16s : %s → %s", d, c, o[:40], n[:40])
    else:
        log.info("SỬA (0): —")

    if removed:
        log.warning("XOÁ (%d): %s", len(removed),
                    ", ".join(f"{d}={old[d]['discount_percent']}%" for d in removed))
        log.warning("Các chương trình trên KHÔNG có trong file → đã bị XOÁ khỏi DB. "
                    "Bot sẽ không còn quảng cáo chúng.")
    else:
        log.info("XOÁ (0): —")

    log.info("HOÀN TẤT: bảng khuyến mãi giờ có %d chương trình "
             "(%d mới, %d sửa, %d xoá)", len(rows), len(added), len(edited), len(removed))

    return {"updated": len(rows), "added": len(added), "edited": len(edited),
            "removed": len(removed), "replaced_all": replace_all}


def template_products() -> bytes:
    """File mẫu giá/tồn kho, ĐIỀN SẴN dữ liệu THẬT đang có → admin chỉ việc sửa số."""
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT product_id, name, price_range, quantity_max
            FROM products ORDER BY product_id
        """)).fetchall()
    df = pd.DataFrame(
        [{"product_id": r[0], "ten_san_pham (chi de tham khao)": r[1],
          "price_range": r[2], "quantity_max": r[3]} for r in rows]
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="gia_tonkho")
    return buf.getvalue()


def template_promotions() -> bytes:
    """File mẫu khuyến mãi, điền sẵn các chương trình đang có."""
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT promo_date, discount_percent, scope, promotion_info
            FROM promotions ORDER BY month, day
        """)).fetchall()
    df = pd.DataFrame(
        [{"promo_date": r[0], "discount_percent": r[1], "scope": r[2], "promotion_info": r[3]}
         for r in rows]
    )
    if df.empty:
        df = pd.DataFrame([{
            "promo_date": "7/7", "discount_percent": 20, "scope": "mọi sản phẩm",
            "promotion_info": "Sale 7/7: giảm 20% mọi sản phẩm",
        }])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="khuyen_mai")
    return buf.getvalue()


# ---------------------------------------------------------------------------
#  CATALOG — nạp / cập nhật sản phẩm đầy đủ + vector ảnh SigLIP
# ---------------------------------------------------------------------------

def _parse_list_cell(v: Any) -> list[str]:
    """Parse ô Excel thành list[str]: JSON array, hoặc 'a; b; c' / 'a|b'."""
    if _blank(v):
        return []
    s = str(v).strip()
    if s.startswith("["):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except Exception:
            pass
    parts = re.split(r"[;|]", s)
    return [p.strip() for p in parts if p.strip()]


def _parse_bool_cell(v: Any, default: bool = True) -> bool:
    if _blank(v):
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "có", "co", "đúng", "dung"):
        return True
    if s in ("0", "false", "no", "n", "không", "khong"):
        return False
    return default


def parse_image_json(v: Any) -> Optional[dict]:
    """Parse cột image → dict {cover, images:[{url, color?}]} đúng format DB.

    Chấp nhận:
      - JSON object đầy đủ (khuyến nghị)
      - URL cover đơn (tự bọc thành {cover, images:[]})
    """
    if _blank(v):
        return None
    s = str(v).strip()
    if s.startswith("{"):
        try:
            data = json.loads(s)
        except Exception as e:
            raise ValueError(f"image JSON không hợp lệ: {e}")
        if not isinstance(data, dict):
            raise ValueError("image phải là object JSON {cover, images}")
        cover = data.get("cover")
        images = data.get("images") or []
        if cover is not None and not isinstance(cover, str):
            raise ValueError("image.cover phải là URL (string)")
        if not isinstance(images, list):
            raise ValueError("image.images phải là mảng")
        norm_images = []
        for im in images:
            if isinstance(im, str):
                norm_images.append({"url": im, "color": None})
            elif isinstance(im, dict) and im.get("url"):
                norm_images.append({
                    "url": str(im["url"]).strip(),
                    "color": im.get("color"),
                })
            else:
                raise ValueError("mỗi phần tử images cần 'url'")
        return {"cover": (str(cover).strip() if cover else None), "images": norm_images}
    # URL đơn
    if s.startswith("http://") or s.startswith("https://"):
        return {"cover": s, "images": []}
    raise ValueError("image phải là JSON {cover, images} hoặc URL https://...")


def extract_image_urls(image: Optional[dict]) -> list[tuple[str, bool]]:
    """[(url, is_cover)] từ object image DB."""
    out: list[tuple[str, bool]] = []
    if not isinstance(image, dict):
        return out
    cover = image.get("cover")
    if cover:
        out.append((str(cover), True))
    for im in image.get("images") or []:
        u = im.get("url") if isinstance(im, dict) else im
        if u:
            out.append((str(u), False))
    seen, uniq = set(), []
    for u, c in out:
        if u not in seen:
            seen.add(u)
            uniq.append((u, c))
    return uniq


def reindex_product_image_vectors(product_id: int, image: Optional[dict]) -> dict:
    """Xoá vector ảnh cũ của product_id, download URL → SigLIP → index lại."""
    import image_embedding as IE

    urls = extract_image_urls(image)
    client = opensearch_service.get_client()
    deleted = 0
    try:
        r = client.delete_by_query(
            index=opensearch_service.config.OS_IMAGE_INDEX,
            body={"query": {"term": {"product_id": int(product_id)}}},
            refresh=True,
        )
        deleted = int(r.get("deleted") or 0)
    except Exception as e:
        log.warning("Xoá image vectors cũ pid=%s lỗi: %s", product_id, e)

    docs = []
    fail_urls = []
    for u, is_cover in urls:
        b = IE.download_bytes(u)
        if not b:
            fail_urls.append(u)
            continue
        try:
            vec = IE.embed_image(b).tolist()
        except Exception as e:
            log.warning("embed ảnh lỗi pid=%s url=%s: %s", product_id, u[:80], e)
            fail_urls.append(u)
            continue
        docs.append({
            "product_id": int(product_id),
            "image_url": u,
            "is_cover": bool(is_cover),
            "embedding": vec,
        })
    indexed = 0
    if docs:
        opensearch_service.bulk_index_image_vectors(docs)
        indexed = len(docs)
    return {
        "urls_total": len(urls),
        "indexed": indexed,
        "deleted_old": deleted,
        "failed": len(fail_urls),
        "failed_urls": fail_urls[:5],
    }


def _index_product_text(row: dict) -> bool:
    """Embed product_description (hoặc name) và upsert doc text OpenSearch."""
    try:
        import embedding_service
        text_src = (row.get("product_description") or "").strip() or row["name"]
        emb = embedding_service.embed_single(text_src)
        opensearch_service.index_product(
            product_id=row["product_id"],
            product_description=text_src,
            embedding=emb,
            metadata={
                "name": row["name"],
                "category": row["category"],
                "material": row.get("material") or [],
                "compatible_elements": row.get("compatible_elements") or [],
                "colors": row.get("colors") or [],
                "product_size": row.get("product_size") or [],
                "brand": row.get("brand") or "Vạn An Group",
                "in_stock": row.get("in_stock", True),
                "price_range": row.get("price_range"),
            },
        )
        return True
    except Exception as e:
        log.warning("Index text OS pid=%s lỗi: %s", row.get("product_id"), e)
        return False


def import_catalog(raw: bytes, reindex_images: bool = True) -> dict:
    """Nạp / cập nhật sản phẩm đầy đủ từ Excel (catalog).

    Cột bắt buộc: product_id, name, category.
    Cột image: JSON đúng format DB
      {"cover": "https://...", "images": [{"url": "...", "color": "..."}, ...]}
    Sau khi ghi Postgres: index text OpenSearch + (tuỳ chọn) SigLIP vector ảnh.

    Gọi progress.emit() ở mỗi pha khi API stream (nếu không stream → no-op).
    """
    try:
        import progress as _progress
    except Exception:  # pragma: no cover
        class _progress:  # type: ignore
            @staticmethod
            def emit(stage: str, message: str, **extra):
                pass

    _progress.emit("validate", "Đang đọc & kiểm tra file catalog…")
    df = _read_excel(raw, "CATALOG SP")
    missing = [c for c in CATALOG_REQUIRED if c not in df.columns]
    if missing:
        raise ImportError_(
            f"Thiếu cột bắt buộc: {', '.join(missing)}. "
            f"Cần: {', '.join(CATALOG_REQUIRED)}."
        )

    errors: list[str] = []
    rows: list[dict] = []
    seen: set[int] = set()

    for i, r in df.iterrows():
        ln = i + 2
        if _blank(r.get("product_id")):
            errors.append(f"Dòng {ln}: thiếu product_id.")
            continue
        try:
            pid = int(float(str(r["product_id"]).strip()))
        except Exception:
            errors.append(f"Dòng {ln}: product_id '{r.get('product_id')}' không phải số.")
            continue
        if pid in seen:
            errors.append(f"Dòng {ln}: product_id {pid} bị lặp trong file.")
            continue
        seen.add(pid)

        name = str(r.get("name") or "").strip()
        category = str(r.get("category") or "").strip()
        if not name:
            errors.append(f"Dòng {ln}: thiếu name.")
            continue
        if not category:
            errors.append(f"Dòng {ln}: thiếu category.")
            continue

        row: dict = {
            "product_id": pid,
            "name": name,
            "category": category,
            "material": _parse_list_cell(r.get("material")) if "material" in df.columns else [],
            "colors": _parse_list_cell(r.get("colors")) if "colors" in df.columns else [],
            "compatible_elements": (
                _parse_list_cell(r.get("compatible_elements"))
                if "compatible_elements" in df.columns else []
            ),
            "product_size": (
                _parse_list_cell(r.get("product_size"))
                if "product_size" in df.columns else []
            ),
            "price_range": (
                None if "price_range" not in df.columns or _blank(r.get("price_range"))
                else str(r.get("price_range")).strip()
            ),
            "brand": (
                "Vạn An Group" if "brand" not in df.columns or _blank(r.get("brand"))
                else str(r.get("brand")).strip()
            ),
            "origin": (
                "Việt Nam" if "origin" not in df.columns or _blank(r.get("origin"))
                else str(r.get("origin")).strip()
            ),
            "warranty": (
                None if "warranty" not in df.columns or _blank(r.get("warranty"))
                else str(r.get("warranty")).strip()
            ),
            "product_description": (
                None if "product_description" not in df.columns
                or _blank(r.get("product_description"))
                else str(r.get("product_description")).strip()
            ),
            "in_stock": (
                _parse_bool_cell(r.get("in_stock"), True)
                if "in_stock" in df.columns else True
            ),
            "is_new": db_service.get_product_by_id(pid) is None,
        }

        if "quantity_max" in df.columns and not _blank(r.get("quantity_max")):
            try:
                qty = int(float(str(r["quantity_max"]).strip()))
                row["quantity_max"] = qty
                row["in_stock"] = qty > 0
            except Exception:
                errors.append(f"Dòng {ln}: quantity_max không hợp lệ.")
                continue

        if "image" in df.columns and not _blank(r.get("image")):
            try:
                row["image"] = parse_image_json(r.get("image"))
            except ValueError as e:
                errors.append(f"Dòng {ln}: {e}")
                continue
        else:
            row["image"] = None

        rows.append(row)

    if errors:
        preview = "\n".join(errors[:15])
        more = f"\n... và {len(errors) - 15} lỗi nữa." if len(errors) > 15 else ""
        raise ImportError_(
            f"File catalog có {len(errors)} lỗi — KHÔNG ghi gì vào DB:\n{preview}{more}"
        )
    if not rows:
        raise ImportError_("Không có dòng catalog hợp lệ để nạp.")

    log.info("Validate OK — %d dòng catalog, bắt đầu ghi DB", len(rows))
    _progress.emit(
        "postgres",
        f"File hợp lệ — {len(rows)} sản phẩm. Đang ghi PostgreSQL…",
        total=len(rows),
    )

    created = updated = 0
    for i, row in enumerate(rows, 1):
        meta = {
            "product_id": row["product_id"],
            "name": row["name"],
            "category": row["category"],
            "material": row["material"],
            "compatible_elements": row["compatible_elements"],
            "colors": row["colors"],
            "product_size": row["product_size"],
            "price_range": row["price_range"],
            "brand": row["brand"],
            "origin": row["origin"],
            "warranty": row["warranty"],
            "in_stock": row["in_stock"],
        }
        if "quantity_max" in row:
            meta["quantity_max"] = row["quantity_max"]
        if row.get("image") is not None:
            meta["image"] = row["image"]
        desc = row.get("product_description") or row["name"]
        db_service.upsert_product(meta, desc)
        if row["is_new"]:
            created += 1
        else:
            updated += 1
        log.info("Catalog %s pid=%s %s",
                 "MỚI" if row["is_new"] else "CẬP NHẬT",
                 row["product_id"], row["name"][:50])
        if i == 1 or i == len(rows) or i % 3 == 0:
            _progress.emit(
                "postgres",
                f"Đang ghi PostgreSQL… {i}/{len(rows)} "
                f"({row['name'][:40]})",
                current=i, total=len(rows),
            )

    # OpenSearch text
    _progress.emit(
        "opensearch_text",
        f"Đang tạo chunk / index OpenSearch (text + embedding)… 0/{len(rows)}",
        current=0, total=len(rows),
    )
    os_ok = os_fail = 0
    for i, row in enumerate(rows, 1):
        if _index_product_text(row):
            os_ok += 1
        else:
            os_fail += 1
        if i == 1 or i == len(rows) or i % 3 == 0:
            _progress.emit(
                "opensearch_text",
                f"Đang index OpenSearch (text)… {i}/{len(rows)}",
                current=i, total=len(rows),
            )

    # SigLIP image vectors
    img_stats = {"products": 0, "indexed": 0, "failed": 0}
    rows_with_img = [r for r in rows if r.get("image")] if reindex_images else []
    if rows_with_img:
        _progress.emit(
            "siglip",
            f"Đang tải ảnh & tạo vector SigLIP… 0/{len(rows_with_img)} sản phẩm",
            current=0, total=len(rows_with_img),
        )
        for i, row in enumerate(rows_with_img, 1):
            _progress.emit(
                "siglip",
                f"Đang tạo vector ảnh (SigLIP)… {i}/{len(rows_with_img)} — "
                f"{row['name'][:36]}",
                current=i, total=len(rows_with_img),
                product_id=row["product_id"],
            )
            st = reindex_product_image_vectors(row["product_id"], row["image"])
            img_stats["products"] += 1
            img_stats["indexed"] += st["indexed"]
            img_stats["failed"] += st["failed"]
            log.info(
                "SigLIP pid=%s: %d/%d ảnh indexed (fail=%d)",
                row["product_id"], st["indexed"], st["urls_total"], st["failed"],
            )
    elif reindex_images:
        _progress.emit("siglip", "Không có cột image / URL ảnh — bỏ qua vector SigLIP.")

    log.info(
        "HOÀN TẤT catalog: %d dòng (mới=%d, cập nhật=%d) | OS text ok=%d fail=%d | "
        "ảnh products=%d vectors=%d fail=%d",
        len(rows), created, updated, os_ok, os_fail,
        img_stats["products"], img_stats["indexed"], img_stats["failed"],
    )
    result = {
        "updated": len(rows),
        "created": created,
        "edited": updated,
        "opensearch_sync": os_ok,
        "opensearch_miss": os_fail,
        "image_products": img_stats["products"],
        "image_vectors_indexed": img_stats["indexed"],
        "image_vectors_failed": img_stats["failed"],
    }
    _progress.emit(
        "done",
        f"Hoàn tất: {len(rows)} SP (mới {created}, sửa {updated}). "
        f"Vector ảnh: {img_stats['indexed']}.",
        result=result,
    )
    return result


def template_catalog() -> bytes:
    """File mẫu CATALOG SP — 1 dòng ví dụ + header đúng format DB."""
    sample_image = {
        "cover": "https://cf.shopee.vn/file/vn-11134207-820l4-mee358rkhfcxc8",
        "images": [
            {
                "url": "https://cf.shopee.vn/file/vn-11134207-820l4-mee35abpvg1veb",
                "color": "xanh dương và trắng",
            },
            {
                "url": "https://cf.shopee.vn/file/vn-11134207-820l4-mee3n8di4efbc1",
                "color": "tím và trắng",
            },
        ],
    }
    # Gợi ý product_id mới (max+1) — admin có thể đổi
    next_pid = 118
    try:
        engine = get_engine()
        with engine.begin() as conn:
            mx = conn.execute(text("SELECT COALESCE(MAX(product_id), 0) FROM products")).scalar()
            next_pid = int(mx or 0) + 1
    except Exception:
        pass

    df = pd.DataFrame([{
        "product_id": next_pid,
        "name": "Ví dụ: Vòng tay đá mẫu mới Vạn An Group",
        "category": "vòng tay",
        "material": "mã não; thạch anh",
        "colors": "xanh dương; trắng",
        "compatible_elements": "Thủy; Mộc",
        "product_size": "6mm; 8mm; 10mm",
        "price_range": "150.000 - 220.000",
        "brand": "Vạn An Group",
        "origin": "Việt Nam",
        "warranty": "thay dây trọn đời",
        "quantity_max": 100,
        "in_stock": True,
        "product_description": (
            "Mô tả sản phẩm (dùng cho semantic search). "
            "Xoá dòng ví dụ và điền SP thật trước khi nạp."
        ),
        "image": json.dumps(sample_image, ensure_ascii=False),
    }])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="catalog")
    return buf.getvalue()
