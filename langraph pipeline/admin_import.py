"""
admin_import.py – Chủ shop nạp file Excel để cập nhật DỮ LIỆU ĐỘNG.

Ba loại dữ liệu động (cố ý KHÔNG train vào model finetune vì chúng đổi liên tục):
  1. price_range – giá bán (bảng products)
  2. quantity_max – số lượng còn lại (bảng products)
  3. promotions – chương trình khuyến mãi (bảng promotions)

Vì sao có module này: trước đây muốn sửa giá/tồn/khuyến mãi phải vào pgAdmin gõ SQL.
Giờ chủ shop tự up file Excel từ giao diện admin.

QUAN TRỌNG — đồng bộ OpenSearch:
  `price_range` và `in_stock` CÓ nằm trong index OpenSearch và `filter_search` lọc theo
  chúng. Sửa Postgres mà quên OpenSearch → bộ lọc chạy trên dữ liệu CŨ. Nên mỗi lần cập
  nhật giá, ta update luôn doc tương ứng trong index. (`quantity_max` không được index
  nên không cần đồng bộ.)

Triết lý xử lý lỗi: KHÔNG chấp nhận "nạp một nửa". Validate TOÀN BỘ file trước; chỉ khi
sạch lỗi mới ghi DB. File có 1 dòng sai → từ chối cả file, báo rõ sai ở dòng nào. Nạp
nửa vời khiến giá sản phẩm này mới, sản phẩm kia cũ — rất khó lần ra.
"""

from __future__ import annotations

import io
import re
from typing import Any

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
