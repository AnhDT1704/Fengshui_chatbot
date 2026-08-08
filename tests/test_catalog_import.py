"""
test_catalog_import.py – catalog SP mới + parse image JSON + SigLIP vector search.

Unit (không cần API):
  - parse_image_json đúng format DB
  - extract_image_urls
  - template_catalog sinh xlsx

Integration (cần API + admin):
  - template /admin/template/catalog
  - vector search: embed cover SP có sẵn → kNN ra đúng product_id
"""

from __future__ import annotations

import io
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Repo root (db_service, models) + langraph pipeline (admin_import, image_embedding)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "langraph pipeline"))

from admin_import import (  # noqa: E402
    extract_image_urls,
    parse_image_json,
    template_catalog,
)

from conftest import XLSX_MIME, requires_api  # noqa: E402


# ── Unit ──────────────────────────────────────────────────────────


def test_parse_image_json_full_format():
    raw = {
        "cover": "https://cf.shopee.vn/file/vn-11134207-820l4-mee358rkhfcxc8",
        "images": [
            {
                "url": "https://cf.shopee.vn/file/vn-11134207-820l4-mee35abpvg1veb",
                "color": "xanh dương và trắng",
            },
            {
                "url": "https://cf.shopee.vn/file/vn-11134207-820l4-mee35ahjzk7761",
                "color": None,
            },
        ],
    }
    parsed = parse_image_json(json.dumps(raw, ensure_ascii=False))
    assert parsed["cover"].startswith("https://")
    assert len(parsed["images"]) == 2
    assert parsed["images"][0]["color"] == "xanh dương và trắng"
    assert parsed["images"][1]["color"] is None


def test_parse_image_json_single_url():
    u = "https://cf.shopee.vn/file/abc"
    assert parse_image_json(u) == {"cover": u, "images": []}


def test_parse_image_json_invalid():
    with pytest.raises(ValueError):
        parse_image_json("not-a-url-or-json")


def test_extract_image_urls_dedupe_cover():
    img = {
        "cover": "https://a.com/c.jpg",
        "images": [
            {"url": "https://a.com/c.jpg", "color": None},
            {"url": "https://a.com/1.jpg", "color": "đen"},
        ],
    }
    urls = extract_image_urls(img)
    # cover + 1 unique gallery (cover trùng bị gộp)
    assert urls[0] == ("https://a.com/c.jpg", True)
    assert ("https://a.com/1.jpg", False) in urls
    assert len(urls) == 2


def test_template_catalog_xlsx():
    raw = template_catalog()
    assert raw[:2] == b"PK"
    df = pd.read_excel(io.BytesIO(raw), dtype=object)
    for col in ("product_id", "name", "category", "image"):
        assert col in df.columns
    img = parse_image_json(df.iloc[0]["image"])
    assert img and img.get("cover")


# ── Integration ───────────────────────────────────────────────────


@pytest.mark.integration
@requires_api
def test_catalog_template_downloadable(api, admin_token):
    r = api.get("/admin/template/catalog", admin_token)
    assert r.status_code == 200, r.text
    assert r.content[:2] == b"PK"
    assert len(r.content) > 500


@pytest.mark.integration
@requires_api
def test_vector_search_existing_product_cover():
    """Finetune đang tắt: embed cover SP #1 từ DB → kNN phải ra đúng product_id.

    Test này kiểm tra SigLIP + OpenSearch image index TRƯỚC khi nạp SP mới.
    """
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))

    import image_embedding as IE
    import opensearch_service as oss
    import db_service

    p = db_service.get_product_by_id(1)
    assert p is not None and p.image, "cần SP #1 có image trong DB"
    img = p.image if isinstance(p.image, dict) else json.loads(p.image)
    cover = img.get("cover")
    assert cover, "SP #1 thiếu cover"

    n = oss.get_image_doc_count()
    assert n > 0, "image index rỗng — chưa sync vector ảnh"

    b = IE.download_bytes(cover)
    assert b, f"không tải được cover: {cover}"
    vec = IE.embed_image(b)
    hits = oss.image_knn_search(vec.tolist(), k=5)
    assert hits, "kNN không trả hit nào"
    top_pids = [h["product_id"] for h in hits]
    assert 1 in top_pids[:3], (
        f"vector search không tìm thấy SP #1 trong top-3: {hits[:3]}"
    )
    # cosine top hit nên khá cao (cùng ảnh hoặc rất gần)
    assert hits[0]["cosine"] > 0.5, f"cosine top quá thấp: {hits[0]}"


@pytest.mark.integration
@requires_api
def test_import_catalog_then_cleanup(api, admin_token):
    """Nạp 1 SP catalog tạm (dùng cover SP có sẵn) → kiểm tra DB → xoá sạch."""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
    import db_service
    from sqlalchemy import text
    from models import get_engine
    import opensearch_service as oss

    # product_id cao, tránh đụng catalog thật
    test_pid = 999001
    # dọn nếu test trước crash
    _cleanup_test_product(test_pid)

    p1 = db_service.get_product_by_id(1)
    assert p1 and p1.image
    img = p1.image if isinstance(p1.image, dict) else json.loads(p1.image)

    row = {
        "product_id": test_pid,
        "name": "TEST ONLY — Vòng catalog import (xoá sau test)",
        "category": "vòng tay",
        "material": "mã não",
        "colors": "trắng; xanh dương",
        "compatible_elements": "Thủy; Mộc",
        "product_size": "8mm",
        "price_range": "99.000 - 199.000",
        "brand": "Vạn An Group",
        "origin": "Việt Nam",
        "warranty": "thay dây trọn đời",
        "quantity_max": 10,
        "in_stock": True,
        "product_description": "Sản phẩm test import catalog — không bán.",
        "image": json.dumps(img, ensure_ascii=False),
    }
    df = pd.DataFrame([row])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="catalog")
    xlsx = buf.getvalue()

    try:
        r = api.post(
            "/admin/import/catalog",
            admin_token,
            files={"file": ("catalog_test.xlsx", xlsx, XLSX_MIME)},
            timeout=600,  # SigLIP embed có thể lâu
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("created", 0) + data.get("edited", 0) >= 1
        assert data.get("image_vectors_indexed", 0) >= 1

        got = db_service.get_product_by_id(test_pid)
        assert got is not None
        assert "TEST ONLY" in got.name
        assert got.image and (got.image.get("cover") if isinstance(got.image, dict) else True)

        # Vector search: ảnh cover (cùng SP#1) có thể trả #1 hoặc #999001 — cả hai OK
        # vì cùng vector. Kiểm tra #999001 có trong index.
        client = oss.get_client()
        q = client.search(
            index=oss.config.OS_IMAGE_INDEX,
            body={"size": 1, "query": {"term": {"product_id": test_pid}}},
        )
        assert q["hits"]["total"]["value"] >= 1 or q["hits"]["hits"], (
            "SP test chưa có vector ảnh trong OpenSearch"
        )
    finally:
        _cleanup_test_product(test_pid)


def _cleanup_test_product(pid: int):
    try:
        from models import get_engine
        from sqlalchemy import text
        import opensearch_service as oss

        eng = get_engine()
        with eng.begin() as conn:
            conn.execute(text("DELETE FROM products WHERE product_id = :p"), {"p": pid})
        try:
            client = oss.get_client()
            client.delete_by_query(
                index=oss.config.OS_IMAGE_INDEX,
                body={"query": {"term": {"product_id": pid}}},
                refresh=True,
            )
        except Exception:
            pass
        try:
            client = oss.get_client()
            if client.exists(index=oss.config.OS_INDEX, id=str(pid)):
                client.delete(index=oss.config.OS_INDEX, id=str(pid), refresh=True)
        except Exception:
            pass
    except Exception:
        pass
