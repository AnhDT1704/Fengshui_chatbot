"""
db_service.py – PostgreSQL CRUD operations for products.
"""

import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional
from difflib import SequenceMatcher
from sqlalchemy import text

from models import Product, Boilerplate, get_session, create_tables, get_engine


def init_db():
    """Initialize database tables."""
    create_tables()
    ensure_products_schema()


def ensure_products_schema():
    """Migrate legacy products columns to the current schema."""
    engine = get_engine()
    with engine.begin() as conn:
        cols = {
            row[0]
            for row in conn.execute(text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'products'
                """
            ))
        }

        if "raw_text" in cols and "product_description" not in cols:
            conn.execute(text("ALTER TABLE products RENAME COLUMN raw_text TO product_description"))
            print("  ✓ Renamed products.raw_text -> products.product_description")
            cols.remove("raw_text")
            cols.add("product_description")

        if "product_description" not in cols:
            conn.execute(text("ALTER TABLE products ADD COLUMN product_description TEXT"))
            print("  ✓ Added products.product_description")

        if "chunk_text" in cols:
            conn.execute(text("ALTER TABLE products DROP COLUMN chunk_text"))
            print("  ✓ Dropped legacy column products.chunk_text")

        if "bead_sizes" in cols and "product_size" not in cols:
            conn.execute(text("ALTER TABLE products RENAME COLUMN bead_sizes TO product_size"))
            print("  ✓ Renamed products.bead_sizes -> products.product_size")
            cols.remove("bead_sizes")
            cols.add("product_size")

        if "product_size" not in cols:
            conn.execute(text("ALTER TABLE products ADD COLUMN product_size VARCHAR[]"))
            print("  ✓ Added products.product_size")

        if "image" not in cols:
            conn.execute(text("ALTER TABLE products ADD COLUMN image JSONB"))
            print("  ✓ Added products.image")


def _normalize_product_name(name: str) -> str:
    name = unicodedata.normalize("NFKC", name).lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[\"“”‘’(),.:;!?/\\\[\]{}<>|]", "", name)
    return name


def _name_variants(name: str) -> List[str]:
    """Generate a few normalized variants to improve name matching."""
    variants = []
    normalized = _normalize_product_name(name)
    variants.append(normalized)

    # Remove common ornamental/marketing phrases that differ between DB and image file.
    cleaned = normalized
    replacements = [
        "vạn an group",
        "trang sức phong thủy",
        "mang đến may mắn",
        "mang đến bình an",
        "mang đến tài lộc",
        "mang đến hạnh phúc",
        "mang đến may mắn, bình an",
        "mang đến may mắn, bình an, tài lộc",
        "mang đến bình an, tài lộc",
        "vòng phong thủy",
        "trang sức màu sắc dịu dàng",
        "trang sức phù hợp tất cả các mệnh",
        "trang sức phong thủy mang đến bình an tài lộc",
        "trang sức phong thủy mang đến may mắn bình an tài lộc",
        "trang sức phong thủy mang đến may mắn bình an",
    ]
    for phrase in replacements:
        cleaned = cleaned.replace(phrase, "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned not in variants:
        variants.append(cleaned)

    comma_part = name.split(",", 1)[0].strip()
    if comma_part:
        variants.append(_normalize_product_name(comma_part))

    group_split = re.split(r"(?i)\bvạn an group\b", name, maxsplit=1)
    if group_split and group_split[0].strip():
        variants.append(_normalize_product_name(group_split[0]))

    # Preserve order while removing duplicates and empties.
    cleaned = []
    seen = set()
    for variant in variants:
        if variant and variant not in seen:
            cleaned.append(variant)
            seen.add(variant)
    return cleaned


def parse_product_images_file(image_file_path: str) -> List[Dict]:
    """Parse the image mapping txt file into product name -> image URL list."""
    text = Path(image_file_path).read_text(encoding="utf-8")
    entries: List[Dict] = []
    current: Optional[Dict] = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("Mã sản phẩm:"):
            if current:
                entries.append(current)
            current = {
                "source_id": stripped.split(":", 1)[1].strip(),
                "name": "",
                "images": [],
            }
            continue

        if current is None:
            continue

        if stripped.startswith("Tên sản phẩm:"):
            current["name"] = stripped.split(":", 1)[1].strip()
            continue

        if "https://cf.shopee.vn/file/" in stripped:
            url = stripped.split()[-1]
            current["images"].append(url)

    if current:
        entries.append(current)

    return entries


def populate_product_images_from_file(image_file_path: str) -> Dict[str, int]:
    """Populate the image column for existing products using the image txt file."""
    init_db()

    entries = parse_product_images_file(image_file_path)
    normalized_entries = [
        (entry, _normalize_product_name(entry["name"]))
        for entry in entries
        if entry.get("name") and entry.get("images")
    ]
    entry_map = {}
    for entry, normalized_name in normalized_entries:
        entry_map.setdefault(normalized_name, []).append(entry)

    session = get_session()
    matched = 0
    exact_matched = 0
    fuzzy_matched = 0
    missing = []
    try:
        products = session.query(Product).order_by(Product.product_id).all()
        for product in products:
            entry = None
            for variant in _name_variants(product.name):
                if variant in entry_map:
                    entry = entry_map[variant][0]
                    exact_matched += 1
                    break

            if entry is None and normalized_entries:
                target = _normalize_product_name(product.name)
                best_entry = None
                best_score = 0.0
                for candidate, normalized_candidate in normalized_entries:
                    score = SequenceMatcher(None, target, normalized_candidate).ratio()
                    if score > best_score:
                        best_score = score
                        best_entry = candidate

                if best_entry is not None and best_score >= 0.58:
                    entry = best_entry
                    fuzzy_matched += 1

            if entry is None:
                missing.append(product.name)
                continue

            product.image = entry["images"]
            matched += 1

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return {
        "parsed_entries": len(entries),
        "matched_products": matched,
        "exact_matched_products": exact_matched,
        "fuzzy_matched_products": fuzzy_matched,
        "missing_products": len(missing),
        "extra_file_entries": max(len(entries) - matched, 0),
    }


def upsert_product(metadata: Dict, product_description: str) -> Product:
    """Insert or update a single product."""
    session = get_session()
    try:
        product = (
            session.query(Product)
            .filter(Product.product_id == metadata["product_id"])
            .first()
        )

        if product is None:
            product = Product()

        product.product_id          = metadata["product_id"]
        product.name                = metadata["name"]
        product.category            = metadata["category"]
        product.material            = metadata.get("material", [])
        product.compatible_elements = metadata.get("compatible_elements", [])
        product.colors              = metadata.get("colors", [])
        product.product_size        = metadata.get("product_size", metadata.get("bead_sizes", []))
        product.price_range         = metadata.get("price_range")
        product.brand               = metadata.get("brand", "Vạn An Group")
        product.origin              = metadata.get("origin", "Việt Nam")
        product.warranty            = metadata.get("warranty")
        product.in_stock            = metadata.get("in_stock", True)
        product.product_description = product_description

        session.add(product)
        session.commit()
        session.refresh(product)
        return product

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def upsert_products_batch(products_data: List[Dict]) -> int:
    """
    Batch upsert products.

    Args:
        products_data: List of {
            "metadata": dict,
            "product_description": str,
        }
    Returns:
        Number of products upserted
    """
    session = get_session()
    count = 0
    try:
        for item in products_data:
            meta = item["metadata"]
            pid = meta["product_id"]

            product = (
                session.query(Product)
                .filter(Product.product_id == pid)
                .first()
            )

            if product is None:
                product = Product()

            product.product_id          = pid
            product.name                = meta["name"]
            product.category            = meta["category"]
            product.material            = meta.get("material", [])
            product.compatible_elements = meta.get("compatible_elements", [])
            product.colors              = meta.get("colors", [])
            product.product_size        = meta.get("product_size", meta.get("bead_sizes", []))
            product.price_range         = meta.get("price_range")
            product.brand               = meta.get("brand", "Vạn An Group")
            product.origin              = meta.get("origin", "Việt Nam")
            product.warranty            = meta.get("warranty")
            product.in_stock            = meta.get("in_stock", True)
            product.product_description = item["product_description"]

            session.add(product)
            count += 1

        session.commit()
        return count

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def upsert_boilerplate(material_type: str, section_type: str, content: str):
    """Store shared boilerplate content."""
    session = get_session()
    try:
        bp = (
            session.query(Boilerplate)
            .filter(
                Boilerplate.material_type == material_type,
                Boilerplate.section_type == section_type,
            )
            .first()
        )
        if bp is None:
            bp = Boilerplate(
                material_type=material_type,
                section_type=section_type,
                content=content,
            )
        else:
            bp.content = content

        session.add(bp)
        session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_all_products() -> List[Product]:
    session = get_session()
    try:
        return session.query(Product).order_by(Product.product_id).all()
    finally:
        session.close()


def get_product_by_id(product_id: int) -> Optional[Product]:
    session = get_session()
    try:
        return (
            session.query(Product)
            .filter(Product.product_id == product_id)
            .first()
        )
    finally:
        session.close()


def get_products_count() -> int:
    session = get_session()
    try:
        return session.query(Product).count()
    finally:
        session.close()


def get_promotions_for_month(month: int) -> List[Dict]:
    """Return all promotions whose month matches, ordered by day.

    Used by the chatbot's promotion_info_tool to answer "shop đang có khuyến
    mãi gì" based on the current date.
    """
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT promo_date, day, month, discount_percent, scope, promotion_info
                FROM promotions
                WHERE month = :m
                ORDER BY day
            """),
            {"m": month},
        ).fetchall()
    return [
        {
            "promo_date":       r[0],
            "day":              r[1],
            "month":            r[2],
            "discount_percent": r[3],
            "scope":            r[4],
            "promotion_info":   r[5],
        }
        for r in rows
    ]


def get_category_summary() -> List[Dict]:
    """Get product count by category."""
    session = get_session()
    try:
        rows = (
            session.query(Product.category, text("count(*)"))
            .group_by(Product.category)
            .all()
        )
        return [{"category": r[0], "count": r[1]} for r in rows]
    finally:
        session.close()
