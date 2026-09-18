"""Restore the full product catalog snapshot into PostgreSQL.

Run after scripts/init_database.py:
    python scripts/seed_catalog.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import Product, get_session


SNAPSHOT_PATH = ROOT / "database" / "catalog_seed.json"


def main() -> None:
    if not SNAPSHOT_PATH.exists():
        raise SystemExit(f"Catalog snapshot not found: {SNAPSHOT_PATH}")

    products_data = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if not isinstance(products_data, list):
        raise SystemExit("Catalog snapshot must be a JSON list of products.")

    session = get_session()
    try:
        for product_data in products_data:
            product_id = product_data["product_id"]
            product = session.query(Product).filter_by(product_id=product_id).one_or_none()
            if product is None:
                session.add(Product(**product_data))
            else:
                for field, value in product_data.items():
                    setattr(product, field, value)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"Restored {len(products_data)} products from database/catalog_seed.json.")


if __name__ == "__main__":
    main()