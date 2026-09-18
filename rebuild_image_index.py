"""Rebuild the visual image index from PostgreSQL product.image JSON.

Rules:
- ignore the cover image
- keep every valid image.images[] entry, even when color is null/empty
- store an empty color for images without color metadata
- use SigLIP to embed every valid image
- delete the existing image index and reindex new vectors

This script intentionally stores only minimal metadata needed for retrieval:
- product_id
- name
- color
- image_url

Full product details remain in PostgreSQL, to be joined by product_id after search.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "langraph pipeline"))

import db_service
import image_embedding as IE
import opensearch_service as oss


def _extract_valid_image_entries(product: Any) -> List[Dict[str, Any]]:
    """Return a list of valid image entries from product.image JSON.

    Schema expected:
    {
      "cover": "https://...",
      "images": [
        {"url": "https://...", "color": "xanh d╞░╞íng v├á trß║»ng"},
        ...
      ]
    }
    """
    if not product or not getattr(product, "image", None):
        return []

    image_data = product.image
    rows: List[Dict[str, Any]] = []

    if isinstance(image_data, dict):
        images = image_data.get("images") or []
        for item in images:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url or not str(url).strip():
                continue
            rows.append({
                "product_id": product.product_id,
                "name": product.name,
                "color": str(item.get("color") or "").strip(),
                "image_url": url,
                "is_cover": False,
            })

    elif isinstance(image_data, list):
        for item in image_data:
            if isinstance(item, dict):
                url = item.get("url")
                if not url or not str(url).strip():
                    continue
                rows.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "color": str(item.get("color") or "").strip(),
                    "image_url": url,
                    "is_cover": False,
                })
            elif isinstance(item, str):
                # fallback: treat plain URLs as valid non-cover images, without color metadata
                if not item.strip():
                    continue
                rows.append({
                    "product_id": product.product_id,
                    "name": product.name,
                    "color": "",
                    "image_url": item.strip(),
                    "is_cover": False,
                })

    # final cleanup: keep every distinct valid URL; color is optional metadata.
    seen = set()
    clean: List[Dict[str, Any]] = []
    for row in rows:
        key = (row["product_id"], row["image_url"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(row)
    return clean


def main() -> int:
    print("Loading products from PostgreSQL...")
    products = db_service.get_all_products()
    print(f"Found {len(products)} products")

    docs: List[Dict[str, Any]] = []
    skipped_total = 0
    valid_total = 0

    for product in products:
        entries = _extract_valid_image_entries(product)
        if not entries:
            skipped_total += 1
            continue

        for entry in entries:
            image_url = entry["image_url"]
            try:
                vec = IE.embed_url(image_url)
            except Exception as e:
                print(f"  ΓÜá embed failed for {image_url}: {e}")
                continue

            if vec is None:
                print(f"  ΓÜá download/embed failed for {image_url}")
                continue

            docs.append({
                "product_id": entry["product_id"],
                "name": entry["name"],
                "color": entry["color"],
                "image_url": image_url,
                "embedding": vec.tolist(),
                "is_cover": entry.get("is_cover", False),
            })

        valid_total += len(entries)

    print(f"Prepared {len(docs)} valid image vectors from {valid_total} valid entries")
    print(f"Skipped products without valid image entries: {skipped_total}")

    print("Deleting old image index...")
    oss.create_image_index(delete_existing=True)

    print("Bulk indexing new image vectors...")
    if docs:
        oss.bulk_index_image_vectors(docs)
    print(f"Total image vectors in OpenSearch: {oss.get_image_doc_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
