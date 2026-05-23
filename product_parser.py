"""
product_parser.py – Parse raw .txt files into a list of raw product dicts.

Both files use the format:
    --N--
    <product content>
    --N+1--

Returns list of {"product_id": int, "raw_text": str}
"""

import re
from pathlib import Path
from typing import List, Dict


# Regex to match product markers like --1--, --41--, --117--
MARKER_PATTERN = re.compile(r"^--(\d+)--\s*$")


def _read_file(filepath: str) -> str:
    """Read file with flexible encoding."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return Path(filepath).read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode file: {filepath}")


def parse_product_file(filepath: str) -> List[Dict]:
    """
    Parse a single txt file containing products separated by --N-- markers.

    Returns:
        List of {"product_id": int, "raw_text": str}
    """
    content = _read_file(filepath)
    lines = content.splitlines()

    products: List[Dict] = []
    current_id: int | None = None
    current_lines: List[str] = []

    for line in lines:
        match = MARKER_PATTERN.match(line.strip())
        if match:
            # Save previous product if exists
            if current_id is not None:
                raw_text = "\n".join(current_lines).strip()
                if raw_text:
                    products.append({
                        "product_id": current_id,
                        "raw_text": raw_text,
                    })
            # Start new product
            current_id = int(match.group(1))
            current_lines = []
        else:
            current_lines.append(line.rstrip("\r"))

    # Don't forget last product
    if current_id is not None:
        raw_text = "\n".join(current_lines).strip()
        if raw_text:
            products.append({
                "product_id": current_id,
                "raw_text": raw_text,
            })

    return products


def parse_all_files(filepaths: List[str]) -> List[Dict]:
    """Parse multiple files and merge, sorted by product_id."""
    all_products: List[Dict] = []
    for fp in filepaths:
        print(f"  Parsing: {fp}")
        parsed = parse_product_file(fp)
        print(f"    → Found {len(parsed)} products")
        all_products.extend(parsed)

    # Sort by product_id and check for duplicates
    all_products.sort(key=lambda p: p["product_id"])
    ids = [p["product_id"] for p in all_products]
    dupes = [pid for pid in ids if ids.count(pid) > 1]
    if dupes:
        print(f"  ⚠ Duplicate product_ids found: {set(dupes)}")

    return all_products


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import config

    products = parse_all_files([config.DATA_FILE_1, config.DATA_FILE_2])
    print(f"\nTotal products parsed: {len(products)}")
    for p in products[:3]:
        print(f"\n--- Product #{p['product_id']} ---")
        print(p["raw_text"][:200] + "...")
