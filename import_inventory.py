"""
import_inventory.py – Import inventory quantities from `so luong san pham.txt`
into the products table by fuzzy-matching product names.

File format: each line is "<product_name>: <quantity_or_range>"
  - Single number: "94375"          → min=max=94375
  - Hyphen range:  "976561-976564"  → min=976561, max=976564
  - Slash range:   "943726/947932"  → min/max sorted

Matching strategy:
  1. Normalize name (lowercase, strip punctuation, drop marketing phrases)
  2. Try exact match across normalized variants
  3. Fallback to SequenceMatcher fuzzy match (>= 0.6)
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy import text

from db_service import _name_variants, _normalize_product_name
from models import get_engine


FILE_PATH = Path(__file__).parent / "so luong san pham.txt"
FUZZY_THRESHOLD = 0.60


def parse_quantity(raw: str) -> tuple[int, int]:
    s = raw.strip()
    parts = re.split(r"[-/]", s)
    nums = [int(p) for p in parts if p.strip().isdigit()]
    if not nums:
        raise ValueError(f"Cannot parse quantity: {raw!r}")
    return min(nums), max(nums)


def parse_file(path: Path) -> list[tuple[int, int, str]]:
    """Return list of (qty_min, qty_max, raw_name)."""
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, qty_raw = line.rsplit(":", 1)
        try:
            qmin, qmax = parse_quantity(qty_raw)
        except ValueError as e:
            print(f"  [warn] Line {idx} skipped ({e})")
            continue
        rows.append((qmin, qmax, name.strip()))
    return rows


def ensure_columns(engine):
    with engine.begin() as conn:
        existing = {
            row[0]
            for row in conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='products'"
            ))
        }
        if "quantity_min" not in existing:
            conn.execute(text("ALTER TABLE products ADD COLUMN quantity_min INT"))
            print("  + Added column products.quantity_min")
        if "quantity_max" not in existing:
            conn.execute(text("ALTER TABLE products ADD COLUMN quantity_max INT"))
            print("  + Added column products.quantity_max")


def reset_quantity_columns(engine):
    """Clear any previous (possibly wrong) import."""
    with engine.begin() as conn:
        conn.execute(text("UPDATE products SET quantity_min = NULL, quantity_max = NULL"))
        print("  + Cleared previous quantity_min / quantity_max")


def fetch_products(engine) -> list[tuple[int, str]]:
    with engine.begin() as conn:
        return [(r[0], r[1]) for r in conn.execute(
            text("SELECT product_id, name FROM products ORDER BY product_id")
        )]


def build_match_index(products: list[tuple[int, str]]) -> tuple[dict, list]:
    """
    exact_index : normalized_variant -> [product_id]
    normalized_list : [(product_id, normalized_name)]  (for fuzzy fallback)
    """
    exact_index: dict[str, list[int]] = {}
    normalized_list: list[tuple[int, str]] = []
    for pid, name in products:
        normalized_list.append((pid, _normalize_product_name(name)))
        for variant in _name_variants(name):
            exact_index.setdefault(variant, []).append(pid)
    return exact_index, normalized_list


def match_one(
    file_name: str,
    exact_index: dict,
    normalized_list: list,
    used_ids: set,
) -> tuple[int | None, str, float]:
    """
    Match a single file row to a product_id.
    Returns (product_id, method, score). product_id is None if no match.
    Prefers IDs not yet used (each product matched once).
    """
    for variant in _name_variants(file_name):
        candidates = exact_index.get(variant, [])
        unused = [pid for pid in candidates if pid not in used_ids]
        if unused:
            return unused[0], "exact", 1.0
        if candidates:
            return candidates[0], "exact-dup", 1.0

    # Fuzzy fallback
    target = _normalize_product_name(file_name)
    best_pid, best_score = None, 0.0
    best_pid_any, best_score_any = None, 0.0
    for pid, normalized in normalized_list:
        score = SequenceMatcher(None, target, normalized).ratio()
        if score > best_score_any:
            best_pid_any, best_score_any = pid, score
        if pid not in used_ids and score > best_score:
            best_pid, best_score = pid, score

    if best_pid is not None and best_score >= FUZZY_THRESHOLD:
        return best_pid, "fuzzy", best_score
    if best_pid_any is not None and best_score_any >= FUZZY_THRESHOLD:
        return best_pid_any, "fuzzy-dup", best_score_any
    return None, "none", best_score_any


def import_rows(engine, file_rows: list[tuple[int, int, str]]) -> dict:
    products = fetch_products(engine)
    exact_index, normalized_list = build_match_index(products)

    stats = {
        "total_file_rows":  len(file_rows),
        "matched":          0,
        "exact":            0,
        "fuzzy":            0,
        "unmatched":        [],
        "duplicates":       [],   # file rows matched to already-used pid
    }

    pid_to_qty: dict[int, tuple[int, int]] = {}
    used: set[int] = set()

    for qmin, qmax, file_name in file_rows:
        pid, method, score = match_one(file_name, exact_index, normalized_list, used)
        if pid is None:
            stats["unmatched"].append(file_name[:60])
            continue

        if pid in used:
            # Duplicate match: pick the larger quantity (assume same product, latest data wins)
            stats["duplicates"].append((pid, file_name[:50]))
            old_min, old_max = pid_to_qty[pid]
            pid_to_qty[pid] = (max(old_min, qmin), max(old_max, qmax))
            continue

        pid_to_qty[pid] = (qmin, qmax)
        used.add(pid)
        stats["matched"] += 1
        if method.startswith("exact"):
            stats["exact"] += 1
        else:
            stats["fuzzy"] += 1

    with engine.begin() as conn:
        for pid, (qmin, qmax) in pid_to_qty.items():
            conn.execute(
                text("""
                    UPDATE products
                    SET quantity_min = :qmin,
                        quantity_max = :qmax,
                        in_stock     = (:qmax > 0)
                    WHERE product_id = :pid
                """),
                {"qmin": qmin, "qmax": qmax, "pid": pid},
            )

    all_ids = {pid for pid, _ in products}
    stats["unfilled_pids"] = sorted(all_ids - set(pid_to_qty.keys()))
    return stats


def main():
    if not FILE_PATH.exists():
        sys.exit(f"File not found: {FILE_PATH}")

    engine = get_engine()

    print(f"Parsing {FILE_PATH.name}...")
    file_rows = parse_file(FILE_PATH)
    print(f"  -> {len(file_rows)} valid lines parsed")

    print("Ensuring schema...")
    ensure_columns(engine)

    print("Clearing previous quantity columns (reset)...")
    reset_quantity_columns(engine)

    print("Matching and importing...")
    stats = import_rows(engine, file_rows)

    print()
    print(f"  File rows:        {stats['total_file_rows']}")
    print(f"  Matched:          {stats['matched']}  "
          f"(exact={stats['exact']}, fuzzy={stats['fuzzy']})")
    print(f"  Unmatched:        {len(stats['unmatched'])}")
    print(f"  Duplicate file rows -> same product: {len(stats['duplicates'])}")
    print(f"  Products WITHOUT quantity: {len(stats['unfilled_pids'])} -> {stats['unfilled_pids'][:20]}")

    if stats["unmatched"]:
        print("\n  Unmatched file rows (first 10):")
        for n in stats["unmatched"][:10]:
            print(f"    - {n}")

    if stats["duplicates"]:
        print("\n  Duplicate file rows (first 10):")
        for pid, name in stats["duplicates"][:10]:
            print(f"    - pid={pid}: {name}")


if __name__ == "__main__":
    main()
