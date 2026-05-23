"""
pipeline.py – Main data processing pipeline.

Workflow:
  1. Parse raw .txt files → list of raw products
  2. Extract structured metadata from each product
    3. Prepare product descriptions
  4. Generate embeddings via OpenRouter
  5. Store metadata in PostgreSQL
    6. Index descriptions + vectors in OpenSearch
  7. Verify with test searches
"""

import json
import time
import argparse
from pathlib import Path

import config
from product_parser import parse_all_files
from metadata_extractor import extract_metadata
from embedding_service import embed_batch
import db_service
import opensearch_service
from models import create_tables, drop_tables


def step_1_parse(data_files: list) -> list:
    """Parse raw text files into product blocks."""
    print("\n═══ STEP 1: Parse raw data files ═══")
    products = parse_all_files(data_files)
    print(f"  ✓ Total raw products: {len(products)}")
    return products


def step_2_extract_metadata(raw_products: list) -> list:
    """Extract structured metadata from each product."""
    print("\n═══ STEP 2: Extract metadata ═══")
    enriched = []
    for p in raw_products:
        meta = extract_metadata(p["product_id"], p["raw_text"])
        enriched.append({
            "product_id": p["product_id"],
            "raw_text":   p["raw_text"],
            "metadata":   meta,
        })
    print(f"  ✓ Extracted metadata for {len(enriched)} products")

    # Print category summary
    categories = {}
    for e in enriched:
        cat = e["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("  Categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    return enriched


def step_3_build_chunks(enriched_products: list) -> list:
    """Prepare product descriptions for embedding."""
    print("\n═══ STEP 3: Prepare product descriptions ═══")
    prepared = []
    total_chars = 0

    for e in enriched_products:
        description = e["raw_text"].strip()
        prepared.append({
            **e,
            "product_description": description,
        })
        total_chars += len(description)

    avg_chars = total_chars // len(prepared) if prepared else 0
    print(f"  ✓ Prepared {len(prepared)} descriptions")
    print(f"  Average description size: {avg_chars} chars")

    # Show a sample
    if prepared:
        sample = prepared[0]
        print(f"\n  Sample description (product #{sample['product_id']}):")
        print(f"  {sample['product_description'][:300]}...")

    return prepared


def step_4_generate_embeddings(chunked_products: list) -> list:
    """Generate embeddings for all product descriptions."""
    print("\n═══ STEP 4: Generate embeddings ═══")

    texts = [c["product_description"] for c in chunked_products]
    print(f"  Embedding {len(texts)} descriptions...")
    start = time.time()

    embeddings = embed_batch(texts, batch_size=20, delay_between_batches=1.0)

    elapsed = time.time() - start
    print(f"  ✓ Generated {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  Vector dimension: {len(embeddings[0])}")

    for i, c in enumerate(chunked_products):
        c["embedding"] = embeddings[i]

    return chunked_products


def step_5_store_postgres(chunked_products: list):
    """Store products and metadata in PostgreSQL."""
    print("\n═══ STEP 5: Store in PostgreSQL ═══")

    db_service.init_db()

    batch_data = [
        {
            "metadata":            c["metadata"],
            "product_description": c["product_description"],
        }
        for c in chunked_products
    ]

    count = db_service.upsert_products_batch(batch_data)
    print(f"  ✓ Upserted {count} products to PostgreSQL")
    print(f"  Total in DB: {db_service.get_products_count()}")

    # Print category summary from DB
    summary = db_service.get_category_summary()
    print("  Category summary (from DB):")
    for s in summary:
        print(f"    {s['category']}: {s['count']}")


def step_6_index_opensearch(chunked_products: list):
    """Index product descriptions and vectors into OpenSearch."""
    print("\n═══ STEP 6: Index in OpenSearch ═══")

    opensearch_service.create_index(delete_existing=True)

    documents = [
        {
            "product_id":           c["product_id"],
            "product_description":  c["product_description"],
            "embedding":            c["embedding"],
            "metadata":             c["metadata"],
        }
        for c in chunked_products
    ]

    success, errors = opensearch_service.bulk_index_products(documents)
    print(f"  Total docs in index: {opensearch_service.get_doc_count()}")


def step_6_index_opensearch_from_postgres():
    """Index persisted PostgreSQL rows into OpenSearch."""
    print("\n═══ STEP 6: Index in OpenSearch from PostgreSQL ═══")

    opensearch_service.create_index(delete_existing=True)

    db_service.init_db()
    products = db_service.get_all_products()
    documents = []

    for product in products:
        documents.append({
            "product_id": product.product_id,
            "product_description": product.product_description or "",
            "embedding": [],
            "metadata": {
                "name": product.name,
                "category": product.category,
                "material": product.material or [],
                "compatible_elements": product.compatible_elements or [],
                "colors": product.colors or [],
                "product_size": product.product_size or [],
                "brand": product.brand,
                "in_stock": product.in_stock,
                "price_range": product.price_range,
            },
        })

    if not documents:
        print("  ⚠ No products found in PostgreSQL, skipping OpenSearch index")
        return

    # Rebuild embeddings from persisted descriptions so OpenSearch stays in sync.
    from embedding_service import embed_batch

    texts = [doc["product_description"] for doc in documents]
    print(f"  Re-embedding {len(texts)} PostgreSQL descriptions...")
    embeddings = embed_batch(texts, batch_size=20, delay_between_batches=1.0)

    for idx, doc in enumerate(documents):
        doc["embedding"] = embeddings[idx]

    success, errors = opensearch_service.bulk_index_products(documents)
    print(f"  Total docs in index: {opensearch_service.get_doc_count()}")


def step_7_verify(chunked_products: list):
    """Run test searches to verify the pipeline output."""
    print("\n═══ STEP 7: Verification ═══")

    from embedding_service import embed_single

    # Test semantic search
    print("\n  Test 1: Semantic Search – 'đá phù hợp mệnh Thủy'")
    query_vec = embed_single("đá phù hợp mệnh Thủy")
    results = opensearch_service.semantic_search(query_vec, k=3)
    for r in results:
        print(f"    #{r['product_id']} {r['name'][:60]} (score: {r['score']:.4f})")

    # Test keyword search
    print("\n  Test 2: Keyword Search – 'aquamarine'")
    results = opensearch_service.keyword_search("aquamarine", k=3)
    for r in results:
        print(f"    #{r['product_id']} {r['name'][:60]} (score: {r['score']:.4f})")

    # Test filter search
    print("\n  Test 3: Filter Search – category='nhang'")
    results = opensearch_service.filter_search({"category": "nhang"}, k=5)
    for r in results:
        print(f"    #{r['product_id']} {r['name'][:60]}")

    # Test filter search with element
    print("\n  Test 4: Filter Search – compatible_elements='Mộc'")
    results = opensearch_service.filter_search(
        {"compatible_elements": "Mộc"}, k=5
    )
    for r in results:
        print(f"    #{r['product_id']} {r['name'][:60]}")

    print("\n  ✓ Verification complete!")


def save_intermediate_json(chunked_products: list, output_path: str = "output"):
    """Save processed data as JSON for inspection."""
    Path(output_path).mkdir(exist_ok=True)

    # Save metadata + descriptions (without embeddings, too large)
    export = []
    for c in chunked_products:
        item = {
            "product_id": c["product_id"],
            "metadata":             c["metadata"],
            "product_description":  c["product_description"],
            "description_chars":    len(c["product_description"]),
        }
        export.append(item)

    filepath = Path(output_path) / "processed_products.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Saved processed data to {filepath}")


# ═══════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fengshui Products Data Pipeline"
    )
    parser.add_argument(
        "--steps", type=str, default="1,2,3,4,5,6,7",
        help="Comma-separated step numbers to run (default: all)"
    )
    parser.add_argument(
        "--reset-db", action="store_true",
        help="Drop and recreate PostgreSQL tables before running"
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save intermediate JSON output for inspection"
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="Skip embedding generation (steps 4,6,7 will be skipped)"
    )
    parser.add_argument(
        "--index-from-postgres", action="store_true",
        help="Rebuild OpenSearch documents from PostgreSQL rows instead of in-memory chunks"
    )
    args = parser.parse_args()

    steps = set(int(s) for s in args.steps.split(","))

    print("╔═══════════════════════════════════════════════════╗")
    print("║   Fengshui Products – Data Processing Pipeline    ║")
    print("╚═══════════════════════════════════════════════════╝")
    print(f"  Steps to run: {sorted(steps)}")
    print(f"  Data files: {config.DATA_FILE_1}, {config.DATA_FILE_2}")

    if args.reset_db:
        print("\n  ⚠ Resetting PostgreSQL database...")
        drop_tables()

    use_postgres_for_indexing = args.index_from_postgres and 6 in steps

    # ── Step 1: Parse ──
    data_files = [config.DATA_FILE_1, config.DATA_FILE_2]
    raw_products = step_1_parse(data_files) if 1 in steps else []

    if not raw_products and steps & {2, 3, 4, 5, 6, 7} and not use_postgres_for_indexing:
        print("  ⚠ No products parsed, running step 1 first...")
        raw_products = step_1_parse(data_files)

    # ── Step 2: Extract metadata ──
    enriched = step_2_extract_metadata(raw_products) if 2 in steps else []
    if not enriched and steps & {3, 4, 5, 6, 7} and not use_postgres_for_indexing:
        enriched = step_2_extract_metadata(raw_products)

    # ── Step 3: Build chunks ──
    chunked = step_3_build_chunks(enriched) if 3 in steps else []
    if not chunked and steps & {4, 5, 6, 7} and not use_postgres_for_indexing:
        chunked = step_3_build_chunks(enriched)

    # ── Save intermediate JSON ──
    if args.save_json and chunked:
        save_intermediate_json(chunked)

    # ── Step 4: Generate embeddings ──
    if 4 in steps and not args.skip_embedding:
        chunked = step_4_generate_embeddings(chunked)

    # ── Step 5: Store in PostgreSQL ──
    if 5 in steps:
        step_5_store_postgres(chunked)

    # ── Step 6: Index in OpenSearch ──
    if 6 in steps and not args.skip_embedding:
        if use_postgres_for_indexing:
            if 5 not in steps:
                print("  ⚠ --index-from-postgres works best with step 5 or an already populated PostgreSQL table")
            step_6_index_opensearch_from_postgres()
        else:
            step_6_index_opensearch(chunked)

    # ── Step 7: Verify ──
    if 7 in steps and not args.skip_embedding:
        step_7_verify(chunked)

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    main()
