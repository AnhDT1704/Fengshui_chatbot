"""
opensearch_service.py – OpenSearch index management & document operations.

Index mapping supports:
  - kNN vector field for Semantic Search
  - text fields with Vietnamese-friendly analyzer for Keyword Search
  - structured metadata fields for Filter Search
"""

from typing import List, Dict, Optional
from opensearchpy import OpenSearch, helpers

import config


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": config.OS_HOST, "port": config.OS_PORT}],
        http_compress=True,
        use_ssl=False,                     # No SSL (security plugin disabled)
        verify_certs=False,
        ssl_show_warn=False,
        scheme="http",                     # HTTP only (not HTTPS)
        timeout=30,
    )


# ═══════════════════════════════════════════════════════════════════
#  INDEX MAPPING
# ═══════════════════════════════════════════════════════════════════
INDEX_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,                  # Enable kNN plugin
            "knn.algo_param.ef_search": 100,
        },
        "number_of_shards": 1,            # Single node → 1 shard
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "vi_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "vi_stop"],
                },
            },
            "filter": {
                "vi_stop": {
                    "type": "stop",
                    "stopwords": [
                        "và", "của", "là", "cho", "với", "các", "một",
                        "được", "trong", "có", "này", "đã", "để", "không",
                        "từ", "những", "về", "như", "khi", "sẽ", "cũng",
                        "tại", "theo", "nên", "mà", "thì",
                    ],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            # ── Vector field for Semantic Search ──
            "embedding": {
                "type": "knn_vector",
                "dimension": config.EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16,
                    },
                },
            },
            # ── Text field for Keyword / Fulltext Search ──
            "product_description": {
                "type": "text",
                "analyzer": "vi_analyzer",
                "fields": {
                    "raw": {"type": "keyword", "ignore_above": 5000},
                },
            },
            # ── Structured metadata for Filter Search ──
            "product_id": {
                "type": "integer",
            },
            "name": {
                "type": "text",
                "analyzer": "vi_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 500},
                },
            },
            "category": {
                "type": "keyword",
            },
            "material": {
                "type": "keyword",
            },
            "compatible_elements": {
                "type": "keyword",
            },
            "colors": {
                "type": "keyword",
            },
            "product_size": {
                "type": "keyword",
            },
            "brand": {
                "type": "keyword",
            },
            "in_stock": {
                "type": "boolean",
            },
            "price_range": {
                "type": "keyword",
            },
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
#  INDEX OPERATIONS
# ═══════════════════════════════════════════════════════════════════
def create_index(delete_existing: bool = False):
    """Create the OpenSearch index with mapping."""
    client = get_client()
    index = config.OS_INDEX

    if client.indices.exists(index=index):
        if delete_existing:
            client.indices.delete(index=index)
            print(f"  ✓ Deleted existing index: {index}")
        else:
            print(f"  ℹ Index '{index}' already exists, skipping creation")
            return

    client.indices.create(index=index, body=INDEX_SETTINGS)
    print(f"  ✓ Created index: {index}")


def delete_index():
    """Delete the index."""
    client = get_client()
    if client.indices.exists(index=config.OS_INDEX):
        client.indices.delete(index=config.OS_INDEX)
        print(f"  ✓ Deleted index: {config.OS_INDEX}")


# ═══════════════════════════════════════════════════════════════════
#  DOCUMENT OPERATIONS
# ═══════════════════════════════════════════════════════════════════
def index_product(
    product_id: int,
    product_description: str,
    embedding: List[float],
    metadata: Dict,
):
    """Index a single product document."""
    client = get_client()

    doc = {
        "product_id":          product_id,
        "product_description": product_description,
        "embedding":           embedding,
        "name":                metadata.get("name", ""),
        "category":            metadata.get("category", ""),
        "material":            metadata.get("material", []),
        "compatible_elements": metadata.get("compatible_elements", []),
        "colors":              metadata.get("colors", []),
        "product_size":        metadata.get("product_size", metadata.get("bead_sizes", [])),
        "brand":               metadata.get("brand", "Vạn An Group"),
        "in_stock":            metadata.get("in_stock", True),
        "price_range":         metadata.get("price_range"),
    }

    client.index(
        index=config.OS_INDEX,
        id=str(product_id),     # Use product_id as doc ID for easy upsert
        body=doc,
        refresh=True,
    )


def bulk_index_products(documents: List[Dict]):
    """
    Bulk index product documents.

    Args:
        documents: List of {
            "product_id": int,
            "product_description": str,
            "embedding": [float],
            "metadata": dict,
        }
    """
    client = get_client()
    actions = []

    for doc in documents:
        action = {
            "_index":  config.OS_INDEX,
            "_id":     str(doc["product_id"]),
            "_source": {
                "product_id":          doc["product_id"],
                "product_description": doc["product_description"],
                "embedding":           doc["embedding"],
                "name":                doc["metadata"].get("name", ""),
                "category":            doc["metadata"].get("category", ""),
                "material":            doc["metadata"].get("material", []),
                "compatible_elements": doc["metadata"].get("compatible_elements", []),
                "colors":              doc["metadata"].get("colors", []),
                "product_size":        doc["metadata"].get("product_size", doc["metadata"].get("bead_sizes", [])),
                "brand":               doc["metadata"].get("brand", "Vạn An Group"),
                "in_stock":            doc["metadata"].get("in_stock", True),
                "price_range":         doc["metadata"].get("price_range"),
            },
        }
        actions.append(action)

    success, errors = helpers.bulk(client, actions, refresh=True)
    print(f"  ✓ Bulk indexed {success} documents")
    if errors:
        print(f"  ⚠ {len(errors)} errors during bulk indexing")
        for err in errors[:3]:
            print(f"    {err}")

    return success, errors


def get_doc_count() -> int:
    """Get total document count in the index."""
    client = get_client()
    try:
        result = client.count(index=config.OS_INDEX)
        return result["count"]
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════
#  SEARCH OPERATIONS (for testing pipeline output)
# ═══════════════════════════════════════════════════════════════════
def semantic_search(
    query_embedding: List[float],
    k: int = 5,
    filters: Optional[Dict] = None,
) -> List[Dict]:
    """kNN vector search with optional metadata filters."""
    client = get_client()

    knn_query = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k,
                },
            },
        },
    }

    # Add filter if provided
    if filters:
        knn_query["query"] = {
            "bool": {
                "must": [
                    {"knn": {"embedding": {"vector": query_embedding, "k": k}}},
                ],
                "filter": [filters],
            },
        }

    result = client.search(index=config.OS_INDEX, body=knn_query)
    return [
        {
            "product_id": hit["_source"]["product_id"],
            "name":       hit["_source"]["name"],
            "score":      hit["_score"],
            "category":   hit["_source"]["category"],
        }
        for hit in result["hits"]["hits"]
    ]


def keyword_search(query: str, k: int = 5) -> List[Dict]:
    """Fulltext keyword search on product_description and name."""
    client = get_client()

    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["name^3", "product_description"],
                "type": "best_fields",
            },
        },
    }

    result = client.search(index=config.OS_INDEX, body=body)
    return [
        {
            "product_id": hit["_source"]["product_id"],
            "name":       hit["_source"]["name"],
            "score":      hit["_score"],
            "category":   hit["_source"]["category"],
        }
        for hit in result["hits"]["hits"]
    ]


def filter_search(filters: Dict, k: int = 10) -> List[Dict]:
    """Structured metadata filter search."""
    client = get_client()

    must_clauses = []

    if "category" in filters:
        must_clauses.append({"term": {"category": filters["category"]}})
    if "material" in filters:
        must_clauses.append({"term": {"material": filters["material"]}})
    if "compatible_elements" in filters:
        must_clauses.append(
            {"term": {"compatible_elements": filters["compatible_elements"]}}
        )
    if "colors" in filters:
        must_clauses.append({"term": {"colors": filters["colors"]}})
    if "in_stock" in filters:
        must_clauses.append({"term": {"in_stock": filters["in_stock"]}})
    if "price_range" in filters:
        must_clauses.append({"term": {"price_range": filters["price_range"]}})

    body = {
        "size": k,
        "query": {"bool": {"must": must_clauses}} if must_clauses else {"match_all": {}},
    }

    result = client.search(index=config.OS_INDEX, body=body)
    return [
        {
            "product_id": hit["_source"]["product_id"],
            "name":       hit["_source"]["name"],
            "category":   hit["_source"]["category"],
            "price_range": hit["_source"].get("price_range"),
        }
        for hit in result["hits"]["hits"]
    ]
