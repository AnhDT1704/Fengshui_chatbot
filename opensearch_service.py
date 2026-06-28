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


# ═══════════════════════════════════════════════════════════════════
#  IMAGE VECTOR INDEX (Visual Search – SigLIP 2, 768-dim)
#  Index riêng vì dimension ảnh (768) khác text (1536). Mỗi ẢNH = 1 doc,
#  gắn product_id, để kNN trả về ảnh gần nhất rồi suy ra sản phẩm.
# ═══════════════════════════════════════════════════════════════════
IMAGE_INDEX_SETTINGS = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        },
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "image_embedding": {
                "type": "knn_vector",
                "dimension": config.IMAGE_EMBEDDING_DIM,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
            "product_id": {"type": "integer"},
            "image_url":  {"type": "keyword"},
            "is_cover":   {"type": "boolean"},
        },
    },
}


def create_image_index(delete_existing: bool = False):
    """Create the image-vector index."""
    client = get_client()
    index = config.OS_IMAGE_INDEX
    if client.indices.exists(index=index):
        if delete_existing:
            client.indices.delete(index=index)
            print(f"  ✓ Deleted existing image index: {index}")
        else:
            print(f"  ℹ Image index '{index}' already exists, skipping")
            return
    client.indices.create(index=index, body=IMAGE_INDEX_SETTINGS)
    print(f"  ✓ Created image index: {index}")


def bulk_index_image_vectors(documents: List[Dict]):
    """Bulk index image vectors.

    Args:
        documents: list of {product_id:int, image_url:str, embedding:[float],
                            is_cover:bool}. doc id = image_url (idempotent upsert).
    """
    client = get_client()
    actions = []
    for doc in documents:
        actions.append({
            # _id gồm cả product_id: 1 ảnh có thể thuộc NHIỀU sản phẩm (biến thể
            # chung ảnh listing) → giữ đủ từng cặp (sản phẩm, ảnh), không ghi đè.
            "_index": config.OS_IMAGE_INDEX,
            "_id":    f"{doc['product_id']}::{doc['image_url']}",
            "_source": {
                "product_id":      doc["product_id"],
                "image_url":       doc["image_url"],
                "is_cover":        doc.get("is_cover", False),
                "image_embedding": doc["embedding"],
            },
        })
    success, errors = helpers.bulk(client, actions, refresh=True)
    print(f"  ✓ Bulk indexed {success} image vectors")
    if errors:
        print(f"  ⚠ {len(errors)} errors during image bulk indexing")
        for err in errors[:3]:
            print(f"    {err}")
    return success, errors


def get_image_doc_count() -> int:
    client = get_client()
    try:
        return client.count(index=config.OS_IMAGE_INDEX)["count"]
    except Exception:
        return 0


def image_knn_search(query_embedding: List[float], k: int = 10) -> List[Dict]:
    """kNN over image vectors. Trả về từng ẢNH gần nhất kèm product_id + score.

    LƯU Ý score: với space_type=cosinesimil (engine lucene), OpenSearch trả
    _score = (1 + cosine) / 2. Cosine thật = 2*_score - 1 (đã tính sẵn ở 'cosine').
    """
    client = get_client()
    body = {
        "size": k,
        "query": {"knn": {"image_embedding": {"vector": query_embedding, "k": k}}},
    }
    result = client.search(index=config.OS_IMAGE_INDEX, body=body)
    hits = []
    for hit in result["hits"]["hits"]:
        score = hit["_score"]
        hits.append({
            "product_id": hit["_source"]["product_id"],
            "image_url":  hit["_source"]["image_url"],
            "is_cover":   hit["_source"].get("is_cover", False),
            "score":      score,
            "cosine":     2.0 * score - 1.0,
        })
    return hits


def filter_search(filters: Dict, k: int = 10) -> List[Dict]:
    """Structured metadata filter search."""
    client = get_client()

    def _match(field, value):
        # Cho phép truyền nhiều giá trị ngăn cách dấu phẩy (vd "Thủy, Kim") →
        # terms query (khớp BẤT KỲ giá trị nào). Một giá trị → term thường.
        if isinstance(value, str) and "," in value:
            vals = [v.strip() for v in value.split(",") if v.strip()]
            return {"terms": {field: vals}}
        return {"term": {field: value}}

    must_clauses = []

    if "category" in filters:
        must_clauses.append(_match("category", filters["category"]))
    if "material" in filters:
        must_clauses.append(_match("material", filters["material"]))
    if "compatible_elements" in filters:
        must_clauses.append(_match("compatible_elements", filters["compatible_elements"]))
    if "colors" in filters:
        must_clauses.append(_match("colors", filters["colors"]))
    if "in_stock" in filters:
        must_clauses.append({"term": {"in_stock": filters["in_stock"]}})
    if "price_range" in filters:
        must_clauses.append(_match("price_range", filters["price_range"]))

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
