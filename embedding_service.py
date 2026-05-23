"""
embedding_service.py – Generate embeddings via OpenRouter (text-embedding-3-small).

Supports both single and batch embedding with rate-limit retry.
"""

import time
from typing import List, Optional

from openai import OpenAI

import config


def _get_client() -> OpenAI:
    return OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
    )


def embed_single(text: str, max_retries: int = 3) -> List[float]:
    """Embed a single text string. Returns a list of floats (1536-dim)."""
    client = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                extra_headers={
                    "HTTP-Referer": "https://vanangroup.com",
                    "X-OpenRouter-Title": "FengShuiChatbot",
                },
                model=config.EMBEDDING_MODEL,
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  ⚠ Embedding error (attempt {attempt+1}): {e}")
                print(f"    Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Embedding failed after {max_retries} attempts: {e}")


def embed_batch(
    texts: List[str],
    batch_size: int = 20,
    delay_between_batches: float = 1.0,
) -> List[List[float]]:
    """
    Embed multiple texts in batches.

    Args:
        texts: List of text strings to embed
        batch_size: Number of texts per API call (OpenRouter limit varies)
        delay_between_batches: Seconds to wait between batches (rate limiting)

    Returns:
        List of embedding vectors, same order as input texts
    """
    client = _get_client()
    all_embeddings: List[List[float]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(f"  Embedding batch {batch_num}/{total_batches} "
              f"({len(batch)} texts)...")

        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    extra_headers={
                        "HTTP-Referer": "https://vanangroup.com",
                        "X-OpenRouter-Title": "FengShuiChatbot",
                    },
                    model=config.EMBEDDING_MODEL,
                    input=batch,
                    encoding_format="float",
                )
                # OpenAI API returns embeddings sorted by index
                batch_embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(batch_embeddings)
                break

            except Exception as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(f"    ⚠ Batch error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

        # Rate limit between batches
        if i + batch_size < len(texts):
            time.sleep(delay_between_batches)

    return all_embeddings


# ── Quick test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing single embedding...")
    vec = embed_single("Vòng tay đá Aquamarine mệnh Thủy")
    print(f"  Dimension: {len(vec)}")
    print(f"  First 5 values: {vec[:5]}")

    print("\nTesting batch embedding...")
    texts = [
        "Vòng tay đá mã não đen phong thủy",
        "Nhang nụ trầm hương tự nhiên",
    ]
    vecs = embed_batch(texts)
    print(f"  Got {len(vecs)} vectors, dim={len(vecs[0])}")
