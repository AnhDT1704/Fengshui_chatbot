"""
image_embedding.py – SigLIP 2 image embeddings for VISUAL product search.

Biến CHÍNH BỨC ẢNH thành vector (không qua mô tả chữ), để so ảnh-với-ảnh bằng
cosine / kNN. Dùng cho:
  - sync offline: embed toàn bộ ảnh sản phẩm 1 lần (build_image_index)
  - query: embed 1 ảnh khách gửi rồi tìm sản phẩm gần nhất

Model self-host (miễn phí, chạy local). Mặc định SigLIP 2 base; tự fallback về
SigLIP base nếu transformers chưa hỗ trợ siglip2. Torch import nằm TRONG hàm nên
import module này không bắt buộc phải có torch sẵn.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Optional, Union

import numpy as np

# Cho phép override qua .env nếu muốn đổi model (vd MobileCLIP qua open_clip sau này)
_PRIMARY_MODEL  = os.getenv("IMAGE_EMBED_MODEL", "google/siglip2-base-patch16-224")
_FALLBACK_MODEL = "google/siglip-base-patch16-224"

ImageLike = Union[bytes, bytearray, str, "object"]  # bytes | path | PIL.Image


@lru_cache(maxsize=1)
def _load():
    """Load model + processor once (cached). Returns (model, processor, model_id)."""
    import torch  # noqa: F401  (kept for side-effect / availability check)
    from transformers import AutoModel, AutoProcessor

    for model_id in (_PRIMARY_MODEL, _FALLBACK_MODEL):
        try:
            model = AutoModel.from_pretrained(model_id)
            proc = AutoProcessor.from_pretrained(model_id)
            model.eval()
            return model, proc, model_id
        except Exception as e:  # pragma: no cover - depends on transformers version
            last = e
            continue
    raise RuntimeError(f"Không load được model SigLIP ({_PRIMARY_MODEL}/{_FALLBACK_MODEL}): {last}")


def model_id() -> str:
    """Tên model thực sự đang dùng (sau fallback)."""
    return _load()[2]


def _to_pil(img: ImageLike):
    from PIL import Image
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    # assume already a PIL.Image
    return img.convert("RGB")


def embed_image(img: ImageLike) -> np.ndarray:
    """Embed 1 ảnh (bytes | path | PIL.Image) → vector float32 đã chuẩn hoá L2."""
    import torch

    pil = _to_pil(img)
    model, proc, _ = _load()
    inputs = proc(images=pil, return_tensors="pt")
    with torch.no_grad():
        out = model.get_image_features(**inputs)

    # transformers 5.x: get_image_features có thể trả Tensor HOẶC một
    # BaseModelOutputWithPooling. Vector ảnh ĐÚNG là pooler_output (head pooling
    # của SigLIP) — KHÔNG phải mean-pool các patch (last_hidden_state).
    if isinstance(out, torch.Tensor):
        t = out
    else:
        t = getattr(out, "pooler_output", None)
        if t is None:
            t = out.last_hidden_state.mean(dim=1)  # fallback hiếm gặp

    arr = t.detach().cpu().numpy().astype("float32")
    if arr.ndim == 3:
        arr = arr.mean(axis=1)
    v = arr.reshape(-1).astype("float32")
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def download_bytes(url: str, timeout: int = 20, retries: int = 3) -> Optional[bytes]:
    """Tải ảnh từ URL với retry (Shopee CDN hay rate-limit). None nếu thất bại."""
    import time
    import requests
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    return None


def embed_url(url: str, timeout: int = 20) -> Optional[np.ndarray]:
    """Tải ảnh từ URL rồi embed. Trả về None nếu tải/đọc lỗi."""
    b = download_bytes(url, timeout=timeout)
    return embed_image(b) if b is not None else None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine giữa 2 vector đã chuẩn hoá L2 = dot product."""
    return float(np.dot(a.reshape(-1), b.reshape(-1)))
