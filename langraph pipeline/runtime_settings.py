"""
runtime_settings.py – Cấu hình runtime do ADMIN bật/tắt trên UI (lưu file JSON).

size_mode:
  - "code"     (mặc định, chế độ 2): tính size bằng hàm compute_bracelet + direct answer
  - "finetune" (chế độ 1): tính size qua model finetune phong thủy (FENGSHUI_API_URL)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from logger import get_logger

log = get_logger("runtime_settings")

_LOCK = threading.Lock()
_SETTINGS_PATH = Path(
    os.getenv(
        "RUNTIME_SETTINGS_PATH",
        str(Path(__file__).resolve().parent / "runtime_settings.json"),
    )
)

# Mặc định: chế độ 2 = code (nhanh, deterministic)
_DEFAULTS: dict[str, Any] = {
    "size_mode": "code",  # "code" | "finetune"
}

_VALID_SIZE_MODES = frozenset({"code", "finetune"})


def _read_file() -> dict[str, Any]:
    if not _SETTINGS_PATH.exists():
        return dict(_DEFAULTS)
    try:
        data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return dict(_DEFAULTS)
        out = dict(_DEFAULTS)
        out.update(data)
        return out
    except Exception as e:
        log.warning("Đọc runtime_settings lỗi (%s) → defaults", e)
        return dict(_DEFAULTS)


def _write_file(data: dict[str, Any]) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_SETTINGS_PATH)


def get_settings() -> dict[str, Any]:
    with _LOCK:
        data = _read_file()
    mode = str(data.get("size_mode") or "code").strip().lower()
    if mode not in _VALID_SIZE_MODES:
        mode = "code"
    return {"size_mode": mode}


def get_size_mode() -> str:
    """'code' | 'finetune' — mặc định 'code'."""
    return get_settings()["size_mode"]


def use_finetune_for_size() -> bool:
    return get_size_mode() == "finetune"


def update_settings(**kwargs: Any) -> dict[str, Any]:
    """Cập nhật một phần settings. Raise ValueError nếu giá trị không hợp lệ."""
    with _LOCK:
        data = _read_file()
        if "size_mode" in kwargs and kwargs["size_mode"] is not None:
            mode = str(kwargs["size_mode"]).strip().lower()
            if mode not in _VALID_SIZE_MODES:
                raise ValueError(
                    f"size_mode phải là 'code' hoặc 'finetune' (nhận {mode!r})"
                )
            data["size_mode"] = mode
        _write_file(data)
        log.info("runtime_settings updated: %s", data)
        return {"size_mode": data.get("size_mode", "code")}
