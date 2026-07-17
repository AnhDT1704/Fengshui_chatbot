"""
fengshui_finetune_client.py – Gọi model Qwen3 đã finetune TRI THỨC PHONG THỦY (Colab/ngrok).

Cấu hình trong .env (KB agent):

  # [1] MODEL ẢNH (VLM) — nhận diện SP từ ảnh  → POST /predict
  FINETUNE_API_URL=https://xxxx.ngrok-free.app

  # [2] MODEL PHONG THỦY (Qwen3 text) — năm/can chi/cổ tay → POST /generate
  FENGSHUI_API_URL=https://yyyy.ngrok-free.app

File này chỉ dùng [2] FENGSHUI_API_URL. Model ảnh đọc FINETUNE_API_URL
trong knowledge_base_agent.py (finetune_identify).

API kỳ vọng (serve Colab phong thủy):
  GET  /            → {status, base_model, adapter}
  POST /generate    → body {"prompt", "max_new_tokens"?, "temperature"?}
                     response linh hoạt: text / response / output / result / answer
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import requests

from logger import get_logger

log = get_logger("fengshui_ft")

FENGSHUI_API_URL = os.getenv("FENGSHUI_API_URL", "").rstrip("/")
USE_FENGSHUI_FT = bool(FENGSHUI_API_URL)
FENGSHUI_TIMEOUT = float(os.getenv("FENGSHUI_TIMEOUT", "180"))
FENGSHUI_MAX_NEW = int(os.getenv("FENGSHUI_MAX_NEW_TOKENS", "900"))

# Khớp system message lúc train (dataset_fengshui / dataset_fengshui_cot).
SYSTEM_MESSAGE = (
    "Bạn là chuyên gia phong thủy của shop Vạn An Group. Trả lời các câu hỏi:\n"
    "1) NĂM SINH → tính Can Chi, Nạp âm, Mệnh ngũ hành, màu hợp và màu kỵ.\n"
    "2) CAN CHI đầy đủ (vd 'Canh Ngọ') → tính Nạp âm, Mệnh, màu hợp / kỵ.\n"
    "3) CHỈ CÓ CON GIÁP (vd 'tuổi Ngọ') → KHÔNG đủ dữ kiện: cùng một con giáp có 5 mệnh "
    "khác nhau theo chu kỳ 60 năm → phải HỎI LẠI năm sinh, TUYỆT ĐỐI không đoán mệnh.\n"
    "4) CỔ TAY (cm) → tính size hạt (li), số hạt, chiều dài vòng, cung Sinh-Lão-Bệnh-Tử.\n"
    "CHỈ trả về JSON, không giải thích thêm."
)


def extract_think(txt: str) -> str:
    """Lấy nội dung trong <think>...</think> (VARIANT B / dataset_fengshui_cot)."""
    if not txt:
        return ""
    m = re.search(r"<think>(.*?)</think>", txt, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    # Một số serve bỏ tag đóng hoặc dùng biến thể
    m2 = re.search(r"<think>\s*(.*)", txt, flags=re.S | re.I)
    if m2 and "{" in m2.group(1):
        # cắt trước JSON
        body = m2.group(1)
        cut = body.find("{")
        return body[:cut].strip() if cut > 0 else body.strip()
    return ""


def _strip_think(txt: str) -> str:
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S | re.I)


def parse_json_blob(txt: str) -> dict:
    """Lấy object JSON đầu tiên trong text model (sau khi bỏ <think>)."""
    if not txt:
        return {}
    cleaned = _strip_think(txt).strip()
    s, e = cleaned.find("{"), cleaned.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {"raw": txt}
    try:
        return json.loads(cleaned[s : e + 1])
    except Exception:
        return {"raw": txt}


def _extract_text_from_response(data: Any) -> str:
    """Chuẩn hoá nhiều shape response từ serve Colab khác nhau."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return str(data)

    # answer đã là dict JSON model
    ans = data.get("answer")
    if isinstance(ans, dict):
        return json.dumps(ans, ensure_ascii=False)
    if isinstance(ans, str):
        return ans

    for key in ("text", "response", "output", "result", "generated_text", "raw", "content"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val
        if isinstance(val, dict):
            # result: { ... fields ... } hoặc {raw: "..."}
            if "raw" in val and isinstance(val["raw"], str):
                return val["raw"]
            return json.dumps(val, ensure_ascii=False)

    # Toàn bộ dict đã là JSON task menh/size
    if data.get("task") in ("menh", "size") or "element" in data or "bead_count" in data:
        return json.dumps(data, ensure_ascii=False)

    return json.dumps(data, ensure_ascii=False)


def build_prompt(user_question: str) -> str:
    """Chỉ gửi CÂU HỎI khách.

    Serve Colab PHẢI tự gắn SYSTEM_MESSAGE + apply_chat_template (giống lúc train).
    Nếu client nhét system thô vào `prompt`, model Qwen3 dễ sinh rác <tool_call>.
    """
    return (user_question or "").strip()


def call_fengshui_generate(
    user_question: str,
    *,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> dict:
    """
    Gọi POST {FENGSHUI_API_URL}/generate.

    Trả về:
      {
        "ok": True,
        "data": <dict JSON model, đã bỏ think>,
        "think": <str chuỗi suy luận trong <think> nếu có>,
        "raw": <str full text model trả>,
        "latency_s": float,
      }
      {"ok": False, "error": str, "latency_s": float}

    Lưu ý: transformers từ chối temperature=0.0 (ValueError). Serve Colab nên
    dùng do_sample=False cho greedy; client gửi temperature>0 để an toàn nếu
    server vẫn truyền temperature vào generate().
    """
    if not USE_FENGSHUI_FT:
        return {"ok": False, "error": "FENGSHUI_API_URL chưa cấu hình", "latency_s": 0.0}

    # Mặc định > 0 — tránh crash serve khi generate(..., temperature=0.0).
    if temperature is None:
        temperature = float(os.getenv("FENGSHUI_TEMPERATURE", "0.01"))
    if temperature <= 0:
        temperature = 0.01

    prompt = build_prompt(user_question)
    url = f"{FENGSHUI_API_URL}/generate"
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens or FENGSHUI_MAX_NEW,
        "temperature": temperature,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
            },
            timeout=FENGSHUI_TIMEOUT,
        )
        dt = time.perf_counter() - t0
        resp.raise_for_status()
        body = resp.json()
        text = _extract_text_from_response(body)
        think = extract_think(text)
        data = parse_json_blob(text)
        # Dòng TIMING riêng — grep: TIMING.*FINETUNE_FENGSHUI
        log.info(
            "[TIMING] FINETUNE_FENGSHUI | status=ok | latency_s=%.3f | max_new_tokens=%s | "
            "url=%s | q=%s | element=%s | task=%s",
            dt,
            payload.get("max_new_tokens"),
            FENGSHUI_API_URL,
            user_question[:120],
            data.get("element") or data.get("personal_element") or "",
            data.get("task") or "",
        )
        # Log ĐẦY ĐỦ để debug: prompt → think → JSON (đây là nguồn chatbot dùng)
        log.info(
            "┌─ FENGSHUI FT MODEL OUTPUT ─ ⏱ %.1fs\n"
            "│ Q: %s\n"
            "│ URL: %s\n"
            "├─ <think> (%d chars) ─\n%s\n"
            "├─ JSON parsed ─\n%s\n"
            "├─ RAW full (%d chars) ─\n%s\n"
            "└─────────────",
            dt,
            user_question[:200],
            url,
            len(think),
            think if think else "(không có khối <think>)",
            json.dumps(data, ensure_ascii=False, indent=2),
            len(text or ""),
            (text or "")[:8000],
        )
        return {
            "ok": True,
            "data": data,
            "think": think,
            "raw": text,
            "latency_s": dt,
        }
    except Exception as ex:
        dt = time.perf_counter() - t0
        log.warning(
            "[TIMING] FINETUNE_FENGSHUI | status=error | latency_s=%.3f | url=%s | q=%s | error=%s",
            dt, FENGSHUI_API_URL, user_question[:120], ex,
        )
        log.warning("FENGSHUI FT lỗi sau %.1fs: %s (url=%s)", dt, ex, url)
        return {"ok": False, "error": str(ex), "latency_s": dt}


def ask_menh_by_year(birth_year: int) -> dict:
    return call_fengshui_generate(f"Tôi sinh năm {birth_year}, mệnh gì vậy?")


def ask_size(wrist_cm: float, li: Optional[int] = None) -> dict:
    if li is None:
        q = f"Cổ tay tôi {wrist_cm}cm thì đeo vòng bao nhiêu hạt?"
    else:
        q = f"Cổ tay {wrist_cm}cm mà mình thích hạt {li} li thì bao nhiêu hạt?"
    return call_fengshui_generate(q)
