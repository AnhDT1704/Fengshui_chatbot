"""
fengshui_finetune_client.py – Gọi model Qwen3 đã finetune TRI THỨC PHONG THỦY (Colab/ngrok).

Cấu hình trong .env (KB agent):

  # [1] MODEL ẢNH (VLM) — nhận diện SP từ ảnh → POST /predict
  FINETUNE_API_URL=https://xxxx.ngrok-free.app

  # [2] MODEL PHONG THỦY (Qwen3 text) — năm/can chi/cổ tay → POST /generate
  FENGSHUI_API_URL=https://yyyy.ngrok-free.app

File này chỉ dùng [2] FENGSHUI_API_URL. Model ảnh đọc FINETUNE_API_URL
trong knowledge_base_agent.py (finetune_identify).

API kỳ vọng (serve Colab phong thủy):
  GET / → {status, base_model, adapter}
  POST /generate → body {"prompt", "max_new_tokens"?, "temperature"?}
                     response linh hoạt: text / response / output / result / answer
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import requests

import fengshui_rules as fs_rules
from logger import get_logger

log = get_logger("fengshui_ft")

FENGSHUI_API_URL = os.getenv("FENGSHUI_API_URL", "").rstrip("/")
FENGSHUI_MODE = os.getenv("FENGSHUI_MODE", "code").strip().lower()
USE_FENGSHUI_FT = bool(FENGSHUI_API_URL) and FENGSHUI_MODE == "finetune"
FENGSHUI_TIMEOUT = float(os.getenv("FENGSHUI_TIMEOUT", "180"))
FENGSHUI_MAX_NEW = int(os.getenv("FENGSHUI_MAX_NEW_TOKENS", "640"))  # 2 chiều + CoT ngắn

# Khớp system message lúc train (rebuild_fengshui_menh_2dir — mệnh + size, CoT ngắn).
# Màu hợp/kỵ: CODE fengshui_rules (enrich sau khi có mệnh).
SYSTEM_MESSAGE = (
    "Bạn là chuyên gia phong thủy của shop Vạn An Group.\n"
    "Kỹ năng:\n"
    "A) MỆNH: năm sinh ↔ can chi / nạp âm / mệnh ngũ hành (Kim/Mộc/Thủy/Hỏa/Thổ).\n"
    "B) SIZE VÒNG: cổ tay (cm) ↔ size li (6/8/10) ↔ số hạt ↔ cung Sinh-Lão-Bệnh-Tử; "
    "chiều dài = số_hạt × (li/10) cm.\n"
    "Màu hợp/kỵ KHÔNG nằm trong nhiệm vụ này.\n"
    "CHỈ trả về JSON; nếu có suy luận thì <think>…</think> NGẮN rồi JSON."
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
        # Bổ sung màu hợp/kỵ bằng CODE ngay sau khi có mệnh từ model
        if isinstance(data, dict) and "raw" not in data:
            data = fs_rules.enrich_menh_payload(data)
        # Dòng TIMING riêng — grep: TIMING.*FINETUNE_FENGSHUI
        log.info(
            "[TIMING] FINETUNE_FENGSHUI | status=ok | latency_s=%.3f | max_new_tokens=%s | "
            "url=%s | q=%s | element=%s | task=%s | colors=%s",
            dt,
            payload.get("max_new_tokens"),
            FENGSHUI_API_URL,
            user_question[:120],
            data.get("element") or data.get("personal_element") or "",
            data.get("task") or "",
            (data.get("lucky_colors") or [])[:6] if isinstance(data, dict) else "",
        )
        # Log ĐẦY ĐỦ để debug: prompt → think → JSON (đây là nguồn chatbot dùng)
        log.info(
            "FENGSHUI FT MODEL OUTPUT %.1fs\n"
            "Q: %s\n"
            "URL: %s\n"
            " <think> (%d chars) \n%s\n"
            "JSON parsed \n%s\n"
            "RAW full (%d chars) \n%s\n"
            "",
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
    """Năm → mệnh (model) + màu hợp/kỵ (code enrich trong call_fengshui_generate)."""
    return call_fengshui_generate(f"Tôi sinh năm {birth_year}, mệnh gì vậy?")


def ask_menh_to_years(element: str) -> dict:
    """Mệnh → năm: ưu tiên MODEL FT; fallback CODE (chu kỳ 60) nếu FT lỗi."""
    e = (element or "").replace("Mệnh", "").replace("mệnh", "").strip()
    prompt = f"Mệnh {e} là những năm nào?"

    def _code_fallback(extra: Optional[dict] = None) -> dict:
        coded = fs_rules.years_for_element(e)
        if coded.get("error"):
            out = {"ok": False, "error": coded.get("error"), "latency_s": 0.0, "source": "code"}
            if extra:
                out.update(extra)
            return out
        data = fs_rules.enrich_menh_payload({
            "task": "menh", "direction": "menh_to_years", **coded,
        })
        out = {
            "ok": True,
            "data": data,
            "think": "",
            "raw": "",
            "latency_s": 0.0,
            "source": "code_fallback" if extra else "code",
        }
        if extra:
            out.update(extra)
        return out

    if USE_FENGSHUI_FT:
        ft = call_fengshui_generate(prompt)
        if ft.get("ok") and isinstance(ft.get("data"), dict):
            data = dict(ft["data"])
            # Chuẩn hoá direction/element nếu model thiếu
            data.setdefault("task", "menh")
            data.setdefault("direction", "menh_to_years")
            if not data.get("element"):
                data["element"] = e
            years = data.get("years_in_cycle") or data.get("years")
            if isinstance(years, list) and len(years) >= 8:
                try:
                    data = fs_rules.enrich_menh_payload(data)
                except Exception:
                    pass
                # FT thường cắt example_years_modern (vd chỉ tới 2017) → refresh CODE gần năm hiện tại
                try:
                    data = fs_rules.refresh_example_years_modern(data)
                except Exception:
                    pass
                return {
                    "ok": True,
                    "data": data,
                    "think": ft.get("think") or "",
                    "raw": ft.get("raw") or "",
                    "latency_s": ft.get("latency_s") or 0.0,
                    "source": "fengshui_finetune",
                }
            log.warning(
                "FENGSHUI FT menh_to_years thiếu years_in_cycle → code fallback | element=%s",
                e,
            )
            return _code_fallback({
                "ft_error": "missing_years_in_cycle",
                "ft_think": ft.get("think") or "",
                "ft_raw": (ft.get("raw") or "")[:2000],
            })
        log.warning(
            "FENGSHUI FT menh_to_years lỗi (%s) → code fallback | element=%s",
            ft.get("error"), e,
        )
        return _code_fallback({"ft_error": ft.get("error")})

    return _code_fallback()


def ask_size(wrist_cm: float, li: Optional[int] = None) -> dict:
    """Cổ tay → size/số hạt qua FT (admin bật size_mode=finetune). Mặc định shop dùng CODE."""
    if li is None:
        q = f"Cổ tay tôi {wrist_cm}cm thì đeo vòng bao nhiêu hạt?"
    else:
        q = f"Cổ tay {wrist_cm}cm mà mình thích hạt {li} li thì bao nhiêu hạt?"
    return call_fengshui_generate(q)
