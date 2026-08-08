"""
image_turn_intent.py – Reasoning (LLM) ý định tin nhắn CÓ ẢNH.

Không dùng regex. Đọc TEXT câu hỏi (+ ngữ cảnh hội thoại ngắn) rồi quyết:
  - escalate  → khiếu nại / tra đơn / cần chủ shop (KHÔNG gửi ảnh finetune)
  - identify  → nhận diện / tư vấn SP theo ảnh → finetune / SigLIP
  - size      → tính size / số hạt kèm ảnh SP
  - other     → không rõ (mặc định xử lý như identify nếu có ảnh)

Supervisor dùng để lập plan; knowledge_base_agent dùng làm lưới an toàn
(tránh vẫn gọi FT khi plan lệch).
"""

from __future__ import annotations

import json
import re
from typing import Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from gemini import make_llm
from logger import get_logger

log = get_logger("image_turn_intent")

INTENTS = ("escalate", "identify", "size", "other")

_CLASSIFY_PROMPT = """
Bạn là bộ phân loại Ý ĐỊNH cho chatbot shop phong thủy Vạn An Group.

Khách vừa gửi tin nhắn CÓ KÈM ẢNH. Nhiệm vụ của bạn: CHỈ dựa vào LỜI NHẮN / câu hỏi
(và vài lượt hội thoại gần nếu có) để SUY LUẬN khách muốn gì — KHÔNG giả định
"có ảnh = phải nhận diện sản phẩm catalog".

Chọn ĐÚNG 1 nhãn intent:

1) escalate
   Khách cần CHỦ SHOP / CSKH xử lý tay, ảnh chỉ là BẰNG CHỨNG hoặc đính kèm:
   - Khiếu nại hàng: lỗi, vỡ, bể, sờn, đứt, thiếu, sai size/màu/mẫu, giao nhầm
   - Tra đơn / giao hàng chậm / phí ship / COD / đổi địa chỉ / "đơn tới đâu"
   - Yêu cầu đặc biệt shop làm tay: mix hạt theo số, sỉ, khắc tên, trì chú...
   → Hệ thống SẼ chuyển chủ shop; KHÔNG cần model nhận diện catalog.

2) identify
   Khách muốn BIẾT / TƯ VẤN SẢN PHẨM trong ảnh:
   - "Đây là vòng gì / đá gì", còn hàng không, giá bao nhiêu
   - Hợp mệnh không, gợi ý mua theo ảnh, so sánh mẫu
   → Cần gửi ảnh model nhận diện / tìm SP trong kho.

3) size
   Khách hỏi SIZE / SỐ HẠT / vừa cổ tay (cm, chiều cao, cân nặng) kèm ảnh SP.
   → Tính size; nhận diện SP chỉ khi cần gắn với mẫu cụ thể.

4) other
   Không đủ dấu hiệu cho 1–3 (vd chỉ gửi ảnh không chữ, hoặc mơ hồ).

QUY TẮC:
- Ưu tiên escalate nếu lời nhắn là khiếu nại / tra đơn dù có kèm ảnh SP.
- Chỉ chọn identify khi mục tiêu chính là tư vấn / định danh / mua SP trong ảnh.
- CHỈ trả về JSON một dòng, không markdown, không giải thích ngoài JSON:
  {"intent":"escalate|identify|size|other","reason":"một câu ngắn bằng tiếng Việt"}
"""


def _text_from_human(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                t = (p.get("text") or "").strip()
                if t:
                    parts.append(t)
        return " ".join(parts).strip()
    return str(content or "").strip()


def latest_user_text(messages: Sequence[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _text_from_human(m.content)
    return ""


def message_has_image(messages: Sequence[BaseMessage]) -> bool:
    for m in reversed(messages):
        if not isinstance(m, HumanMessage):
            continue
        c = m.content
        if isinstance(c, list):
            return any(
                isinstance(p, dict) and p.get("type") == "image_url" for p in c
            )
        return False
    return False


def _routing_snippet(messages: Sequence[BaseMessage], max_turns: int = 4) -> str:
    """Gom vài lượt user/assistant gần nhất (text only) cho LLM classify."""
    lines: list[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            t = _text_from_human(m.content)
            if t:
                lines.append(f"Khách: {t}")
        elif isinstance(m, AIMessage) and isinstance(m.content, str) and m.content.strip():
            lines.append(f"Shop: {m.content.strip()[:300]}")
    return "\n".join(lines[-max_turns * 2 :])


def classify_image_turn_intent(
    messages: Sequence[BaseMessage],
    default: str = "identify",
) -> dict:
    """LLM reasoning → {intent, reason}. Không regex.

    Nếu không có ảnh: intent=other (caller không nên gọi).
    Lỗi LLM / parse: fallback default (thường identify để không bỏ sót tư vấn SP).
    """
    if not message_has_image(messages):
        return {"intent": "other", "reason": "tin nhắn không kèm ảnh"}

    user_text = latest_user_text(messages)
    context = _routing_snippet(messages)
    user_block = (
        f"Lời nhắn mới nhất của khách (text, có kèm ảnh):\n"
        f"{user_text or '(không có chữ — chỉ gửi ảnh)'}\n\n"
        f"Ngữ cảnh hội thoại gần:\n{context or '(không có)'}"
    )

    try:
        llm = make_llm(temperature=0, max_tokens=256)
        resp = llm.invoke([
            SystemMessage(content=_CLASSIFY_PROMPT),
            HumanMessage(content=user_block),
        ])
        raw = (resp.content if isinstance(resp.content, str) else str(resp.content or "")).strip()
        # Lấy JSON object đầu tiên trong output (Gemini đôi khi bọc ```).
        m = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
        blob = m.group(0) if m else raw
        data = json.loads(blob)
        intent = str(data.get("intent") or default).strip().lower()
        if intent not in INTENTS:
            intent = default
        reason = str(data.get("reason") or "").strip()
        log.info("IMAGE_INTENT → %s | reason=%s | user='%s'",
                 intent, reason[:120], (user_text or "")[:80].replace("\n", " "))
        return {"intent": intent, "reason": reason, "raw": raw}
    except Exception as e:
        log.warning("IMAGE_INTENT classify lỗi (%s) → fallback %s", e, default)
        return {
            "intent": default,
            "reason": f"classify_error: {e}",
            "raw": "",
        }


def is_escalate_intent(intent: Optional[str]) -> bool:
    return (intent or "").strip().lower() == "escalate"
