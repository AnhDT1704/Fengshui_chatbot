"""
memory.py – Persistence layer for the chatbot.

Provides:
  - log_turn(session_id, role, content, ...)           : append to conversation_log
  - load_recent_history(session_id, limit=20)          : LangChain messages
  - create_escalation(session_id, reason, summary, ...): write to escalation_queue
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import json
from typing import Optional

from sqlalchemy import text
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from logger import get_logger
from models import get_engine

log = get_logger("memory")

DEFAULT_TITLE = "Cuộc trò chuyện mới"


def log_turn(
    session_id: str,
    role: str,                    # "user" | "assistant"
    content: str,
    agent_used: Optional[str] = None,
    intent: Optional[str] = None,
    tools_called: Optional[list[str]] = None,
    user_id: Optional[int] = None,
    images: Optional[list[str]] = None,   # URL ảnh đã lưu trên server (cho lượt có ảnh)
    product_ref: Optional[list[dict]] = None,  # [{"id","name"}] SP lượt này xác định/trình bày
) -> None:
    """Append one turn to conversation_log."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO conversation_log
                  (session_id, role, content, agent_used, intent, tools_called, user_id, images, product_ref)
                VALUES
                  (:sid, :role, :content, :agent, :intent, :tools, :uid,
                   CAST(:images AS JSONB), CAST(:pref AS JSONB))
            """),
            {
                "sid":     session_id,
                "role":    role,
                "content": content,
                "agent":   agent_used,
                "intent":  intent,
                "tools":   tools_called or [],
                "uid":     user_id,
                "images":  json.dumps(images) if images else None,
                "pref":    json.dumps(product_ref) if product_ref else None,
            },
        )


# ═══════════════════════════════════════════════════════════════════
#  CHAT SESSIONS (theo user) — thay cho localStorage phía client
# ═══════════════════════════════════════════════════════════════════

def ensure_session(session_id: str, user_id: int, title: Optional[str] = None) -> bool:
    """Tạo phiên nếu chưa có (gắn user). Trả True nếu phiên thuộc về user này
    (mới tạo hoặc đã tồn tại & đúng chủ), False nếu phiên của user KHÁC."""
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT user_id FROM chat_sessions WHERE id = :id"), {"id": session_id}
        ).first()
        if row is None:
            conn.execute(
                text("INSERT INTO chat_sessions (id, user_id, title) VALUES (:id, :u, :t)"),
                {"id": session_id, "u": user_id, "t": (title or DEFAULT_TITLE)[:120]},
            )
            return True
        return row[0] == user_id


def list_sessions(user_id: int) -> list[dict]:
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, title, updated_at, status
                FROM chat_sessions
                WHERE user_id = :u
                ORDER BY updated_at DESC
            """),
            {"u": user_id},
        ).fetchall()
    return [{"id": r[0], "title": r[1], "updatedAt": str(r[2]), "status": r[3]} for r in rows]


# ── Trạng thái phiên: bot | pending_admin | admin (chuyển cho chủ shop) ──────────

def get_session_status(session_id: str) -> str:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT status FROM chat_sessions WHERE id = :id"), {"id": session_id}
        ).first()
    return row[0] if row else "bot"


def set_session_status(session_id: str, status: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE chat_sessions SET status = :s, updated_at = now() WHERE id = :id"),
            {"s": status, "id": session_id},
        )


def list_all_users_with_sessions() -> list[dict]:
    """DÀNH CHO ADMIN: MỌI user + toàn bộ phiên của họ (gồm cả phiên bot tự trả lời).
    User nào có phiên đang chuyển cho shop (status != 'bot') → has_handoff=True, xếp lên đầu."""
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT u.id, u.username, s.id, s.title, s.status, s.updated_at
                FROM users u
                JOIN chat_sessions s ON s.user_id = u.id
                ORDER BY u.id, s.updated_at DESC
            """)
        ).fetchall()
    users: dict[int, dict] = {}
    for uid, uname, sid, title, status, upd in rows:
        u = users.setdefault(uid, {
            "user_id": uid, "username": uname, "sessions": [],
            "has_handoff": False, "handoff_count": 0, "updatedAt": "",
        })
        u["sessions"].append({"id": sid, "title": title, "status": status, "updatedAt": str(upd)})
        if status != "bot":
            u["has_handoff"] = True
            u["handoff_count"] += 1
        if str(upd) > u["updatedAt"]:
            u["updatedAt"] = str(upd)
    result = list(users.values())
    # Ưu tiên user có phiên cần shop trả lời, rồi tới hoạt động gần nhất.
    result.sort(key=lambda x: (x["has_handoff"], x["updatedAt"]), reverse=True)
    return result


def list_handoff_sessions() -> list[dict]:
    """DÀNH CHO ADMIN: tất cả phiên đã chuyển cho chủ shop (status != 'bot'),
    kèm tên khách + tin nhắn mới nhất, xếp phiên mới cập nhật lên đầu."""
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT s.id, s.title, s.status, s.updated_at, u.username, u.id,
                       (SELECT content FROM conversation_log c
                        WHERE c.session_id = s.id
                        ORDER BY c.created_at DESC LIMIT 1) AS last_msg
                FROM chat_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.status <> 'bot'
                ORDER BY s.updated_at DESC
            """)
        ).fetchall()
    return [
        {
            "id": r[0], "title": r[1], "status": r[2], "updatedAt": str(r[3]),
            "customer": r[4], "user_id": r[5], "last_msg": r[6] or "",
        }
        for r in rows
    ]


def touch_session(session_id: str, user_id: int, title_text: Optional[str] = None) -> None:
    """Cập nhật updated_at; nếu phiên còn tiêu đề mặc định thì đặt theo tin đầu."""
    engine = get_engine()
    with engine.begin() as conn:
        if title_text:
            conn.execute(
                text("""
                    UPDATE chat_sessions
                    SET updated_at = now(),
                        title = CASE WHEN title = :default OR title IS NULL OR title = ''
                                     THEN :title ELSE title END
                    WHERE id = :id AND user_id = :u
                """),
                {"id": session_id, "u": user_id, "default": DEFAULT_TITLE,
                 "title": title_text.strip()[:120]},
            )
        else:
            conn.execute(
                text("UPDATE chat_sessions SET updated_at = now() WHERE id = :id AND user_id = :u"),
                {"id": session_id, "u": user_id},
            )


def delete_session(session_id: str, user_id: int) -> int:
    """Xoá phiên + lịch sử của phiên (chỉ khi đúng chủ). Trả số dòng log đã xoá."""
    engine = get_engine()
    with engine.begin() as conn:
        owned = conn.execute(
            text("SELECT 1 FROM chat_sessions WHERE id = :id AND user_id = :u"),
            {"id": session_id, "u": user_id},
        ).first()
        if not owned:
            return 0
        deleted = conn.execute(
            text("DELETE FROM conversation_log WHERE session_id = :id"), {"id": session_id}
        ).rowcount or 0
        conn.execute(
            text("DELETE FROM chat_sessions WHERE id = :id AND user_id = :u"),
            {"id": session_id, "u": user_id},
        )
    return deleted


def session_owner(session_id: str) -> Optional[int]:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT user_id FROM chat_sessions WHERE id = :id"), {"id": session_id}
        ).first()
    return row[0] if row else None


def get_session_product_context(session_id: str, recent_limit: int = 6) -> dict:
    """Ngữ cảnh sản phẩm của CẢ PHIÊN (không phụ thuộc cửa sổ nhớ):
      - primary: sản phẩm của lượt GẦN NHẤT có product_ref → "sản phẩm này / nó" trỏ về đây
      - recent:  danh sách SP đã nhắc trong phiên (dedupe theo id, mới nhất trước) → khớp khi
                 khách gọi đích danh tên SP cũ.
    """
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT product_ref FROM conversation_log
                WHERE session_id = :sid AND product_ref IS NOT NULL
                ORDER BY created_at DESC
            """),
            {"sid": session_id},
        ).fetchall()

    primary: list[dict] = list(rows[0][0]) if rows and rows[0][0] else []
    recent: list[dict] = []
    seen: set = set()
    for (pref,) in rows:
        for p in (pref or []):
            pid = p.get("id")
            if pid is not None and pid not in seen:
                seen.add(pid)
                recent.append({"id": pid, "name": p.get("name")})
                if len(recent) >= recent_limit:
                    break
        if len(recent) >= recent_limit:
            break
    return {"primary": primary, "recent": recent}


def _product_context_note(session_id: str) -> Optional[BaseMessage]:
    ctx = get_session_product_context(session_id)
    primary, recent = ctx["primary"], ctx["recent"]
    if not primary and not recent:
        return None
    parts = []
    if primary:
        prim = ", ".join(
            f'{p.get("name")} (product_id={p.get("id")})'
            for p in primary if p.get("id") is not None
        )
        if prim:
            parts.append(f"Khi khách nói 'sản phẩm này / nó / cái này / mẫu đó' (không nêu "
                         f"tên mới) → hiểu là: {prim}.")
    prim_ids = {p.get("id") for p in primary}
    others = [p for p in recent if p["id"] not in prim_ids]
    if others:
        oth = ", ".join(f'{p["name"]} (product_id={p["id"]})' for p in others)
        parts.append(f"Sản phẩm khác đã nhắc trong phiên (khớp theo TÊN nếu khách gọi đích "
                     f"danh): {oth}.")
    if not parts:
        return None
    return HumanMessage(content=(
        "[NGỮ CẢNH SẢN PHẨM — không phải lời khách] " + " ".join(parts) +
        " DÙNG ĐÚNG product_id để tra cứu (get_product_detail_tool/product_care_tool), "
        "TUYỆT ĐỐI không đoán id."
    ))


def load_recent_history(session_id: str, limit: int = 20) -> list[BaseMessage]:
    """Load the last N turns (chronological) as LangChain messages.

    Kèm 1 dòng NGỮ CẢNH SẢN PHẨM (lấy theo CẢ PHIÊN, không giới hạn theo cửa sổ nhớ)
    để lượt sau biết 'sản phẩm này' là product_id nào — kể cả khi SP được nhắc đã lâu."""
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content
                FROM conversation_log
                WHERE session_id = :sid
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            {"sid": session_id, "lim": limit},
        ).fetchall()

    rows = list(reversed(rows))
    messages: list[BaseMessage] = []
    for role, content in rows:
        messages.append(HumanMessage(content=content) if role == "user" else AIMessage(content=content))

    note = _product_context_note(session_id)
    if note is not None:
        messages.append(note)

    log.debug("loaded %d past turns for session=%s (product_ctx=%s)",
              len(messages), session_id, note is not None)
    return messages


def create_escalation(
    session_id: str,
    reason: str,
    user_summary: str,
    full_context: str = "",
) -> int:
    """Insert an escalation row and return its id."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                INSERT INTO escalation_queue
                  (session_id, reason, user_summary, full_context)
                VALUES
                  (:sid, :reason, :summary, :context)
                RETURNING id
            """),
            {
                "sid":     session_id,
                "reason":  reason,
                "summary": user_summary,
                "context": full_context,
            },
        )
        ticket_id = result.scalar_one()
        log.warning("ESCALATION #%d session=%s reason=%s summary=%r",
                    ticket_id, session_id, reason, (user_summary or "")[:80])
        return ticket_id
