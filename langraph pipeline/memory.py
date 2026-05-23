"""
memory.py – Persistence layer for the chatbot.

Provides:
  - log_turn(session_id, role, content, ...)           : append to conversation_log
  - load_recent_history(session_id, limit=20)          : LangChain messages
  - create_escalation(session_id, reason, summary, ...): write to escalation_queue
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from typing import Optional

from sqlalchemy import text
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from logger import get_logger
from models import get_engine

log = get_logger("memory")


def log_turn(
    session_id: str,
    role: str,                    # "user" | "assistant"
    content: str,
    agent_used: Optional[str] = None,
    intent: Optional[str] = None,
    tools_called: Optional[list[str]] = None,
) -> None:
    """Append one turn to conversation_log."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO conversation_log
                  (session_id, role, content, agent_used, intent, tools_called)
                VALUES
                  (:sid, :role, :content, :agent, :intent, :tools)
            """),
            {
                "sid":     session_id,
                "role":    role,
                "content": content,
                "agent":   agent_used,
                "intent":  intent,
                "tools":   tools_called or [],
            },
        )


def load_recent_history(session_id: str, limit: int = 20) -> list[BaseMessage]:
    """Load the last N turns (in chronological order) as LangChain messages."""
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
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    log.debug("loaded %d past turns for session=%s", len(messages), session_id)
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
