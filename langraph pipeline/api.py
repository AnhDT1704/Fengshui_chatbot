"""
api.py – FastAPI server exposing the chatbot.

Endpoints:
    POST /chat              { session_id, message }                 → text chat
    POST /chat/image        { session_id, message, image_b64, mime} → multimodal chat
    GET  /history/{sid}                                              → conversation log
    GET  /escalations                                                → pending tickets
    GET  /health
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

from logger import get_logger
from models import get_engine
import graph as chat_graph

load_dotenv()
log = get_logger("api")

_STATIC_DIR = Path(__file__).parent / "static"


def _check_required_env():
    """Fail fast if mandatory env vars are missing — clearer than a cryptic
    google.auth DefaultCredentialsError at first request."""
    from gemini import available_keys_count
    n_keys = available_keys_count()
    if n_keys == 0:
        log.error(
            "No Gemini API key found. Add GOOGLE_API_KEY1 (and optional "
            "GOOGLE_API_KEY2..N for fallback) to .env "
            "(free key at https://aistudio.google.com/apikey), then restart."
        )
        raise SystemExit(1)
    log.info("Gemini keys available: %d (will rotate on quota errors)", n_keys)
    if not os.getenv("SERPAPI_KEY", "").strip():
        log.warning("SERPAPI_KEY is empty — web_search_tool will return a graceful fallback.")


_check_required_env()


# ═══════════════════════════════════════════════════════════════════
#  SCHEMAS
# ═══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message:    str


class ChatImageRequest(BaseModel):
    session_id: str = Field(default="default")
    message:    str = ""
    image_b64:  str
    mime:       str = "image/jpeg"


class ChatResponse(BaseModel):
    response:     str
    agent_used:   str
    intent:       str
    tools_called: list[str]


# ═══════════════════════════════════════════════════════════════════
#  APP
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(title="Vạn An Group – Fengshui Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static (UI) ────────────────────────────────────────────────────
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def root():
    """Serve the chat UI."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "UI not built. POST /chat directly."}


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("CHATBOT_MODEL", "gemini-2.5-flash")}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    log.info("HTTP /chat        session=%s  msg='%s'",
             req.session_id, req.message[:80].replace("\n", " "))
    out = chat_graph.chat(req.message, session_id=req.session_id)
    return ChatResponse(
        response=out["response"],
        agent_used=out["agent_used"],
        intent=out["intent"],
        tools_called=out["tools_called"],
    )


@app.post("/chat/image", response_model=ChatResponse)
def chat_image_endpoint(req: ChatImageRequest):
    if not req.image_b64:
        raise HTTPException(400, "image_b64 must not be empty")
    log.info("HTTP /chat/image  session=%s  msg='%s'  img_bytes≈%d",
             req.session_id, (req.message or "")[:80], len(req.image_b64))
    out = chat_graph.chat_with_image(
        user_message = req.message,
        image_base64 = req.image_b64,
        image_mime   = req.mime,
        session_id   = req.session_id,
    )
    return ChatResponse(
        response=out["response"],
        agent_used=out["agent_used"],
        intent=out["intent"],
        tools_called=out["tools_called"],
    )


@app.get("/history/{session_id}")
def history_endpoint(session_id: str, limit: int = 50):
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content, agent_used, intent, tools_called, created_at
                FROM conversation_log
                WHERE session_id = :sid
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            {"sid": session_id, "lim": limit},
        ).fetchall()
    return {
        "session_id": session_id,
        "turns": [
            {
                "role":         r[0],
                "content":      r[1],
                "agent_used":   r[2],
                "intent":       r[3],
                "tools_called": r[4],
                "created_at":   str(r[5]),
            }
            for r in reversed(rows)
        ],
    }


@app.get("/escalations")
def escalations_endpoint(status: Optional[str] = "pending", limit: int = 50):
    engine = get_engine()
    with engine.begin() as conn:
        if status:
            rows = conn.execute(
                text("""
                    SELECT id, session_id, reason, user_summary, status, created_at
                    FROM escalation_queue
                    WHERE status = :status
                    ORDER BY created_at DESC
                    LIMIT :lim
                """),
                {"status": status, "lim": limit},
            ).fetchall()
        else:
            rows = conn.execute(
                text("""
                    SELECT id, session_id, reason, user_summary, status, created_at
                    FROM escalation_queue
                    ORDER BY created_at DESC
                    LIMIT :lim
                """),
                {"lim": limit},
            ).fetchall()
    return {
        "items": [
            {
                "id":           r[0],
                "session_id":   r[1],
                "reason":       r[2],
                "user_summary": r[3],
                "status":       r[4],
                "created_at":   str(r[5]),
            }
            for r in rows
        ],
    }


# Allow `python api.py` for quick start
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
