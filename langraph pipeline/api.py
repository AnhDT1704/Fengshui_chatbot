"""
api.py – FastAPI server exposing the chatbot.

Endpoints:
    POST /chat { session_id, message } → text chat
    POST /chat/image { session_id, message, images[]|image_b64, mime} → multimodal chat (≤5 ảnh)
    GET /history/{sid} → conversation log
    DELETE /history/{sid} → xoá lịch sử phiên
    GET /escalations → pending tickets
    GET /health
"""

from __future__ import annotations

import _bootstrap # noqa: F401

import asyncio
import base64
import io
import json
import os
import queue
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import text

import admin_import
import auth
import knowledge_base_agent
import memory
import progress
from logger import get_logger
from models import get_engine
import graph as chat_graph

load_dotenv()
log = get_logger("api")

_STATIC_DIR = Path(__file__).parent / "static"
# Ảnh khách gửi lưu ở /app/uploads (repo root, bind-mount .:/app → còn trên host).
_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_EXT_BY_MIME = {
    "image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png",
    "image/webp": "webp", "image/gif": "gif",
}


def _save_images(images: list[dict]) -> list[str]:
    """Lưu ảnh (base64) ra ổ đĩa, trả danh sách URL /uploads/<file> để hiển thị lại."""
    urls: list[str] = []
    for im in images:
        b64 = im.get("base64") or ""
        if not b64:
            continue
        ext = _EXT_BY_MIME.get((im.get("mime") or "").lower(), "jpg")
        name = f"{uuid.uuid4().hex}.{ext}"
        try:
            (_UPLOAD_DIR / name).write_bytes(base64.b64decode(b64))
            urls.append(f"/uploads/{name}")
        except Exception as e:
            log.warning("Lưu ảnh thất bại: %s", e)
    return urls


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


class ChatRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str


class ImageItem(BaseModel):
    image_b64: str
    mime: str = "image/jpeg"


class ChatImageRequest(BaseModel):
    session_id: str = Field(default="default")
    message: str = ""
    # Nhiều ảnh (tối đa MAX_IMAGES). Hai field single bên dưới giữ để tương thích
    # ngược với client cũ chỉ gửi 1 ảnh.
    images: list[ImageItem] = Field(default_factory=list)
    image_b64: Optional[str] = None
    mime: str = "image/jpeg"
    # Nút gạt trên UI. False (mặc định) = model DỪNG SỚM, bỏ product_description →
    # ~8s/ảnh thay vì ~90-140s. Mô tả vẫn lấy từ Postgres nên câu trả lời không đổi.
    finetune_full: bool = False


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    intent: str
    tools_called: list[str]
    session_status: str = "bot" # bot | pending_admin | admin


class AuthRequest(BaseModel):
    username: str
    password: str


class AdminReply(BaseModel):
    session_id: str
    message: str


def _user_out(user: dict) -> dict:
    """Bổ sung cờ is_admin cho user trả về client."""
    return {**user, "is_admin": auth.is_admin(user.get("username"))}


class SessionCreate(BaseModel):
    session_id: Optional[str] = None
    title: Optional[str] = None


app = FastAPI(title="Vạn An Group – Fengshui Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.mount("/uploads", StaticFiles(directory=str(_UPLOAD_DIR)), name="uploads")


def current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """Lấy user từ header 'Authorization: Bearer <token>'. 401 nếu thiếu/sai."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    user = auth.user_for_token(token) if token else None
    if not user:
        raise HTTPException(401, "Cần đăng nhập")
    return user


def current_admin(user: dict = Depends(current_user)) -> dict:
    """Chỉ cho phép chủ shop (admin). 403 nếu không phải."""
    if not auth.is_admin(user.get("username")):
        raise HTTPException(403, "Chỉ chủ shop mới truy cập được")
    return user


# Tải + load model SigLIP 2 NGAY khi boot (chạy nền, không chặn startup), để
# khách gửi ảnh ĐẦU TIÊN không phải chờ chi phí một-lần (tải ~1.4GB + load).
@app.on_event("startup")
def _warmup_image_model():
    import threading

    def _warm():
        try:
            import image_embedding
            log.info("Warmup model ảnh (%s)...", image_embedding._PRIMARY_MODEL)
            image_embedding.model_id() # trigger download + load (đã cache sau đó)
            log.info("Model ảnh sẵn sàng cho visual search.")
        except Exception as e:
            log.warning("Warmup model ảnh thất bại (sẽ load khi có ảnh): %s", e)

    threading.Thread(target=_warm, name="img-warmup", daemon=True).start()


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


@app.post("/auth/register")
def register_endpoint(req: AuthRequest):
    try:
        user = auth.register_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    token = auth.create_token(user["id"])
    return {"token": token, "user": _user_out(user)}


@app.post("/auth/login")
def login_endpoint(req: AuthRequest):
    user = auth.login(req.username, req.password)
    if not user:
        raise HTTPException(401, "Sai tên đăng nhập hoặc mật khẩu")
    token = auth.create_token(user["id"])
    return {"token": token, "user": _user_out(user)}


@app.post("/auth/logout")
def logout_endpoint(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        auth.revoke_token(authorization[7:].strip())
    return {"ok": True}


@app.get("/auth/me")
def me_endpoint(user: dict = Depends(current_user)):
    return {"user": _user_out(user)}


@app.get("/sessions")
def list_sessions_endpoint(user: dict = Depends(current_user)):
    return {"sessions": memory.list_sessions(user["id"])}


@app.post("/sessions")
def create_session_endpoint(req: SessionCreate, user: dict = Depends(current_user)):
    sid = (req.session_id or "").strip() or ("web-" + uuid.uuid4().hex[:10])
    if not memory.ensure_session(sid, user["id"], title=req.title):
        raise HTTPException(403, "Phiên này thuộc về tài khoản khác")
    return {"session_id": sid}


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str, user: dict = Depends(current_user)):
    deleted = memory.delete_session(session_id, user["id"])
    return {"session_id": session_id, "deleted": deleted}


@app.get("/admin/handoffs")
def admin_handoffs_endpoint(admin: dict = Depends(current_admin)):
    """Danh sách MỌI phiên đã chuyển cho chủ shop (pending_admin / admin)."""
    return {"sessions": memory.list_handoff_sessions()}


@app.get("/admin/users")
def admin_users_endpoint(admin: dict = Depends(current_admin)):
    """MỌI khách + toàn bộ phiên của họ (trừ tài khoản admin), để chủ shop quản lý."""
    users = [u for u in memory.list_all_users_with_sessions()
             if not auth.is_admin(u["username"])]
    return {"users": users}


@app.post("/admin/reply", response_model=ChatResponse)
def admin_reply_endpoint(req: AdminReply, admin: dict = Depends(current_admin)):
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    owner = memory.session_owner(req.session_id)
    if owner is None:
        raise HTTPException(404, "Không tìm thấy phiên")
    # Lưu lời chủ shop như một lượt assistant (agent_used='admin') + khoá phiên ở 'admin'.
    memory.log_turn(req.session_id, "assistant", req.message,
                    agent_used="admin", intent="admin", user_id=owner)
    memory.set_session_status(req.session_id, "admin")
    log.info("HTTP /admin/reply admin=%s session=%s", admin["username"], req.session_id)
    return ChatResponse(response=req.message, agent_used="admin", intent="admin",
                        tools_called=[], session_status="admin")


@app.post("/admin/sessions/{session_id}/return-to-bot")
def admin_return_to_bot_endpoint(session_id: str, admin: dict = Depends(current_admin)):
    """Trả phiên về cho BOT tự trả lời lại (tuỳ chọn)."""
    memory.set_session_status(session_id, "bot")
    return {"session_id": session_id, "status": "bot"}


class AdminSettingsUpdate(BaseModel):
    # "code" = hàm shop (mặc định) | "finetune" = model FT phong thủy
    size_mode: Optional[str] = None


@app.get("/admin/settings")
def admin_get_settings(admin: dict = Depends(current_admin)):
    """Đọc cấu hình runtime (size_mode, …)."""
    import runtime_settings
    return runtime_settings.get_settings()


@app.put("/admin/settings")
def admin_put_settings(req: AdminSettingsUpdate, admin: dict = Depends(current_admin)):
    """Cập nhật cấu hình runtime — chỉ admin."""
    import runtime_settings
    try:
        out = runtime_settings.update_settings(size_mode=req.size_mode)
    except ValueError as e:
        raise HTTPException(400, str(e))
    log.info("ADMIN %s đổi settings: %s", admin["username"], out)
    return out


# Giá / tồn kho / khuyến mãi là dữ liệu ĐỘNG (cố ý không train vào model finetune).
# Trước đây muốn sửa phải vào pgAdmin gõ SQL — giờ chủ shop tự up file Excel.

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _xlsx_response(data: bytes, filename: str) -> StreamingResponse:
    return StreamingResponse(
        io.BytesIO(data),
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/admin/template/products")
def admin_template_products(admin: dict = Depends(current_admin)):
    """Tải file mẫu GIÁ + TỒN KHO, điền sẵn dữ liệu THẬT đang có → admin chỉ việc sửa số."""
    return _xlsx_response(admin_import.template_products(), "gia_tonkho.xlsx")


@app.get("/admin/template/promotions")
def admin_template_promotions(admin: dict = Depends(current_admin)):
    """Tải file mẫu KHUYẾN MÃI, điền sẵn các chương trình đang có."""
    return _xlsx_response(admin_import.template_promotions(), "khuyen_mai.xlsx")


@app.get("/admin/template/catalog")
def admin_template_catalog(admin: dict = Depends(current_admin)):
    """Tải file mẫu CATALOG SP mới (full field + image JSON format DB)."""
    return _xlsx_response(admin_import.template_catalog(), "catalog_san_pham.xlsx")


@app.post("/admin/import/products")
async def admin_import_products(file: UploadFile = File(...),
                                admin: dict = Depends(current_admin)):
    """Nạp file Excel cập nhật price_range + quantity_max (và đồng bộ OpenSearch)."""
    try:
        result = admin_import.import_products(await file.read())
    except admin_import.ImportError_ as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("admin_import products lỗi")
        raise HTTPException(500, f"Lỗi hệ thống khi nạp file: {e}")
    log.info("ADMIN %s nạp giá/tồn: %s", admin["username"], result)
    return result


@app.post("/admin/import/promotions")
async def admin_import_promotions(file: UploadFile = File(...),
                                  admin: dict = Depends(current_admin)):
    """Nạp file Excel cập nhật bảng khuyến mãi (file là NGUỒN SỰ THẬT — thay toàn bộ)."""
    try:
        result = admin_import.import_promotions(await file.read())
    except admin_import.ImportError_ as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("admin_import promotions lỗi")
        raise HTTPException(500, f"Lỗi hệ thống khi nạp file: {e}")
    log.info("ADMIN %s nạp khuyến mãi: %s", admin["username"], result)
    return result


@app.post("/admin/import/catalog")
async def admin_import_catalog(file: UploadFile = File(...),
                               admin: dict = Depends(current_admin)):
    """Nạp Excel catalog SP (insert/update full) + index text OS + vector ảnh SigLIP.
    Bản không stream (giữ tương thích). UI dùng /admin/import/catalog/stream để hiện tiến trình.
    """
    try:
        result = admin_import.import_catalog(await file.read(), reindex_images=True)
    except admin_import.ImportError_ as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("admin_import catalog lỗi")
        raise HTTPException(500, f"Lỗi hệ thống khi nạp catalog: {e}")
    log.info("ADMIN %s nạp catalog: %s", admin["username"], result)
    return result


@app.post("/admin/import/catalog/stream")
async def admin_import_catalog_stream(file: UploadFile = File(...),
                                      admin: dict = Depends(current_admin)):
    """Nạp catalog + SSE tiến trình (validate → Postgres → OpenSearch text → SigLIP).

    Events:
      stage=validate|postgres|opensearch_text|siglip|done|error
      message=... (hiển thị UI)
      result=... (khi done)
    """
    raw = await file.read()
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def work():
        token = progress.set_emitter(q.put)
        try:
            return admin_import.import_catalog(raw, reindex_images=True)
        finally:
            progress.reset_emitter(token)
            q.put(_DONE)

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def gen():
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, work)
        while True:
            try:
                ev = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.12)
                continue
            if ev is _DONE:
                break
            if isinstance(ev, dict):
                yield sse(ev)

        try:
            result = await fut
            # import_catalog đã emit done; gửi lại result nếu client miss
            if isinstance(result, dict):
                yield sse({
                    "stage": "done",
                    "message": (
                        f"Hoàn tất: {result.get('updated', 0)} SP "
                        f"(mới {result.get('created', 0)}, sửa {result.get('edited', 0)})."
                    ),
                    "result": result,
                })
            log.info("ADMIN %s nạp catalog (stream): %s", admin["username"], result)
        except admin_import.ImportError_ as e:
            yield sse({"stage": "error", "message": str(e)})
        except Exception as e:
            log.exception("admin_import catalog/stream lỗi")
            yield sse({"stage": "error", "message": f"Lỗi hệ thống khi nạp catalog: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest, user: dict = Depends(current_user)):
    if not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    if not memory.ensure_session(req.session_id, user["id"], title=req.message):
        raise HTTPException(403, "Phiên này thuộc về tài khoản khác")

    # Phiên đã chuyển cho chủ shop → bot NGỪNG trả lời, chỉ lưu tin nhắn khách.
    status = memory.get_session_status(req.session_id)
    if status != "bot":
        memory.log_turn(req.session_id, "user", req.message, user_id=user["id"])
        memory.touch_session(req.session_id, user["id"])
        # Khách nhắn thêm sau khi shop đã reply → cần shop trả lời lại (pending).
        if status == "admin":
            memory.set_session_status(req.session_id, "pending_admin")
            status = "pending_admin"
        return ChatResponse(response="", agent_used="handoff", intent="handoff",
                            tools_called=[], session_status=status)

    log.info("HTTP /chat user=%s session=%s msg='%s'",
             user["username"], req.session_id, req.message[:80].replace("\n", " "))
    out = chat_graph.chat(req.message, session_id=req.session_id, user_id=user["id"])
    memory.touch_session(req.session_id, user["id"], title_text=req.message)
    return ChatResponse(
        response=out["response"],
        agent_used=out["agent_used"],
        intent=out["intent"],
        tools_called=out["tools_called"],
        session_status=memory.get_session_status(req.session_id),
    )


def _prepare_images(req: ChatImageRequest, user: dict) -> tuple[list[dict], list[str]]:
    """Validate + lưu ảnh ra đĩa. Dùng chung cho endpoint thường và endpoint SSE."""
    images = [{"base64": it.image_b64, "mime": it.mime} for it in req.images]
    if not images and req.image_b64:
        images = [{"base64": req.image_b64, "mime": req.mime}]
    if not images:
        raise HTTPException(400, "Cần ít nhất 1 ảnh (images[] hoặc image_b64)")
    if len(images) > chat_graph.MAX_IMAGES:
        raise HTTPException(400, f"Tối đa {chat_graph.MAX_IMAGES} ảnh mỗi lượt (nhận {len(images)})")
    if not memory.ensure_session(req.session_id, user["id"], title=req.message or "(đã gửi ảnh)"):
        raise HTTPException(403, "Phiên này thuộc về tài khoản khác")
    return images, _save_images(images)


def _run_chat_image(req: ChatImageRequest, user: dict, images: list[dict],
                    image_urls: list[str]) -> ChatResponse:
    """Chạy pipeline cho lượt có ảnh (ĐỒNG BỘ, chặn — nhanh/chậm tuỳ finetune_full)."""
    # Chế độ trả dữ liệu của model finetune, đặt theo nút gạt trên UI. Contextvar nên
    # an toàn khi nhiều khách gửi ảnh cùng lúc với chế độ khác nhau.
    knowledge_base_agent.set_finetune_full(req.finetune_full)

    # Phiên đã chuyển cho chủ shop → bot NGỪNG, chỉ lưu tin + ảnh của khách.
    status = memory.get_session_status(req.session_id)
    if status != "bot":
        memory.log_turn(req.session_id, "user", (req.message or "(đã gửi ảnh)"),
                        user_id=user["id"], images=image_urls)
        memory.touch_session(req.session_id, user["id"])
        if status == "admin":
            memory.set_session_status(req.session_id, "pending_admin")
            status = "pending_admin"
        return ChatResponse(response="", agent_used="handoff", intent="handoff",
                            tools_called=[], session_status=status)

    log.info("HTTP /chat/image user=%s session=%s msg='%s' imgs=%d",
             user["username"], req.session_id, (req.message or "")[:80], len(images))
    out = chat_graph.chat_with_image(
        user_message = req.message,
        images = images,
        session_id = req.session_id,
        user_id = user["id"],
        image_urls = image_urls,
    )
    memory.touch_session(req.session_id, user["id"], title_text=(req.message or "(đã gửi ảnh)"))
    return ChatResponse(
        response=out["response"],
        agent_used=out["agent_used"],
        intent=out["intent"],
        tools_called=out["tools_called"],
        session_status=memory.get_session_status(req.session_id),
    )


@app.post("/chat/image", response_model=ChatResponse)
def chat_image_endpoint(req: ChatImageRequest, user: dict = Depends(current_user)):
    """Bản KHÔNG stream — giữ nguyên để tương thích ngược / làm dự phòng cho UI."""
    images, image_urls = _prepare_images(req, user)
    return _run_chat_image(req, user, images, image_urls)


@app.post("/chat/image/stream")
async def chat_image_stream_endpoint(req: ChatImageRequest, user: dict = Depends(current_user)):
    """Bản STREAM (SSE) — vừa chạy vừa báo tiến trình về UI.

    Khách gửi ảnh phải chờ ~90-140s cho model finetune nhận diện. Nếu im lặng suốt
    thời gian đó, khách tưởng bot treo. Endpoint này phát các mốc:
        identifying → "Đang xác minh sản phẩm trong ảnh..."
        identified → "Đã nhận ra: <tên SP>. Đang tra giá & tồn kho..."
        answering → "Đang soạn câu trả lời..."
        done → câu trả lời cuối (kèm agent_used / tools_called)
        error → lỗi

    Pipeline là code ĐỒNG BỘ (chặn), nên chạy nó trong thread; các bước bên trong đẩy
    sự kiện vào hàng đợi, generator ở đây rút ra và stream. run_in_executor có COPY
    contextvars sang thread nên progress.emit() tìm đúng hàng đợi của request này.
    """
    images, image_urls = _prepare_images(req, user)
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def work() -> ChatResponse:
        token = progress.set_emitter(q.put) # đăng ký cho THREAD này
        try:
            return _run_chat_image(req, user, images, image_urls)
        finally:
            progress.reset_emitter(token)
            q.put(_DONE) # báo generator: hết sự kiện

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def gen():
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, work)

        # Rút hàng đợi cho tới khi thread báo xong.
        while True:
            try:
                ev = q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.15) # nhường event loop
                continue
            if ev is _DONE:
                break
            yield sse(ev)

        try:
            out: ChatResponse = await fut
            yield sse({"stage": "done", **out.model_dump()})
        except HTTPException as e:
            yield sse({"stage": "error", "message": e.detail})
        except Exception as e:
            log.exception("chat/image/stream lỗi")
            yield sse({"stage": "error", "message": f"Lỗi hệ thống: {e}"})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no", # tắt buffering nếu sau này có nginx đứng trước
        },
    )


@app.get("/history/{session_id}")
def history_endpoint(session_id: str, limit: int = 50, user: dict = Depends(current_user)):
    owner = memory.session_owner(session_id)
    if owner is not None and owner != user["id"] and not auth.is_admin(user.get("username")):
        raise HTTPException(403, "Phiên này thuộc về tài khoản khác")
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content, agent_used, intent, tools_called, created_at, images
                FROM conversation_log
                WHERE session_id = :sid
                ORDER BY created_at DESC
                LIMIT :lim
            """),
            {"sid": session_id, "lim": limit},
        ).fetchall()
    return {
        "session_id": session_id,
        "session_status": memory.get_session_status(session_id),
        "turns": [
            {
                "role": r[0],
                "content": r[1],
                "agent_used": r[2],
                "intent": r[3],
                "tools_called": r[4],
                "created_at": str(r[5]),
                "images": r[6] or [],
            }
            for r in reversed(rows)
        ],
    }


@app.delete("/history/{session_id}")
def delete_history_endpoint(session_id: str, user: dict = Depends(current_user)):
    """Xoá phiên + lịch sử của phiên (chỉ chủ sở hữu)."""
    deleted = memory.delete_session(session_id, user["id"])
    log.info("HTTP DELETE /history user=%s session=%s deleted=%d turns",
             user["username"], session_id, deleted)
    return {"session_id": session_id, "deleted": deleted}


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
                "id": r[0],
                "session_id": r[1],
                "reason": r[2],
                "user_summary": r[3],
                "status": r[4],
                "created_at": str(r[5]),
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
