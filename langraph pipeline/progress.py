"""
progress.py – Kênh phát TRẠNG THÁI XỬ LÝ về UI (Server-Sent Events).

Vì sao cần: khách gửi ảnh → model finetune nhận diện mất ~90-140s. Suốt thời gian đó
UI chỉ hiện 3 chấm nhấp nháy, khách tưởng bot treo. Ta phát các mốc ("đang nhận diện",
"đã nhận ra <tên SP>", "đang tra giá/tồn") để khách thấy hệ thống VẪN đang chạy.

Cách hoạt động:
  - api.py mở 1 hàng đợi cho mỗi request, đăng ký hàm emit qua set_emitter().
  - Pipeline (graph / knowledge_base_agent) gọi progress.emit(...) ở các mốc quan trọng.
  - Endpoint SSE rút hàng đợi và stream về trình duyệt.

Dùng contextvars để an toàn khi nhiều request chạy song song. Pipeline chạy trong
thread qua loop.run_in_executor(); contextvars được COPY sang thread đó nên emit()
vẫn tìm đúng hàng đợi của request tương ứng.

Nếu KHÔNG có emitter (vd endpoint /chat/image cũ, không stream) thì emit() là no-op —
pipeline chạy y hệt như trước, không ảnh hưởng gì.
"""

from __future__ import annotations

import contextvars
from typing import Callable, Optional

# Hàm nhận sự kiện của request hiện tại (None = không stream → emit() không làm gì).
_EMIT: contextvars.ContextVar[Optional[Callable[[dict], None]]] = contextvars.ContextVar(
    "progress_emit", default=None
)


def set_emitter(fn: Callable[[dict], None]):
    """Đăng ký hàm nhận sự kiện cho request hiện tại. Trả token để reset sau."""
    return _EMIT.set(fn)


def reset_emitter(token) -> None:
    _EMIT.reset(token)


def emit(stage: str, message: str, **extra) -> None:
    """Phát 1 mốc trạng thái về UI.

    Args:
        stage: mã bước, để UI tuỳ biến (vd 'identifying', 'identified', 'answering')
        message: câu hiển thị cho KHÁCH (tiếng Việt, thân thiện — khách sẽ đọc câu này)
        extra: dữ liệu kèm theo (vd product_name)
    """
    fn = _EMIT.get()
    if fn is None:
        return # không ở chế độ stream → bỏ qua, pipeline chạy như cũ
    try:
        fn({"stage": stage, "message": message, **extra})
    except Exception:
        # Sự cố khi phát tiến trình TUYỆT ĐỐI không được làm hỏng câu trả lời.
        pass
