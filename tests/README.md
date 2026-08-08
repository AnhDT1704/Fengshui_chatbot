# Tests — Chatbot phong thủy

## Cấu trúc
| File | Loại | Cần gì |
|---|---|---|
| `test_logic.py` | **Unit** (deterministic, không LLM/mạng) | chỉ import module |
| `test_chatbot_api.py` | **Integration** luồng khách | API `localhost:8000` + Gemini + (FT nếu bật) |
| `test_admin_api.py` | **Integration** tài khoản admin | API + user admin (`admin1`) |
| `conftest.py` | fixtures dùng chung (token user/admin, helper HTTP) | |

## Bao phủ
- **Logic**: tính size vòng (số hạt / cung Sinh-Lão-Bệnh-Tử), câu tồn kho (`_stock_phrase`, chống đọc số kho thô), mệnh từ năm sinh, Shopee CTA (chống cụt nội dung khuyến mãi), capture metadata mệnh (đa người), parse JSON model FT.
- **Khách**: health, bắt buộc auth, small talk, tư vấn mệnh + sản phẩm thật, tồn kho không đọc số thô, bảo hành theo cột SP, khuyến mãi đủ 2 vế, off-platform redirect, nhớ mệnh giữa các lượt.
- **Admin**: phân quyền (403/401), `size_mode` get/put/validate, danh sách handoff/users, máy trạng thái handoff (bot→admin→bot), tải template Excel, nạp Excel round-trip (idempotent — không đổi dữ liệu shop), từ chối file rác.

## Chạy
```bash
# Trong container (đã có sẵn dependency + DB + OpenSearch):
docker exec fengshui_chatbot pip install -q pytest          # 1 lần (xem requirements-test.txt)
docker exec -e PYTHONIOENCODING=utf-8 fengshui_chatbot python -m pytest /app/tests -v

# Chỉ unit (nhanh, không cần API):
docker exec fengshui_chatbot python -m pytest /app/tests/test_logic.py -v

# Bỏ integration (khi API/model chưa chạy):
docker exec fengshui_chatbot python -m pytest /app/tests -v -m "not integration"
```
Integration tests **tự SKIP** nếu API không chạy tại `CHATBOT_TEST_URL` (mặc định `http://localhost:8000`).

## Lưu ý
- Test integration phụ thuộc LLM (không tất định) → assertion cố ý **lỏng**, chỉ bám invariant (không rỗng, không lộ thuật ngữ nội bộ, đúng cột, đủ vế…).
- Test tạo vài user tạm (`pytest_user_*`) trong DB — vô hại.
- Import Excel round-trip dùng template có sẵn dữ liệu THẬT → nạp lại là idempotent, **không đổi** giá/tồn/khuyến mãi của shop.
