"""
008_add_care_instructions.py – Bổ sung mục HƯỚNG DẪN BẢO QUẢN vào
product_description cho các sản phẩm còn thiếu, theo nhóm.

Map theo cột PK `id` (KHÔNG phải product_id). Idempotent: chỉ thêm nếu mô tả
CHƯA chứa từ "bảo quản" → chạy lại nhiều lần không bị nhân đôi.

Chạy:  python migrations/008_add_care_instructions.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Product, get_session  # noqa: E402


# (heading, [bullet], [ids])  — nội dung do shop cung cấp
GROUPS = [
    (
        "HƯỚNG DẪN BẢO QUẢN",
        [
            "Bảo quản sản phẩm ở nơi khô ráo, thoáng mát, tuyệt đối tránh môi trường ẩm ướt để không làm giảm chất lượng hương thơm và tránh ẩm mốc.",
            "Sau khi sử dụng, cần đậy kín nắp hộp hoặc đóng kín túi zip để lưu giữ mùi hương trầm/quế nguyên bản lâu nhất.",
            "Tránh để sản phẩm tiếp xúc trực tiếp với ánh nắng mặt trời gay gắt trong thời gian dài vì sẽ làm phai mùi hương tự nhiên.",
        ],
        [15, 19, 23, 36, 59, 63, 65, 74, 90, 99, 104],
    ),
    (
        "HƯỚNG DẪN BẢO QUẢN VÀ SỬ DỤNG TRẦM HƯƠNG",
        [
            "Tuyệt đối tránh để trầm hương tiếp xúc trực tiếp với nước. Nếu lỡ dính nước, hãy dùng khăn giấy thấm nhẹ và sấy khô ở nhiệt độ thấp hoặc để khô tự nhiên.",
            "Không để trầm hương tiếp xúc với hóa chất, chất tẩy rửa, mỹ phẩm hoặc nước hoa vì sẽ làm lấn át và mất đi mùi thơm tự nhiên của trầm.",
            "Khi thấy trầm giảm mùi thơm, bạn có thể dùng một chiếc khăn vải cotton mềm chà xát nhẹ nhàng nhiều lần xung quanh hạt trầm để lấy lại độ bóng và làm ấm vân dầu, giúp mùi hương tỏa ra trở lại.",
            "Khi không sử dụng, hãy cất sản phẩm vào hộp hoặc túi zip kín.",
        ],
        [8, 24, 45, 82, 102, 105, 109],
    ),
    (
        "HƯỚNG DẪN BẢO QUẢN VÀ VỆ SINH LƯ XÔNG",
        [
            "Sau khi xông trầm (đặc biệt là trầm nụ), tinh dầu trầm thường sẽ tích tụ lại ở đáy lư hoặc đường dẫn khói. Bạn nên dùng khăn mềm ướt hoặc tăm bông thấm cồn y tế lau sạch để lư luôn bóng đẹp và không bị tắc nghẽn.",
            "Đối với lư xông gốm sứ/thác khói: Đặt ở mặt phẳng cố định, tránh xa tầm tay trẻ em và thú cưng để phòng ngừa rơi vỡ.",
            "Đối với lư đồng: Lau khô ngay nếu dính nước để tránh đồng bị xỉn màu hoặc xuất hiện các đốm xanh oxit. Thường xuyên lau chùi bằng khăn vải mềm để duy trì độ bóng.",
        ],
        [47, 49, 66, 67, 77, 86, 94],
    ),
    (
        "HƯỚNG DẪN BẢO QUẢN",
        [
            "Với lư xông gỗ: Đổ bỏ tàn nhang sau mỗi lần sử dụng. Tránh để lư gỗ tiếp xúc với nước hoặc để ở nơi có độ ẩm cao để phòng ngừa mốc mọt. Tránh dùng các hóa chất tẩy rửa mạnh lau trực tiếp lên bề mặt gỗ.",
            "Với lư xông điện: Luôn rút phích cắm điện ra khỏi ổ cắm ngay sau khi sử dụng xong hoặc khi tiến hành lau chùi. Chờ khay nhôm nguội hẳn mới dùng nhíp gắp tàn trầm ra. Tuyệt đối không để nước dính vào mâm nhiệt và các linh kiện điện tử.",
        ],
        [32, 73, 100],
    ),
    (
        "HƯỚNG DẪN BẢO QUẢN",
        [
            "Đậy thật kín nắp ngay sau khi sử dụng để tránh tinh dầu bay hơi làm giảm chất lượng.",
            "Bảo quản ở nơi thoáng mát, khô ráo, tránh ánh sáng mặt trời chiếu trực tiếp làm biến đổi các thành phần tự nhiên.",
            "Tránh xa tầm tay trẻ em và thú cưng.",
        ],
        [31, 70, 85],
    ),
    (
        "HƯỚNG DẪN BẢO QUẢN",
        [
            "Tượng để bàn / Lá bồ đề / Đồng xu: Dùng chổi cọ mềm hoặc khăn khô để phủi bụi thường xuyên. Không dùng vật nhám cứng chà xát để tránh làm trầy xước lớp mạ bảo vệ bên ngoài.",
            "Than hoạt tính: Đặc biệt chú ý bảo quản trong túi zip bọc kín, để ở môi trường khô ráo hoàn toàn. Nếu than hút ẩm từ không khí sẽ rất khó mồi lửa và tạo nhiều khói khi đốt.",
        ],
        [48, 72, 79, 88, 98, 107],
    ),
]


def _block(heading: str, bullets: list[str]) -> str:
    return "\n\n" + heading + ":\n" + "\n".join("- " + b for b in bullets)


def main():
    session = get_session()
    updated, skipped, missing = [], [], []
    try:
        for heading, bullets, ids in GROUPS:
            block = _block(heading, bullets)
            for pid in ids:
                p = session.query(Product).filter(Product.id == pid).first()
                if p is None:
                    missing.append(pid)
                    continue
                desc = p.product_description or ""
                if "bảo quản" in desc.lower():
                    skipped.append(pid)
                    continue
                p.product_description = desc.rstrip() + block
                updated.append(pid)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"Đã bổ sung bảo quản: {len(updated)} sản phẩm -> {sorted(updated)}")
    print(f"Bỏ qua (đã có): {len(skipped)} -> {sorted(skipped)}")
    if missing:
        print(f"Không tồn tại: {sorted(missing)}")


if __name__ == "__main__":
    main()
