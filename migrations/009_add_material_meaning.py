"""
009_add_material_meaning.py – Bổ sung mục "Ý NGHĨA PHONG THỦY CHẤT LIỆU" vào
product_description cho từng sản phẩm, dựa trên cột `material`.

Mục tiêu: chatbot lấy ý nghĩa/công dụng sản phẩm TỪ DỮ LIỆU THẬT (product_description)
thay vì kiến thức tự thân của model. Nội dung ý nghĩa tổng hợp từ các nguồn phong
thủy Việt, đã được shop duyệt.

Idempotent: chỉ thêm nếu product_description CHƯA chứa marker "Ý NGHĨA PHONG THỦY
CHẤT LIỆU" → chạy lại nhiều lần không bị nhân đôi.

Bỏ qua material "giấy dán" và "nước" (không mang ý nghĩa phong thủy).

Chạy:  python migrations/009_add_material_meaning.py
   (hoặc trong container: docker exec fengshui_chatbot python migrations/009_add_material_meaning.py)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import Product, get_session  # noqa: E402


MARKER = "Ý NGHĨA PHONG THỦY CHẤT LIỆU"

# material (chữ thường, đúng như trong DB) -> ý nghĩa 2-3 câu (shop đã duyệt)
MATERIAL_MEANING = {
    "trầm hương": (
        "Trầm hương mang dương khí cực mạnh, từ lâu được xem là vật phẩm phong thủy quý "
        "giúp trừ tà, xua đuổi năng lượng tiêu cực và vận xui. Đồng thời chiêu tài lộc, "
        "mang lại bình an, may mắn cho người đeo; hương thơm dịu nhẹ còn giúp an thần, "
        "thư giãn tinh thần. Trầm hương hợp với mọi mệnh và mọi lứa tuổi."
    ),
    "thạch anh": (
        "Đá thạch anh tích tụ nguồn năng lượng tự nhiên mạnh mẽ, có tác dụng tiêu trừ "
        "năng lượng xấu, trấn trạch, hóa giải sát khí và mang lại bình an, may mắn. Đeo "
        "thạch anh giúp tinh thần minh mẫn, cân bằng cảm xúc và thu hút vượng khí cho "
        "công việc, sự nghiệp."
    ),
    "thạch anh xanh": (
        "Đá thạch anh xanh tích tụ năng lượng tự nhiên mạnh mẽ, giúp tiêu trừ khí xấu, "
        "an thần và cân bằng cảm xúc. Sắc xanh thuộc hành Mộc, tượng trưng cho sự tươi "
        "mới, sinh sôi và phát triển, mang lại bình an, may mắn cho người đeo."
    ),
    "mã não": (
        "Trong phong thủy, đá mã não là biểu tượng của sự bình an và cân bằng năng lượng, "
        "được xem như bùa hộ mệnh giúp trừ tà, bảo vệ sức khỏe và mang đến sung túc, "
        "trường thọ. Đeo mã não giúp ổn định cảm xúc, giảm căng thẳng, tăng sự tự tin "
        "trong giao tiếp và thuận lợi trong công danh, kinh doanh."
    ),
    "mã não đa sắc": (
        "Mã não đa sắc hội tụ nhiều màu sắc cân bằng đủ ngũ hành nên hợp với mọi mệnh, "
        "mang lại bình an toàn diện và hóa giải khí xấu. Đá là biểu tượng của sự cân bằng "
        "năng lượng, giúp ổn định cảm xúc, trừ tà và thu hút may mắn, tài lộc cho người đeo."
    ),
    "mã não rêu": (
        "Mã não rêu mang ý nghĩa bình an, cân bằng và kết nối với thiên nhiên, giúp ổn "
        "định cảm xúc và giảm căng thẳng. Sắc xanh rêu thuộc hành Mộc, tượng trưng cho "
        "sự sinh sôi, phát triển và may mắn trong sự nghiệp."
    ),
    "mã não đen": (
        "Mã não đen được xem như lá chắn bảo vệ, giúp xua đuổi tà khí và năng lượng tiêu "
        "cực, mang lại cảm giác an tâm, vững vàng. Sắc đen thuộc hành Thủy, hợp người "
        "mệnh Thủy và Mộc, hỗ trợ củng cố ý chí và sự kiên định."
    ),
    "mã não trắng": (
        "Mã não trắng tượng trưng cho sự thuần khiết, bình an và cân bằng năng lượng, "
        "giúp thanh lọc khí xấu và mang lại tinh thần minh mẫn. Sắc trắng thuộc hành "
        "Kim, hợp người mệnh Kim và Thủy, hỗ trợ sự tập trung và sáng suốt."
    ),
    "mã não xanh lá": (
        "Mã não xanh lá là biểu tượng của bình an, cân bằng và may mắn, giúp ổn định cảm "
        "xúc và giảm căng thẳng. Sắc xanh lá thuộc hành Mộc, tượng trưng cho sự sinh sôi, "
        "phát triển và tươi mới, hợp người mệnh Mộc và Hỏa."
    ),
    "tourmaline": (
        "Đá tourmaline được xem như lá bùa năng lượng, giúp xua đuổi năng lượng tiêu cực, "
        "giảm căng thẳng lo âu và mang lại cảm giác an toàn, tích cực. Đặc biệt tourmaline "
        "đa sắc cân bằng đủ ngũ hành nên hợp mọi mệnh, hỗ trợ thu hút tài lộc, may mắn và "
        "thành công trong công việc."
    ),
    "aquamarine": (
        "Đá Aquamarine (ngọc xanh biển, họ Beryl) mang năng lượng của nước, giúp làm dịu "
        "tâm trí, giảm căng thẳng và mang lại sự thanh thản, sáng suốt. Đá là biểu tượng "
        "của tình yêu, hôn nhân hạnh phúc, giúp cải thiện giao tiếp; thuộc hành Thủy, hợp "
        "mệnh Thủy và Mộc."
    ),
    "đá beryl": (
        "Đá Beryl là dòng đá quý (cùng họ với aquamarine, ngọc lục bảo) mang năng lượng "
        "làm dịu tâm trí, giảm căng thẳng và tăng sự minh mẫn, sáng suốt. Đá giúp cải "
        "thiện giao tiếp, mang lại các mối quan hệ hài hòa và cảm giác bình an cho người đeo."
    ),
    "mắt mèo": (
        "Đá mắt mèo được xem như bùa hộ mệnh giúp xua đuổi tà khí, thanh lọc năng lượng "
        "xấu và bảo vệ chủ nhân khỏi tai ương, xui xẻo. Đá mang lại may mắn, tài lộc, "
        "thịnh vượng, đồng thời giúp giữ vững tinh thần, tăng sự tự tin và sáng suốt."
    ),
    "mắt mèo xanh": (
        "Đá mắt mèo xanh là bùa hộ mệnh giúp xua đuổi tà khí, thanh lọc năng lượng xấu và "
        "bảo vệ chủ nhân, đồng thời mang lại may mắn, tài lộc. Sắc xanh tượng trưng cho sự "
        "tươi mới, sinh sôi và bình an, giúp giữ tinh thần vững vàng, sáng suốt."
    ),
    "mắt mèo đỏ": (
        "Đá mắt mèo đỏ là bùa hộ mệnh giúp xua đuổi tà khí, bảo vệ chủ nhân khỏi xui xẻo "
        "và mang lại may mắn, tài lộc. Sắc đỏ thuộc hành Hỏa, tượng trưng cho nhiệt huyết, "
        "năng lượng mạnh mẽ và sự quyết đoán, hợp người mệnh Hỏa và Thổ."
    ),
    "mắt mèo vàng": (
        "Đá mắt mèo vàng là bùa hộ mệnh giúp xua đuổi tà khí, bảo vệ chủ nhân và thu hút "
        "may mắn, tài lộc. Sắc vàng thuộc hành Thổ/Kim, tượng trưng cho vượng khí, tiền "
        "tài và sự sung túc, giúp giữ tinh thần vững vàng, tự tin."
    ),
    "gốm sứ": (
        "Sản phẩm gốm sứ (lư xông trầm, thác khói) dùng để xông trầm hương, giúp thanh "
        "lọc không gian, xua đuổi tà khí và thu hút tài lộc, bình an cho gia chủ. Chất "
        "men gốm bền đẹp, cách nhiệt tốt, vừa mang ý nghĩa tâm linh vừa là vật trang trí "
        "trang nhã cho không gian sống."
    ),
    "đồng": (
        "Vật phẩm phong thủy bằng đồng giúp thu hút tài lộc, may mắn, sức khỏe và sự thịnh "
        "vượng, đồng thời cân bằng âm dương, tạo không gian sống hài hòa. Đồng cũng có tác "
        "dụng trừ tà, xua đuổi âm khí, mang lại bình an cho gia chủ; chất liệu bền đẹp, "
        "sang trọng."
    ),
    "đồng thau": (
        "Vật phẩm bằng đồng thau mang ý nghĩa thu hút tài lộc, may mắn và thịnh vượng, "
        "cân bằng âm dương cho không gian sống. Đồng thau bền, không bị oxy hóa, vừa trừ "
        "tà xua âm khí vừa mang vẻ đẹp sang trọng, tinh tế."
    ),
    "chỉ đỏ": (
        "Vòng chỉ đỏ theo quan niệm dân gian có khả năng bảo vệ chủ nhân khỏi điều xấu, "
        "xua đuổi tà ma và vận xui, mang lại bình an, may mắn. Màu đỏ còn giúp thu hút "
        "cát khí, tài lộc và giúp đường tình duyên thuận lợi."
    ),
    "chỉ": (
        "Vòng chỉ là vật phẩm phong thủy quấn từ sợi chỉ, theo quan niệm dân gian giúp "
        "bảo vệ chủ nhân, xua đuổi vận xui và mang lại bình an, may mắn cho người đeo."
    ),
    "gỗ": (
        "Vật phẩm bằng gỗ phong thủy mang lại bình an, may mắn và tài lộc, đồng thời xua "
        "đuổi tà khí, bảo vệ người dùng khỏi năng lượng tiêu cực. Gỗ thuộc hành Mộc, "
        "tượng trưng cho sự sinh sôi, phát triển và sức sống."
    ),
    "vỏ quế": (
        "Vỏ quế có hương thơm ấm nồng, theo quan niệm dân gian giúp xua đuổi tà khí, côn "
        "trùng và thanh lọc không khí, mang lại không gian ấm cúng, bình an. Quế còn được "
        "dùng để tẩy uế, mang ý nghĩa cầu may mắn và bảo vệ cho gia chủ."
    ),
    "thảo mộc": (
        "Các loại thảo mộc có hương thơm tự nhiên giúp đuổi côn trùng, thanh lọc không "
        "khí và xua đuổi tà khí, mang lại không gian trong lành, bình an. Theo quan niệm "
        "phong thủy, thảo mộc là biểu tượng của sự an lành, may mắn cho gia chủ."
    ),
    "rễ cây bài": (
        "Rễ cây bài (hương bài) có hương thơm dịu tự nhiên, thường dùng làm hương/nhang, "
        "giúp thanh lọc không gian, xua đuổi côn trùng và tà khí. Theo quan niệm dân gian, "
        "hương bài mang lại sự ấm cúng, bình an và may mắn cho gia đình."
    ),
}

# material KHÔNG ghi ý nghĩa (shop xác nhận không mang ý nghĩa phong thủy)
SKIP_MATERIALS = {"giấy dán", "nước"}


def _meaning_for(material: str):
    """Tra ý nghĩa theo material (không phân biệt hoa/thường, bỏ khoảng trắng thừa)."""
    key = (material or "").strip().lower()
    if key in SKIP_MATERIALS:
        return None
    return MATERIAL_MEANING.get(key)


def _build_block(materials: list[str]) -> str:
    """Gom ý nghĩa cho các material của 1 sản phẩm (giữ thứ tự, không trùng)."""
    seen, bullets = set(), []
    for m in materials or []:
        key = (m or "").strip().lower()
        if not key or key in seen:
            continue
        meaning = _meaning_for(m)
        if not meaning:
            continue
        seen.add(key)
        label = m.strip()
        label = label[:1].upper() + label[1:]
        bullets.append(f"- {label}: {meaning}")
    if not bullets:
        return ""
    return "\n\n" + MARKER + ":\n" + "\n".join(bullets)


def main():
    session = get_session()
    updated, skipped, no_meaning = [], [], []
    try:
        products = session.query(Product).all()
        for p in products:
            desc = p.product_description or ""
            if MARKER in desc:
                skipped.append(p.id)
                continue
            block = _build_block(list(p.material or []))
            if not block:
                no_meaning.append(p.id)
                continue
            p.product_description = desc.rstrip() + block
            updated.append(p.id)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"Đã bổ sung ý nghĩa: {len(updated)} sản phẩm -> {sorted(updated)}")
    print(f"Bỏ qua (đã có marker): {len(skipped)} -> {sorted(skipped)}")
    print(f"Không có material khớp ý nghĩa: {len(no_meaning)} -> {sorted(no_meaning)}")


if __name__ == "__main__":
    main()
