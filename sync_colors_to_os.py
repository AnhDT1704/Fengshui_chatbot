"""
sync_colors_to_os.py – Đẩy lại trường `colors` từ PostgreSQL sang OpenSearch
cho các sản phẩm vừa cập nhật (partial update, không re-embedding).
"""
import db_service
import opensearch_service
import config

# id (PK) của 15 sản phẩm vừa đổi màu trong migration 002
CHANGED_PK_IDS = [1, 4, 12, 13, 14, 16, 18, 25, 27, 29, 38, 39, 40, 41, 96]

db_service.init_db()
client = opensearch_service.get_client()

updated, missing = 0, []
for product in db_service.get_all_products():
    if product.id not in CHANGED_PK_IDS:
        continue
    pid = product.product_id
    colors = product.colors or []
    if not client.exists(index=config.OS_INDEX, id=str(pid)):
        missing.append(pid)
        continue
    client.update(
        index=config.OS_INDEX,
        id=str(pid),
        body={"doc": {"colors": colors}},
        refresh=True,
    )
    print(f"  ✓ product_id={pid} (pk={product.id}) -> colors={colors}")
    updated += 1

print(f"\nĐã cập nhật {updated}/{len(CHANGED_PK_IDS)} docs trong OpenSearch.")
if missing:
    print(f"⚠ Không tìm thấy doc cho product_id: {missing}")
