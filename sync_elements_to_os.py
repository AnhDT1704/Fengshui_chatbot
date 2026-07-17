"""
sync_elements_to_os.py – Đẩy lại trường `compatible_elements` từ PostgreSQL
sang OpenSearch cho các sản phẩm vừa cập nhật (partial update, không re-embedding).
Cập nhật danh sách CHANGED_PK_IDS mỗi lần dùng lại.
"""
import db_service
import opensearch_service
import config

# id (PK) của các sản phẩm vòng tay vừa đổi mệnh trong migration 005
CHANGED_PK_IDS = [9, 11, 27, 33, 57, 62, 96, 108]

db_service.init_db()
client = opensearch_service.get_client()

updated, missing = 0, []
for product in db_service.get_all_products():
    if product.id not in CHANGED_PK_IDS:
        continue
    pid = product.product_id
    elements = product.compatible_elements or []
    if not client.exists(index=config.OS_INDEX, id=str(pid)):
        missing.append(pid)
        continue
    client.update(
        index=config.OS_INDEX,
        id=str(pid),
        body={"doc": {"compatible_elements": elements}},
        refresh=True,
    )
    print(f"  ✓ product_id={pid} (pk={product.id}) -> compatible_elements={elements}")
    updated += 1

print(f"\nĐã cập nhật {updated}/{len(CHANGED_PK_IDS)} docs trong OpenSearch.")
if missing:
    print(f"⚠ Không tìm thấy doc cho product_id: {missing}")
