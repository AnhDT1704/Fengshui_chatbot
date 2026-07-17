-- 007_cleanup_materials.sql
-- Chuẩn hóa dữ liệu cột material:
--   1. "Đá Mã Não" (viết hoa, id 1/4/12) -> "mã não" cho nhất quán casing.
--   2. "không xác định" (id 49 - lư đồng) -> "đồng" (theo product_description: "Chất liệu: đồng cao cấp").

SET client_encoding TO 'UTF8';

BEGIN;

UPDATE products SET material = array_replace(material, 'Đá Mã Não', 'mã não')
WHERE 'Đá Mã Não' = ANY(material);

UPDATE products SET material = array_replace(material, 'không xác định', 'đồng')
WHERE id = 49;

COMMIT;

SELECT id, product_id, material FROM products WHERE id IN (1, 4, 12, 49) ORDER BY id;
