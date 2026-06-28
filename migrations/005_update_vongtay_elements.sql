-- 005_update_vongtay_elements.sql
-- Điều chỉnh mệnh (compatible_elements) cho một số vòng tay.
-- Nguồn: Product_color.txt (đợt thay đổi cuối).

SET client_encoding TO 'UTF8';

BEGIN;

-- Vòng đa sắc / ngũ sắc / theo mệnh -> hợp mọi mệnh (cả 5 hành)
UPDATE products SET compatible_elements = ARRAY['Kim','Mộc','Thủy','Hỏa','Thổ'] WHERE id IN (9,11,27,62,96,108);

-- Vòng đá mắt mèo trắng (Kim, Thủy)
UPDATE products SET compatible_elements = ARRAY['Kim','Thủy'] WHERE id = 33;

-- Vòng chỉ đỏ đồng điếu (Hỏa, Thổ, Mộc)
UPDATE products SET compatible_elements = ARRAY['Hỏa','Thổ','Mộc'] WHERE id = 57;

COMMIT;

-- Kết quả sau cập nhật
SELECT id, compatible_elements FROM products WHERE id IN (9,11,27,33,57,62,96,108) ORDER BY id;
