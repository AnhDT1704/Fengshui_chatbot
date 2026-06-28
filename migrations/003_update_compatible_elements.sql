-- 003_update_compatible_elements.sql
-- Điền mệnh phù hợp (compatible_elements) cho các sản phẩm lư/thác khói xông trầm.
-- Nguồn: Product_color.txt. Trước đó tất cả đều để chung {Kim,Mộc,Thủy,Hỏa,Thổ}.
-- id 49 ghi "Chưa xác định" trong file -> giữ nguyên, không cập nhật.

SET client_encoding TO 'UTF8';

BEGIN;

-- Lư gỗ / hoa sen / điện xông (hành Mộc sinh Hỏa)
UPDATE products SET compatible_elements = ARRAY['Mộc','Hỏa'] WHERE id = 32;
UPDATE products SET compatible_elements = ARRAY['Mộc','Hỏa'] WHERE id = 73;
UPDATE products SET compatible_elements = ARRAY['Mộc','Hỏa'] WHERE id = 86;

-- Lư đồng / kim loại (hành Kim sinh Thủy)
UPDATE products SET compatible_elements = ARRAY['Kim','Thủy'] WHERE id = 47;
UPDATE products SET compatible_elements = ARRAY['Kim','Thủy'] WHERE id = 76;

-- Thác khói / lư có yếu tố nước (Thủy, Mộc, Thổ, Kim)
UPDATE products SET compatible_elements = ARRAY['Thủy','Mộc','Thổ','Kim'] WHERE id IN (66,67,77,94,100);

COMMIT;

-- Kết quả sau cập nhật
SELECT id, compatible_elements FROM products WHERE id IN (32,47,66,67,73,76,77,86,94,100) ORDER BY id;
