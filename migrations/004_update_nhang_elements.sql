-- 004_update_nhang_elements.sql
-- Điền mệnh phù hợp (compatible_elements) cho các sản phẩm nhang.
-- Nguồn: Product_color.txt. Trước đó tất cả đều để chung {Kim,Mộc,Thủy,Hỏa,Thổ}.
-- 4 sản phẩm "Mọi mệnh" (id 56,58,61,95) đã đúng all-5 -> giữ nguyên, không cập nhật.

SET client_encoding TO 'UTF8';

BEGIN;

-- Nhang trầm hương (Kim, Thủy, Mộc)
UPDATE products SET compatible_elements = ARRAY['Kim','Thủy','Mộc'] WHERE id IN (19,23,36,59,63,74,90);

-- Nhang quế (Hỏa, Thổ)
UPDATE products SET compatible_elements = ARRAY['Hỏa','Thổ'] WHERE id IN (65,75);

-- Nhang bài (Thổ, Kim)
UPDATE products SET compatible_elements = ARRAY['Thổ','Kim'] WHERE id = 97;

-- Nhang khuynh diệp (Mộc, Hỏa)
UPDATE products SET compatible_elements = ARRAY['Mộc','Hỏa'] WHERE id = 110;

COMMIT;

-- Kết quả sau cập nhật
SELECT id, compatible_elements FROM products WHERE id IN (19,23,36,59,63,65,74,75,90,97,110) ORDER BY id;
