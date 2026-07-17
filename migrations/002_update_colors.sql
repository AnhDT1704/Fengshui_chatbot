-- 002_update_colors.sql
-- Cập nhật chi tiết màu cho các sản phẩm vốn có colors = "đa sắc".
-- Nguồn: Product_color.txt. Giữ nguyên các "cặp phối" làm 1 phần tử mảng.
-- Chỉ cập nhật 15 sản phẩm có chi tiết màu mới; 9 sản phẩm vẫn chỉ "đa sắc"
-- (id 3,5,9,10,11,22,52,62,108) được giữ nguyên.

SET client_encoding TO 'UTF8';

BEGIN;

-- Cặp phối 2in1 (trắng+xanh dương, ...)
UPDATE products SET colors = ARRAY['trắng + xanh dương','trắng + vàng','tím + hồng','đỏ + vàng','đỏ + xanh lá','xanh dương + xanh lá'] WHERE id = 1;
UPDATE products SET colors = ARRAY['trắng + xanh dương','trắng + vàng','tím + hồng','đỏ + vàng','đỏ + xanh lá','xanh dương + xanh lá'] WHERE id = 4;
UPDATE products SET colors = ARRAY['trắng + xanh dương','trắng + vàng','tím + hồng','đỏ + vàng','đỏ + xanh lá','xanh dương + xanh lá'] WHERE id = 12;

-- Vòng tay đá mã não bện dây (9 màu)
UPDATE products SET colors = ARRAY['tím','đỏ','xanh aqua','xanh ngọc bích','trắng','vàng','tourmaline','xanh rêu','trầm hương'] WHERE id IN (13,14,16,18,25,38,39,40,41);

-- Vòng tay đá mã não theo mệnh
UPDATE products SET colors = ARRAY['xanh lá','xanh dương','đỏ','vàng','trắng'] WHERE id = 27;

-- Vòng tay trầm hương tự nhiên bện dây
UPDATE products SET colors = ARRAY['nâu'] WHERE id = 29;

-- Vòng tay chỉ ngũ sắc
UPDATE products SET colors = ARRAY['xanh','vàng'] WHERE id = 96;

COMMIT;

-- Kết quả sau cập nhật
SELECT id, colors FROM products WHERE id IN (1,4,12,13,14,16,18,25,38,39,40,41,27,29,96) ORDER BY id;
