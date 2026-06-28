-- 006_create_promotions.sql
-- Bảng khuyến mãi của shop theo ngày (Gap 2).
-- Quy ước: mỗi tháng có 1 ngày sale "trùng số" (1/1, 2/2, ... 12/12) + một số
-- ngày lễ (30/4, 1/5, 2/9). discount_percent nằm trong khoảng 10-20%.
-- promotion_info là câu mô tả chatbot có thể đọc trực tiếp.

SET client_encoding TO 'UTF8';

CREATE TABLE IF NOT EXISTS promotions (
    id               SERIAL       PRIMARY KEY,
    promo_date       VARCHAR(10)  NOT NULL UNIQUE,   -- "d/m" để hiển thị
    day              INT          NOT NULL,
    month            INT          NOT NULL,
    discount_percent INT          NOT NULL,
    scope            VARCHAR(50)  NOT NULL DEFAULT 'mọi sản phẩm',
    promotion_info   TEXT         NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_promotions_month ON promotions (month);

-- Seed (idempotent: xoá trước rồi nạp lại để chạy nhiều lần không nhân đôi)
TRUNCATE promotions RESTART IDENTITY;

INSERT INTO promotions (promo_date, day, month, discount_percent, scope, promotion_info) VALUES
  ('1/1',   1,  1,  11, 'mọi sản phẩm',          'Sale Tết Dương 1/1: giảm 11% mọi sản phẩm'),
  ('2/2',   2,  2,  15, 'mọi sản phẩm',          'Sale 2/2: giảm 15% mọi sản phẩm'),
  ('3/3',   3,  3,  12, 'mọi sản phẩm',          'Sale 3/3: giảm 12% mọi sản phẩm'),
  ('8/3',   8,  3,  18, 'mọi sản phẩm',          'Sale Quốc tế Phụ nữ 8/3: giảm 18% mọi sản phẩm'),
  ('4/4',   4,  4,  13, 'mọi sản phẩm',          'Sale 4/4: giảm 13% mọi sản phẩm'),
  ('30/4', 30,  4,  18, 'mọi sản phẩm',          'Sale lễ 30/4: giảm 18% mọi sản phẩm'),
  ('1/5',   1,  5,  15, 'mọi sản phẩm',          'Sale lễ 1/5: giảm 15% mọi sản phẩm'),
  ('5/5',   5,  5,  14, 'mọi sản phẩm',          'Sale 5/5: giảm 14% mọi sản phẩm'),
  ('6/6',   6,  6,  17, 'mọi sản phẩm vòng tay', 'Sale 6/6: giảm 17% mọi sản phẩm vòng tay'),
  ('7/7',   7,  7,  20, 'mọi sản phẩm',          'Sale 7/7: giảm 20% mọi sản phẩm'),
  ('8/8',   8,  8,  13, 'mọi sản phẩm',          'Sale 8/8: giảm 13% mọi sản phẩm'),
  ('9/9',   9,  9,  19, 'mọi sản phẩm',          'Sale 9/9: giảm 19% mọi sản phẩm'),
  ('2/9',   2,  9,  16, 'mọi sản phẩm',          'Sale Quốc khánh 2/9: giảm 16% mọi sản phẩm'),
  ('10/10',10, 10,  16, 'mọi sản phẩm',          'Sale 10/10: giảm 16% mọi sản phẩm'),
  ('20/10',20, 10,  18, 'mọi sản phẩm',          'Sale Phụ nữ VN 20/10: giảm 18% mọi sản phẩm'),
  ('11/11',11, 11,  20, 'mọi sản phẩm',          'Sale 11/11: giảm 20% mọi sản phẩm'),
  ('12/12',12, 12,  20, 'mọi sản phẩm',          'Sale 12/12: giảm 20% mọi sản phẩm');

SELECT promo_date, discount_percent, scope FROM promotions ORDER BY month, day;
