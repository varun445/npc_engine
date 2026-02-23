-- Seed data for demo/testing.

BEGIN;

-- Users
INSERT INTO users (id, full_name, email, role, password_hash)
VALUES
    ('11111111-1111-1111-1111-111111111111', 'Amina Storeowner', 'owner@smartshop.com', 'owner', 'demo_hash_owner'),
    ('22222222-2222-2222-2222-222222222222', 'Daniel Staff', 'staff@smartshop.com', 'staff', 'demo_hash_staff'),
    ('33333333-3333-3333-3333-333333333333', 'Layla Customer', 'customer@smartshop.com', 'customer', NULL)
ON CONFLICT (email) DO NOTHING;

-- Products
INSERT INTO products (id, sku, name, category, price, description)
VALUES
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1', 'SKU-MILK-1L', 'Milk (1L)', 'dairy', 2.50, 'Fresh whole milk 1 liter'),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2', 'SKU-BREAD-WHITE', 'White Bread', 'bakery', 2.00, 'Soft white sandwich bread'),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa3', 'SKU-APPLE-RED', 'Apple (Red)', 'fruits', 1.00, 'Red apple sold per piece'),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa4', 'SKU-COLA-2L', 'Cola (2L)', 'beverages', 2.50, 'Carbonated soft drink 2 liters'),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa5', 'SKU-CHIPS-CLASSIC', 'Potato Chips', 'snacks', 2.50, 'Salted potato chips pack')
ON CONFLICT (sku) DO NOTHING;

-- Inventory snapshots
INSERT INTO inventory (product_id, quantity, reorder_level)
VALUES
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1', 15, 6),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2', 10, 5),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa3', 30, 10),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa4', 20, 8),
    ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa5', 25, 7)
ON CONFLICT (product_id) DO NOTHING;

-- Inventory movements history
INSERT INTO inventory_movements (id, product_id, actor_user_id, change_qty, reason, notes)
VALUES
    ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbb1', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1', '11111111-1111-1111-1111-111111111111', 20, 'restock', 'Morning supplier restock'),
    ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbb2', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2', '22222222-2222-2222-2222-222222222222', -2, 'sale', 'Sold during breakfast rush'),
    ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbb3', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa3', '22222222-2222-2222-2222-222222222222', -5, 'sale', 'Walk-in customer sales')
ON CONFLICT (id) DO NOTHING;

-- Conversations between customer and assistant
INSERT INTO conversations (id, user_id, session_id, role, message, metadata)
VALUES
    ('cccccccc-cccc-cccc-cccc-ccccccccccc1', '33333333-3333-3333-3333-333333333333', 'dddddddd-dddd-dddd-dddd-dddddddddddd', 'customer', 'Hi, do you have milk and bread?', '{"channel":"in_store_kiosk"}'),
    ('cccccccc-cccc-cccc-cccc-ccccccccccc2', '33333333-3333-3333-3333-333333333333', 'dddddddd-dddd-dddd-dddd-dddddddddddd', 'assistant', 'Yes, Milk (1L) is in stock and White Bread is available too.', '{"intent":"stock_check"}')
ON CONFLICT (id) DO NOTHING;

-- Sample order
INSERT INTO orders (id, customer_user_id, status, total_amount)
VALUES
    ('eeeeeeee-eeee-eeee-eeee-eeeeeeeeeee1', '33333333-3333-3333-3333-333333333333', 'confirmed', 4.50)
ON CONFLICT (id) DO NOTHING;

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price)
VALUES
    ('ffffffff-ffff-ffff-ffff-fffffffffff1', 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeee1', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1', 1, 2.50),
    ('ffffffff-ffff-ffff-ffff-fffffffffff2', 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeee1', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2', 1, 2.00)
ON CONFLICT (id) DO NOTHING;

COMMIT;
