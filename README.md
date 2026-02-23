# npc_engine

A prototype shop-assistant NPC project using `pygame` + local LLM.

## Database setup (PostgreSQL)

This project now includes a production-oriented PostgreSQL schema and seed data.

### Why PostgreSQL?
- Production-proven and widely used.
- Scales vertically/horizontally with mature tooling.
- Supports JSONB, indexing, transactions, and strong relational constraints.

### Files
- Migration: `database/migrations/001_init_shop_assistant.sql`
- Seed data: `database/seeds/001_seed_sample_data.sql`
- Setup script: `database/scripts/apply_db_setup.sh`

### Core tables
- `users` (owner/staff/customer)
- `products`
- `inventory`
- `inventory_movements`
- `conversations`
- `orders`
- `order_items`

### Apply schema + seed

1. Create a PostgreSQL database (example: `npc_engine`).
2. Set your connection string:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/npc_engine
```

3. Run:

```bash
./database/scripts/apply_db_setup.sh
```

This will create all relations and load sample shop data for development/demo.
