#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: DATABASE_URL is not set."
  echo "Example: export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/npc_engine"
  exit 1
fi

echo "Applying schema migration..."
psql "$DATABASE_URL" -f database/migrations/001_init_shop_assistant.sql

echo "Seeding sample data..."
psql "$DATABASE_URL" -f database/seeds/001_seed_sample_data.sql

echo "Done. Database schema + seed are ready."
