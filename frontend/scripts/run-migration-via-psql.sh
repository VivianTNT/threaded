#!/bin/bash

# Supabase connection details
PROJECT_REF="dfxauievbyqwcynwtvib"
DB_PASSWORD="${SUPABASE_DB_PASSWORD:-}"

# Check if password is provided
if [ -z "$DB_PASSWORD" ]; then
  echo "❌ Database password not provided"
  echo ""
  echo "To run this migration, you need the database password."
  echo ""
  echo "Get it from Supabase Dashboard:"
  echo "1. Go to: https://supabase.com/dashboard/project/$PROJECT_REF/settings/database"
  echo "2. Look for 'Database password' or 'Connection string'"
  echo "3. Run this script with: SUPABASE_DB_PASSWORD='your-password' ./scripts/run-migration-via-psql.sh"
  echo ""
  echo "Alternatively, you can use the connection pooler:"
  echo "Connection string format: postgresql://postgres.[project-ref]:[password]@aws-0-us-west-1.pooler.supabase.com:6543/postgres"
  exit 1
fi

# Connection string for transaction pooler (port 6543)
CONNECTION_STRING="postgresql://postgres.${PROJECT_REF}:${DB_PASSWORD}@aws-0-us-west-1.pooler.supabase.com:6543/postgres"

echo "🚀 Running migration via direct PostgreSQL connection..."
echo ""

# Check if psql is installed
if ! command -v psql &> /dev/null; then
  echo "❌ psql is not installed"
  echo ""
  echo "Install PostgreSQL client:"
  echo "  macOS: brew install postgresql"
  echo "  Ubuntu: sudo apt-get install postgresql-client"
  echo ""
  exit 1
fi

# Run the migration
psql "$CONNECTION_STRING" -f supabase/migrations/20250124_add_user_upload_support.sql

if [ $? -eq 0 ]; then
  echo ""
  echo "✅ Migration completed successfully!"
  echo ""
  echo "Verify with:"
  echo "  npx tsx scripts/check-migration-status.ts"
else
  echo ""
  echo "❌ Migration failed"
  echo ""
  echo "Try running the SQL manually in Supabase Dashboard:"
  echo "  https://supabase.com/dashboard/project/$PROJECT_REF/sql/new"
fi
