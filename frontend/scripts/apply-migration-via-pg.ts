/**
 * Apply Migration 010 via PostgreSQL direct connection
 * Uses node-postgres (pg) library to execute raw SQL
 */

import { Pool } from 'pg';
import * as fs from 'fs';
import * as path from 'path';

// Construct PostgreSQL connection string from Supabase details
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const password = process.env.DATABASE_PASSWORD || '';

if (!supabaseUrl) {
  console.error('❌ NEXT_PUBLIC_SUPABASE_URL not found');
  process.exit(1);
}

// Extract project reference from Supabase URL
// Format: https://PROJECT_REF.supabase.co
const projectRef = supabaseUrl.replace('https://', '').split('.')[0];

// Construct direct PostgreSQL connection string
const connectionString = `postgresql://postgres:${password}@db.${projectRef}.supabase.co:5432/postgres`;

console.log('🔗 Connecting to PostgreSQL...');
console.log('Project Ref:', projectRef);

const pool = new Pool({
  connectionString,
  ssl: { rejectUnauthorized: false }
});

async function runMigration() {
  const client = await pool.connect();

  try {
    console.log('✓ Connected to database\n');
    console.log('🚀 Starting Migration 010: Add Financial Metrics\n');

    // Read the migration file
    const migrationPath = path.join(__dirname, '../supabase/migrations/010_add_financial_metrics.sql');
    const sql = fs.readFileSync(migrationPath, 'utf-8');

    console.log('📄 Executing migration SQL...\n');

    // Execute the entire migration
    await client.query(sql);

    console.log('✅ Migration executed successfully!\n');

    // Verify the columns were added
    console.log('🔍 Verifying columns...\n');

    const result = await client.query(`
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name = 'projects'
      AND column_name IN (
        'npv_usd_millions',
        'irr_percentage',
        'capex_usd_millions',
        'discount_rate_percentage',
        'financial_metrics_updated_at'
      )
      ORDER BY column_name;
    `);

    if (result.rows.length === 5) {
      console.log('✓ All 5 financial metric columns verified:');
      result.rows.forEach(row => {
        console.log(`  - ${row.column_name} (${row.data_type})`);
      });
      console.log('\n✅ Migration completed successfully!');
    } else {
      console.warn(`⚠️  Only ${result.rows.length}/5 columns found:`);
      result.rows.forEach(row => {
        console.log(`  - ${row.column_name}`);
      });
    }

    return true;

  } catch (error: any) {
    console.error('❌ Migration failed:', error.message);
    throw error;
  } finally {
    client.release();
    await pool.end();
  }
}

// Run the migration
runMigration()
  .then(() => {
    console.log('\n✅ Script completed');
    process.exit(0);
  })
  .catch(error => {
    console.error('\n❌ Script failed:', error);
    process.exit(1);
  });
