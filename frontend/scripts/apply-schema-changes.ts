import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

// Construct postgres connection string from Supabase URL
// Format: postgresql://postgres.[ref]:[password]@aws-0-us-west-1.pooler.supabase.com:6543/postgres
const ref = supabaseUrl.replace('https://', '').replace('.supabase.co', '');
const connectionString = `postgresql://postgres.${ref}:${supabaseKey}@aws-0-us-west-1.pooler.supabase.com:6543/postgres`;

console.log('🔄 Applying schema changes to projects table...\n');

// For now, let's just use raw SQL via a TypeScript approach
// We'll create a simple schema modification script

async function main() {
  const { Client } = await import('pg');
  const client = new Client({ connectionString });

  try {
    await client.connect();
    console.log('✓ Connected to database\n');

    // Remove old columns
    console.log('🗑️  Removing resource_estimate, reserve_estimate, ownership_percentage...');
    await client.query('ALTER TABLE projects DROP COLUMN IF EXISTS resource_estimate');
    await client.query('ALTER TABLE projects DROP COLUMN IF EXISTS reserve_estimate');
    await client.query('ALTER TABLE projects DROP COLUMN IF EXISTS ownership_percentage');
    console.log('✓ Removed old columns\n');

    // Add new columns
    console.log('➕ Adding operator, production_rate, mine_life, capex, first_production, project_type...');
    await client.query('ALTER TABLE projects ADD COLUMN IF NOT EXISTS operator TEXT');
    await client.query('ALTER TABLE projects ADD COLUMN IF NOT EXISTS production_rate TEXT');
    await client.query('ALTER TABLE projects ADD COLUMN IF NOT EXISTS mine_life TEXT');
    await client.query('ALTER TABLE projects ADD COLUMN IF NOT EXISTS capex TEXT');
    await client.query('ALTER TABLE projects ADD COLUMN IF NOT EXISTS first_production DATE');
    await client.query('ALTER TABLE projects ADD COLUMN IF NOT EXISTS project_type TEXT');
    console.log('✓ Added new columns\n');

    // Create indexes
    console.log('📇 Creating indexes...');
    await client.query('CREATE INDEX IF NOT EXISTS idx_projects_operator ON projects(operator)');
    await client.query('CREATE INDEX IF NOT EXISTS idx_projects_project_type ON projects(project_type)');
    console.log('✓ Created indexes\n');

    console.log('✅ Schema migration complete!');
  } catch (error: any) {
    console.error('❌ Error:', error.message);
  } finally {
    await client.end();
  }
}

main();
