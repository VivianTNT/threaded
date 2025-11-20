import pg from 'pg';
import { readFileSync } from 'fs';
import { join } from 'path';

const { Client } = pg;

// Construct PostgreSQL connection string from Supabase credentials
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

// Extract project ref from URL (dfxauievbyqwcynwtvib from https://dfxauievbyqwcynwtvib.supabase.co)
const projectRef = supabaseUrl.replace('https://', '').split('.')[0];

// Supabase direct database connection
const connectionString = `postgresql://postgres:${supabaseKey}@db.${projectRef}.supabase.co:5432/postgres`;

async function runMigration(filename: string, client: pg.Client) {
  const sql = readFileSync(join(__dirname, '..', 'supabase', 'migrations', filename), 'utf-8');

  console.log(`\n🚀 Running migration: ${filename}`);

  try {
    await client.query(sql);
    console.log(`✅ Successfully ran ${filename}`);
  } catch (error: any) {
    console.error(`❌ Error in ${filename}:`, error.message);
    throw error;
  }
}

async function main() {
  const client = new Client({ connectionString });

  try {
    console.log('Connecting to database...');
    await client.connect();
    console.log('✅ Connected!');

    await runMigration('006_create_companies_table.sql', client);
    await runMigration('007_create_projects_table.sql', client);
    await runMigration('008_create_news_table.sql', client);

    console.log('\n✨ All migrations completed successfully!');
  } catch (error) {
    console.error('\n❌ Migration failed:', error);
    process.exit(1);
  } finally {
    await client.end();
  }
}

main();
