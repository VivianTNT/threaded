import { readFileSync } from 'fs';
import { join } from 'path';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

async function executeSql(sql: string) {
  const response = await fetch(`${supabaseUrl}/rest/v1/rpc/exec`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'apikey': supabaseServiceKey,
      'Authorization': `Bearer ${supabaseServiceKey}`,
    },
    body: JSON.stringify({ query: sql }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HTTP ${response.status}: ${error}`);
  }

  return response.json();
}

async function runMigration(filename: string) {
  const sql = readFileSync(join(__dirname, '..', 'supabase', 'migrations', filename), 'utf-8');

  console.log(`\n🚀 Running migration: ${filename}`);

  try {
    // Split by semicolons and execute each statement
    const statements = sql
      .split(';')
      .map(s => s.trim())
      .filter(s => s.length > 0 && !s.startsWith('--'));

    for (const statement of statements) {
      await executeSql(statement);
    }

    console.log(`✅ Successfully ran ${filename}`);
  } catch (error) {
    console.error(`❌ Error in ${filename}:`, error);
    throw error;
  }
}

async function main() {
  console.log('Starting migrations...');
  console.log('Supabase URL:', supabaseUrl);

  try {
    await runMigration('006_create_companies_table.sql');
    await runMigration('007_create_projects_table.sql');
    await runMigration('008_create_news_table.sql');

    console.log('\n✨ All migrations completed successfully!');
  } catch (error) {
    console.error('\n❌ Migration failed:', error);
    process.exit(1);
  }
}

main();
