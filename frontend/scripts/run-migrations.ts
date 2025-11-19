import { createClient } from '@supabase/supabase-js';
import { readFileSync } from 'fs';
import { join } from 'path';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function runMigration(filename: string) {
  const sql = readFileSync(join(__dirname, '..', 'supabase', 'migrations', filename), 'utf-8');

  console.log(`\n🚀 Running migration: ${filename}`);

  const { data, error } = await supabase.rpc('exec_sql', { sql_string: sql });

  if (error) {
    console.error(`❌ Error in ${filename}:`, error);
    throw error;
  }

  console.log(`✅ Successfully ran ${filename}`);
}

async function main() {
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
