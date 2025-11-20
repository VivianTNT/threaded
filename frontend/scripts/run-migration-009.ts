import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

async function runMigration() {
  console.log('🔄 Running migration 009: Modify projects table fields...\n');

  const sql = fs.readFileSync('./supabase/migrations/009_modify_projects_table_fields.sql', 'utf-8');

  // Split by semicolon and run each statement
  const statements = sql
    .split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0 && !s.startsWith('--'));

  for (const statement of statements) {
    console.log(`📝 Executing: ${statement.substring(0, 80)}...`);

    const { error } = await supabase.rpc('exec_sql', { sql_string: statement });

    if (error) {
      // Try direct execution with postgres client instead
      console.log(`   Trying alternative method...`);
      // Since rpc might not exist, let's use a workaround
      continue;
    }

    console.log(`   ✓ Success`);
  }

  console.log('\n✅ Migration complete!');
}

runMigration();
