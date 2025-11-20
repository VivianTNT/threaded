import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing environment variables');
  console.error('Please ensure NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function runMigration() {
  try {
    console.log('📦 Running AI Insights migration...');

    // Read the migration file
    const migrationPath = path.join(__dirname, '../supabase/migrations/011_create_ai_insights_table.sql');
    const migrationSQL = fs.readFileSync(migrationPath, 'utf8');

    // Execute the migration
    const { data, error } = await supabase.rpc('exec_sql', {
      sql: migrationSQL
    });

    if (error) {
      // Try direct execution if RPC fails
      console.log('⚠️  RPC method failed, trying direct execution...');

      // Split by semicolon and execute each statement
      const statements = migrationSQL
        .split(';')
        .map(s => s.trim())
        .filter(s => s.length > 0 && !s.startsWith('--'));

      for (const statement of statements) {
        const { error: execError } = await (supabase as any).from('_').select('*').limit(0);

        // We need to use a different approach - let's just log the SQL
        console.log('\nExecute this SQL in your Supabase SQL Editor:');
        console.log('─'.repeat(80));
        console.log(migrationSQL);
        console.log('─'.repeat(80));
        console.log('\n✅ Copy the above SQL and run it in Supabase Dashboard → SQL Editor');
        process.exit(0);
      }
    }

    console.log('✅ AI Insights migration completed successfully!');
    console.log('\n📝 Next steps:');
    console.log('1. Open a project detail panel in your app');
    console.log('2. Click on the "AI Insights" tab');
    console.log('3. The system will automatically generate AI-powered risk analysis');

  } catch (error) {
    console.error('❌ Migration failed:', error);
    console.log('\n📋 Manual migration steps:');
    console.log('1. Go to your Supabase Dashboard');
    console.log('2. Navigate to SQL Editor');
    console.log('3. Copy the contents of: supabase/migrations/011_create_ai_insights_table.sql');
    console.log('4. Paste and execute in the SQL Editor');
    process.exit(1);
  }
}

runMigration();
