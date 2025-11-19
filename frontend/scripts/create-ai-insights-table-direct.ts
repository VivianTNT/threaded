import { Client } from 'pg';
import * as fs from 'fs';
import * as path from 'path';

// Parse Supabase URL to get connection details
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl) {
  console.error('❌ Missing NEXT_PUBLIC_SUPABASE_URL');
  process.exit(1);
}

// Extract project ref from URL (e.g., https://dfxauievbyqwcynwtvib.supabase.co)
const projectRef = supabaseUrl.replace('https://', '').replace('.supabase.co', '');

// Supabase connection pooler uses port 6543 for transactions
const connectionString = `postgresql://postgres.${projectRef}:${process.env.SUPABASE_DB_PASSWORD || '[YOUR-PASSWORD]'}@aws-0-us-west-1.pooler.supabase.com:6543/postgres`;

async function createAIInsightsTable() {
  console.log('📦 Creating AI Insights table using direct Postgres connection...\n');

  // Check if we have DB password
  if (!process.env.SUPABASE_DB_PASSWORD) {
    console.log('⚠️  Database password not found in environment variables.\n');
    console.log('📋 **MANUAL MIGRATION REQUIRED**\n');
    console.log('Please follow these steps:\n');
    console.log('1. Go to https://supabase.com/dashboard/project/' + projectRef);
    console.log('2. Navigate to: SQL Editor (left sidebar)');
    console.log('3. Click "New query"');
    console.log('4. Copy and paste the SQL below:');
    console.log('─'.repeat(80));

    const migrationPath = path.join(__dirname, '../supabase/migrations/011_create_ai_insights_table.sql');
    const migrationSQL = fs.readFileSync(migrationPath, 'utf8');
    console.log(migrationSQL);
    console.log('─'.repeat(80));
    console.log('\n5. Click "Run" or press Cmd+Enter (Mac) / Ctrl+Enter (Windows)');
    console.log('6. Verify success message appears\n');
    console.log('✅ Once complete, restart your dev server and test the AI Insights tab!\n');
    return;
  }

  const client = new Client({
    connectionString,
    ssl: { rejectUnauthorized: false }
  });

  try {
    await client.connect();
    console.log('✅ Connected to Supabase database\n');

    // Read and execute migration
    const migrationPath = path.join(__dirname, '../supabase/migrations/011_create_ai_insights_table.sql');
    const migrationSQL = fs.readFileSync(migrationPath, 'utf8');

    console.log('📝 Executing migration...');
    await client.query(migrationSQL);

    console.log('✅ AI Insights table created successfully!\n');
    console.log('📊 Table structure:');
    console.log('   - geography_risk_score (0-10)');
    console.log('   - legal_risk_score (0-10)');
    console.log('   - commodity_risk_score (0-10)');
    console.log('   - team_risk_score (0-10)');
    console.log('   - overall_risk_score (0-10)');
    console.log('   - investment_recommendation (Strong Buy/Buy/Hold/Pass)');
    console.log('   - 7-day caching for performance\n');

    console.log('🎉 Setup complete! You can now:');
    console.log('   1. Open any project in your app');
    console.log('   2. Click the "AI Insights" tab');
    console.log('   3. Watch AI generate comprehensive risk analysis!\n');

  } catch (error: any) {
    console.error('❌ Migration failed:', error.message);
    console.log('\n📋 **MANUAL MIGRATION REQUIRED**\n');
    console.log('Please follow these steps:\n');
    console.log('1. Go to https://supabase.com/dashboard/project/' + projectRef);
    console.log('2. Navigate to: SQL Editor (left sidebar)');
    console.log('3. Click "New query"');
    console.log('4. Copy the contents of: supabase/migrations/011_create_ai_insights_table.sql');
    console.log('5. Paste into the SQL Editor');
    console.log('6. Click "Run" or press Cmd+Enter (Mac) / Ctrl+Enter (Windows)\n');
  } finally {
    await client.end();
  }
}

createAIInsightsTable();
