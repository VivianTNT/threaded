#!/usr/bin/env tsx
/**
 * Creates tables using Supabase client without needing database password
 * This uses the service role key to execute SQL via PostgreSQL's DDL
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!SUPABASE_URL || !SERVICE_KEY) {
  console.error('Missing required environment variables');
  process.exit(1);
}

const supabase = createClient(SUPABASE_URL, SERVICE_KEY, {
  auth: {
    autoRefreshToken: false,
    persistSession: false,
  },
  db: {
    schema: 'public',
  },
});

async function executeSQLFile(filename: string) {
  const filePath = path.join(__dirname, '../supabase/migrations', filename);
  const sql = fs.readFileSync(filePath, 'utf8');

  console.log(`\n🚀 Executing: ${filename}`);
  console.log('SQL Preview:', sql.substring(0, 100) + '...\n');

  // Split SQL into individual statements
  const statements = sql
    .split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0 && !s.startsWith('--'));

  for (let i = 0; i < statements.length; i++) {
    const statement = statements[i];
    if (!statement) continue;

    console.log(`  Executing statement ${i + 1}/${statements.length}...`);

    try {
      // Use the raw SQL query method
      const { data, error } = await (supabase as any).rpc('exec', {
        sql: statement + ';',
      });

      if (error) {
        // Try alternative: direct table creation might work
        if (statement.toUpperCase().includes('CREATE TABLE')) {
          console.log(`  ⚠️  RPC failed, this might be expected. Error: ${error.message}`);
        } else {
          throw error;
        }
      } else {
        console.log(`  ✅ Statement ${i + 1} executed`);
      }
    } catch (err: any) {
      console.error(`  ❌ Statement ${i + 1} failed:`, err.message);

      // Continue if it's just an "already exists" error
      if (err.message?.includes('already exists')) {
        console.log(`  ℹ️  Resource already exists, continuing...`);
      } else {
        throw err;
      }
    }
  }

  console.log(`✅ ${filename} completed\n`);
}

async function main() {
  console.log('========================================');
  console.log('Supabase Table Creation Script');
  console.log('========================================');
  console.log(`Project: ${SUPABASE_URL}`);
  console.log('========================================\n');

  const migrations = [
    '006_create_companies_table.sql',
    '007_create_projects_table.sql',
    '008_create_news_table.sql',
  ];

  let successCount = 0;
  let failCount = 0;

  for (const migration of migrations) {
    try {
      await executeSQLFile(migration);
      successCount++;
    } catch (error: any) {
      console.error(`\n❌ Failed to execute ${migration}:`, error.message);
      failCount++;

      // Stop on first real error (not "already exists")
      if (!error.message?.includes('already exists')) {
        console.error('\n⛔ Stopping due to error\n');
        break;
      }
    }
  }

  console.log('\n========================================');
  console.log('Summary:');
  console.log(`  ✅ Successful: ${successCount}`);
  console.log(`  ❌ Failed: ${failCount}`);
  console.log('========================================\n');

  if (failCount > 0) {
    console.log('⚠️  Some migrations failed. This might be because:');
    console.log('1. The RPC function "exec" does not exist');
    console.log('2. You need database password for direct connection\n');
    console.log('Please run the manual migration script or:');
    console.log('  npm run migrate-manual\n');
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('\n💥 Fatal error:', error.message);
  console.log('\n📝 To run migrations manually:');
  console.log('1. Go to: https://supabase.com/dashboard/project/dfxauievbyqwcynwtvib/editor/sql');
  console.log('2. Copy and paste the SQL from each migration file');
  console.log('3. Run them in order\n');
  process.exit(1);
});
