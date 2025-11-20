/**
 * Migration Script 010: Add Financial Metrics to Projects Table
 * Adds NPV, IRR, and CAPEX columns to the projects table
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// Load environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing required environment variables');
  console.error('NEXT_PUBLIC_SUPABASE_URL:', supabaseUrl ? '✓' : '✗');
  console.error('SUPABASE_SERVICE_ROLE_KEY:', supabaseServiceKey ? '✓' : '✗');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function runMigration() {
  console.log('🚀 Starting Migration 010: Add Financial Metrics\n');

  try {
    // Read the migration file
    const migrationPath = path.join(__dirname, '../supabase/migrations/010_add_financial_metrics.sql');
    const sql = fs.readFileSync(migrationPath, 'utf-8');

    console.log('📄 Migration SQL loaded from:', migrationPath);
    console.log('\n--- Migration SQL ---');
    console.log(sql);
    console.log('--- End Migration SQL ---\n');

    // Split the SQL into individual statements (rough split on semicolons)
    const statements = sql
      .split(';')
      .map(s => s.trim())
      .filter(s => s && !s.startsWith('--') && s.length > 0);

    console.log(`📝 Found ${statements.length} SQL statements to execute\n`);

    // Execute each statement
    let successCount = 0;
    for (let i = 0; i < statements.length; i++) {
      const statement = statements[i];
      console.log(`\n[${i + 1}/${statements.length}] Executing statement...`);
      console.log(statement.substring(0, 100) + (statement.length > 100 ? '...' : ''));

      try {
        const { error } = await supabase.rpc('exec_sql', { sql_query: statement + ';' });

        if (error) {
          // Try direct execution via REST API as fallback
          console.log('⚠️  RPC failed, trying direct execution...');
          const response = await fetch(`${supabaseUrl}/rest/v1/rpc/exec`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'apikey': supabaseServiceKey,
              'Authorization': `Bearer ${supabaseServiceKey}`,
            },
            body: JSON.stringify({ query: statement + ';' })
          });

          if (!response.ok) {
            console.error(`❌ Statement ${i + 1} failed:`, error);
            console.log('⏭️  Continuing with next statement...\n');
          } else {
            successCount++;
            console.log(`✓ Statement ${i + 1} executed successfully`);
          }
        } else {
          successCount++;
          console.log(`✓ Statement ${i + 1} executed successfully`);
        }
      } catch (err) {
        console.error(`❌ Error executing statement ${i + 1}:`, err);
        console.log('⏭️  Continuing with next statement...\n');
      }
    }

    console.log(`\n\n✅ Migration completed: ${successCount}/${statements.length} statements executed successfully\n`);

    // Verify the columns were added
    console.log('🔍 Verifying columns were added...\n');
    const { data: columns, error: verifyError } = await supabase
      .from('projects')
      .select('*')
      .limit(1);

    if (verifyError) {
      console.error('❌ Error verifying columns:', verifyError);
    } else if (columns && columns.length > 0) {
      const project = columns[0];
      console.log('✓ Sample project columns:', Object.keys(project));

      const expectedCols = ['npv_usd_millions', 'irr_percentage', 'capex_usd_millions', 'discount_rate_percentage', 'financial_metrics_updated_at'];
      const missingCols = expectedCols.filter(col => !(col in project));

      if (missingCols.length === 0) {
        console.log('✓ All expected columns are present!');
      } else {
        console.warn('⚠️  Missing columns:', missingCols);
      }
    } else {
      console.log('⚠️  No projects in table to verify columns');
    }

  } catch (error) {
    console.error('❌ Migration failed:', error);
    process.exit(1);
  }
}

// Run the migration
runMigration()
  .then(() => {
    console.log('\n✅ Migration script completed successfully');
    process.exit(0);
  })
  .catch(error => {
    console.error('\n❌ Migration script failed:', error);
    process.exit(1);
  });
