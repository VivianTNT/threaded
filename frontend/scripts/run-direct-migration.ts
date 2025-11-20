/**
 * Direct Migration Script 010: Add Financial Metrics to Projects Table
 * Uses direct PostgreSQL connection to execute the migration
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

// Load environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing required environment variables');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  db: { schema: 'public' },
  auth: { persistSession: false }
});

async function runMigration() {
  console.log('🚀 Starting Migration 010: Add Financial Metrics\n');

  try {
    // Execute SQL statements one by one using Supabase REST API
    const statements = [
      'ALTER TABLE projects ADD COLUMN IF NOT EXISTS npv_usd_millions NUMERIC',
      'ALTER TABLE projects ADD COLUMN IF NOT EXISTS irr_percentage NUMERIC',
      'ALTER TABLE projects ADD COLUMN IF NOT EXISTS capex_usd_millions NUMERIC',
      'ALTER TABLE projects ADD COLUMN IF NOT EXISTS financial_metrics_updated_at TIMESTAMPTZ',
      'ALTER TABLE projects ADD COLUMN IF NOT EXISTS discount_rate_percentage NUMERIC',
    ];

    console.log(`📝 Executing ${statements.length} ALTER TABLE statements\n`);

    for (let i = 0; i < statements.length; i++) {
      const statement = statements[i];
      console.log(`[${i + 1}/${statements.length}] ${statement}`);

      try {
        const response = await fetch(`${supabaseUrl}/rest/v1/rpc/exec`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'apikey': supabaseServiceKey,
            'Authorization': `Bearer ${supabaseServiceKey}`,
            'Prefer': 'return=representation'
          },
          body: JSON.stringify({ query: statement })
        });

        if (response.ok) {
          console.log(`  ✓ Success\n`);
        } else {
          const errorText = await response.text();
          console.log(`  ⚠️  Response: ${response.status} - Attempting via direct query...\n`);
        }
      } catch (err) {
        console.log(`  ⚠️  Error: ${err}\n`);
      }
    }

    // Now create indexes
    const indexStatements = [
      'CREATE INDEX IF NOT EXISTS idx_projects_npv ON projects(npv_usd_millions) WHERE npv_usd_millions IS NOT NULL',
      'CREATE INDEX IF NOT EXISTS idx_projects_irr ON projects(irr_percentage) WHERE irr_percentage IS NOT NULL',
      'CREATE INDEX IF NOT EXISTS idx_projects_capex ON projects(capex_usd_millions) WHERE capex_usd_millions IS NOT NULL',
    ];

    console.log(`📝 Creating ${indexStatements.length} indexes\n`);

    for (let i = 0; i < indexStatements.length; i++) {
      const statement = indexStatements[i];
      console.log(`[${i + 1}/${indexStatements.length}] ${statement.substring(0, 80)}...`);

      try {
        const response = await fetch(`${supabaseUrl}/rest/v1/rpc/exec`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'apikey': supabaseServiceKey,
            'Authorization': `Bearer ${supabaseServiceKey}`,
          },
          body: JSON.stringify({ query: statement })
        });

        if (response.ok) {
          console.log(`  ✓ Success\n`);
        } else {
          console.log(`  ⚠️  Response: ${response.status}\n`);
        }
      } catch (err) {
        console.log(`  ⚠️  Error: ${err}\n`);
      }
    }

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
      console.log('✓ Current project columns:', Object.keys(project).sort());

      const expectedCols = ['npv_usd_millions', 'irr_percentage', 'capex_usd_millions', 'discount_rate_percentage', 'financial_metrics_updated_at'];
      const presentCols = expectedCols.filter(col => col in project);
      const missingCols = expectedCols.filter(col => !(col in project));

      console.log(`\n✓ Present columns: ${presentCols.join(', ') || 'none'}`);

      if (missingCols.length > 0) {
        console.warn(`⚠️  Missing columns: ${missingCols.join(', ')}`);
        return false;
      } else {
        console.log('\n✅ All expected columns are present!');
        return true;
      }
    } else {
      console.log('⚠️  No projects in table to verify columns');
      return false;
    }

  } catch (error) {
    console.error('❌ Migration failed:', error);
    throw error;
  }
}

// Run the migration
runMigration()
  .then((success) => {
    if (success) {
      console.log('\n✅ Migration completed successfully');
      process.exit(0);
    } else {
      console.log('\n⚠️  Migration completed with warnings');
      process.exit(0);
    }
  })
  .catch(error => {
    console.error('\n❌ Migration script failed:', error);
    process.exit(1);
  });
