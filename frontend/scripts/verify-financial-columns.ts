/**
 * Verify the financial metrics columns were added successfully
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function verifyColumns() {
  console.log('🔍 Verifying financial metrics columns in projects table...\n');

  try {
    // Fetch one project to check the schema
    const { data: projects, error } = await supabase
      .from('projects')
      .select('*')
      .limit(1);

    if (error) {
      console.error('❌ Error querying projects:', error);
      process.exit(1);
    }

    if (!projects || projects.length === 0) {
      console.log('⚠️  No projects found in database');
      console.log('📊 Total projects: 0\n');
    } else {
      const project = projects[0];
      const allColumns = Object.keys(project).sort();

      console.log('✓ Projects table columns:', allColumns.length);
      console.log(allColumns.join(', '));
      console.log();

      // Check for financial metric columns
      const financialColumns = [
        'npv_usd_millions',
        'irr_percentage',
        'capex_usd_millions',
        'discount_rate_percentage',
        'financial_metrics_updated_at'
      ];

      console.log('💰 Financial Metric Columns:');
      financialColumns.forEach(col => {
        const present = col in project;
        const status = present ? '✓' : '✗';
        console.log(`  ${status} ${col}`);
      });

      const allPresent = financialColumns.every(col => col in project);

      if (allPresent) {
        console.log('\n✅ All financial metric columns are present!');
        console.log('\nColumns available for use:');
        console.log('  - npv_usd_millions (NUMERIC) - Net Present Value in millions USD');
        console.log('  - irr_percentage (NUMERIC) - Internal Rate of Return as percentage');
        console.log('  - capex_usd_millions (NUMERIC) - Capital Expenditure in millions USD');
        console.log('  - discount_rate_percentage (NUMERIC) - Discount rate for NPV calculation');
        console.log('  - financial_metrics_updated_at (TIMESTAMPTZ) - Last update timestamp');
      } else {
        console.log('\n⚠️  Some columns are missing!');
      }
    }

    // Get total project count
    const { count } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true });

    console.log(`\n📊 Total projects in database: ${count || 0}`);

  } catch (error) {
    console.error('❌ Verification failed:', error);
    process.exit(1);
  }
}

verifyColumns()
  .then(() => {
    console.log('\n✅ Verification complete');
    process.exit(0);
  })
  .catch(error => {
    console.error('\n❌ Verification failed:', error);
    process.exit(1);
  });
