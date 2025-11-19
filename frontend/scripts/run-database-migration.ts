#!/usr/bin/env tsx

/**
 * Database Migration Script
 * This script runs the database simplification migration
 * Usage: npm run migrate:simplify
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.local' });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing Supabase credentials in .env.local');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function runMigration() {
  console.log('🚀 Starting database migration...\n');

  try {
    // Read the migration file
    const migrationPath = path.join(__dirname, '../supabase/migrations/001_initial_schema.sql');
    const migrationSQL = fs.readFileSync(migrationPath, 'utf-8');

    console.log('📋 Migration steps:');
    console.log('  1. Create companies table');
    console.log('  2. Create projects table');
    console.log('  3. Create news table');
    console.log('  4. Create usr table');
    console.log('  5. Create chat_history table');
    console.log('  6. Create junction tables');
    console.log('  7. Set up triggers and functions\n');

    // Backup warning
    console.log('⚠️  WARNING: This migration will modify your database structure.');
    console.log('   Make sure you have a backup before proceeding.\n');

    // Check current state
    console.log('📊 Checking current database state...');
    
    // Check if companies table exists
    const { data: companiesExists } = await supabase
      .from('companies')
      .select('id')
      .limit(1);
    
    if (companiesExists) {
      console.log('  ✓ Companies table exists');
    } else {
      console.log('  ○ Companies table will be created');
    }

    // Check if unified_news exists
    const { data: newsExists } = await supabase
      .from('unified_news')
      .select('id')
      .limit(1);
    
    if (newsExists) {
      console.log('  ✓ Unified_news table found - will migrate to simplified structure');
    }

    // Count projects
    const { count: projectCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true });
    
    console.log(`  ✓ Found ${projectCount} projects in database`);

    // Get unique companies from projects
    const { data: uniqueCompanies } = await supabase
      .from('projects')
      .select('company_name')
      .not('company_name', 'is', null);
    
    const companyNames = [...new Set(uniqueCompanies?.map(p => p.company_name) || [])];
    console.log(`  ✓ Found ${companyNames.length} unique companies in projects\n`);

    // Run the migration
    console.log('🔄 Running migration...');
    
    // Note: Supabase doesn't support running raw SQL directly via the client
    // You'll need to run this in the Supabase SQL editor or use the CLI
    
    console.log('\n📝 Migration SQL has been prepared.');
    console.log('   Please run the following file in your Supabase SQL editor:');
    console.log(`   ${migrationPath}\n`);
    
    console.log('🎯 After running the migration:');
    console.log('  1. Companies will be populated from projects');
    console.log('  2. Projects will be linked to companies via foreign key');
    console.log('  3. News table will have simplified structure');
    console.log('  4. Watchlist will be standardized across all tables');
    console.log('  5. User-specific watchlists will be supported\n');

    // Verify migration results (after manual execution)
    console.log('🔍 To verify the migration, check:');
    console.log('  - Companies table has been populated');
    console.log('  - Projects have company_id values');
    console.log('  - News table exists with simplified structure');
    console.log('  - Junction tables (news_projects, news_companies) exist\n');

    console.log('💡 If you need to rollback:');
    console.log('   Run: supabase/migrations/013_rollback_simplify_database_schema.sql\n');

  } catch (error) {
    console.error('❌ Migration preparation failed:', error);
    process.exit(1);
  }
}

// Helper function to verify migration results
async function verifyMigration() {
  console.log('🔍 Verifying migration results...\n');

  try {
    // Check companies table
    const { count: companyCount } = await supabase
      .from('companies')
      .select('*', { count: 'exact', head: true });
    
    console.log(`  ✓ Companies table: ${companyCount} records`);

    // Check projects with company_id
    const { data: projectsWithCompany } = await supabase
      .from('projects')
      .select('company_id')
      .not('company_id', 'is', null)
      .limit(1);
    
    if (projectsWithCompany && projectsWithCompany.length > 0) {
      console.log('  ✓ Projects linked to companies');
    } else {
      console.log('  ⚠ Projects not yet linked to companies');
    }

    // Check news table
    const { count: newsCount } = await supabase
      .from('news')
      .select('*', { count: 'exact', head: true });
    
    if (newsCount !== null) {
      console.log(`  ✓ News table: ${newsCount} records`);
    } else {
      console.log('  ⚠ News table not found or empty');
    }

    // Check junction tables
    const { data: newsProjects } = await supabase
      .from('news_projects')
      .select('id')
      .limit(1);
    
    if (newsProjects !== null) {
      console.log('  ✓ news_projects junction table exists');
    }

    const { data: newsCompanies } = await supabase
      .from('news_companies')
      .select('id')
      .limit(1);
    
    if (newsCompanies !== null) {
      console.log('  ✓ news_companies junction table exists');
    }

    console.log('\n✅ Migration verification complete!');

  } catch (error) {
    console.error('❌ Verification failed:', error);
  }
}

// Main execution
(async () => {
  const args = process.argv.slice(2);
  
  if (args.includes('--verify')) {
    await verifyMigration();
  } else {
    await runMigration();
    console.log('\n💡 Run with --verify flag to check migration results:');
    console.log('   npm run migrate:simplify -- --verify');
  }
})();
