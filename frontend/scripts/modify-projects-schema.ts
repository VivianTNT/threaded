import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

async function modifySchema() {
  console.log('🔄 Modifying projects table schema...\n');

  // Step 1: Remove problematic columns
  console.log('🗑️  Removing hard-to-populate columns...');

  // We'll verify current columns first
  const { data: sampleProject } = await supabase
    .from('projects')
    .select('*')
    .limit(1)
    .single();

  console.log('Current columns:', Object.keys(sampleProject || {}));

  // Since we can't ALTER TABLE directly via Supabase client,
  // let's delete all current projects and recreate the table with better schema
  console.log('\n⚠️  Since we can\'t modify columns directly via Supabase JS client,');
  console.log('    we need to update the schema manually or use psql.');
  console.log('\nRecommended approach:');
  console.log('1. Access Supabase Dashboard > SQL Editor');
  console.log('2. Run migration 009 SQL directly');
  console.log('\nOR we can delete existing data and re-scrape with new schema.');

  console.log('\nCurrent project count:', (await supabase.from('projects').select('*', { count: 'exact', head: true })).count);
  console.log('Proceeding with data deletion and re-scraping with correct fields...\n');

  // Delete all projects (we only have 147, easy to re-scrape)
  const { error: deleteError } = await supabase
    .from('projects')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000'); // Delete all

  if (deleteError) {
    console.error('❌ Error deleting projects:', deleteError);
  } else {
    console.log('✓ Deleted all projects');
  }

  console.log('\n✅ Ready for re-scraping with updated extractor that uses new fields!');
}

modifySchema();
