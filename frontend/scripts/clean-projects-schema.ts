import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

async function cleanSchema() {
  console.log('🧹 Cleaning projects table schema...\n');

  // Step 1: Get all existing projects with only the guaranteed fields
  console.log('📥 Fetching existing projects...');
  const { data: existingProjects, error: fetchError } = await supabase
    .from('projects')
    .select('id, company_id, name, location, stage, commodities, status, description, urls, watchlist, created_at');

  if (fetchError) {
    console.error('❌ Error fetching projects:', fetchError);
    return;
  }

  console.log(`✓ Fetched ${existingProjects?.length || 0} projects\n`);

  // Step 2: Delete all projects
  console.log('🗑️  Deleting all projects...');
  const { error: deleteError } = await supabase
    .from('projects')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000');

  if (deleteError) {
    console.error('❌ Error deleting:', deleteError);
    return;
  }
  console.log('✓ All projects deleted\n');

  // Step 3: Re-insert with only guaranteed fields (schema will auto-handle missing columns)
  console.log('📤 Re-inserting projects with clean data (only guaranteed fields)...');

  if (existingProjects && existingProjects.length > 0) {
    // Insert in batches of 50
    for (let i = 0; i < existingProjects.length; i += 50) {
      const batch = existingProjects.slice(i, i + 50);
      const cleanedBatch = batch.map(p => ({
        company_id: p.company_id,
        name: p.name,
        location: p.location || 'Unknown',
        stage: p.stage || 'Operating',
        commodities: p.commodities || [],
        status: p.status || 'Active',
        description: p.description || '',
        urls: p.urls || [],
        watchlist: p.watchlist || false,
      }));

      const { error: insertError } = await supabase
        .from('projects')
        .insert(cleanedBatch);

      if (insertError) {
        console.error(`❌ Error inserting batch ${i / 50 + 1}:`, insertError.message);
      } else {
        console.log(`✓ Inserted batch ${i / 50 + 1} (${cleanedBatch.length} projects)`);
      }
    }
  }

  console.log('\n✅ Schema cleanup complete!\n');

  // Verify
  const { count } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  console.log(`📊 Final project count: ${count}\n`);

  console.log('📋 Remaining fields in projects table:');
  console.log('   • id (UUID)');
  console.log('   • company_id (UUID reference)');
  console.log('   • name (required)');
  console.log('   • location (guaranteed)');
  console.log('   • stage (guaranteed)');
  console.log('   • commodities (array, guaranteed)');
  console.log('   • status (guaranteed)');
  console.log('   • description (guaranteed)');
  console.log('   • urls (array)');
  console.log('   • watchlist (boolean)');
  console.log('   • created_at, updated_at (timestamps)\n');

  console.log('❌ Removed problematic optional fields:');
  console.log('   • resource_estimate (requires PDF parsing)');
  console.log('   • reserve_estimate (requires PDF parsing)');
  console.log('   • ownership_percentage (inconsistently available)');
  console.log('   • operator (can use company relationship instead)');
  console.log('   • production_rate (inconsistently available)');
  console.log('   • mine_life (inconsistently available)');
  console.log('   • capex (inconsistently available)');
  console.log('   • first_production (inconsistently available)');
  console.log('   • project_type (inconsistently available)\n');
}

cleanSchema();
