import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

async function main() {
  console.log('🔄 Clearing projects for re-scraping with better fields...\n');

  // Delete all existing projects
  const { error } = await supabase
    .from('projects')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000'); // Deletes all

  if (error) {
    console.error('❌ Error deleting projects:', error);
    return;
  }

  const { count } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  console.log(`✓ Projects deleted. Current count: ${count}`);
  console.log('\n📋 Database now ready for re-scraping with fields that are actually available:');
  console.log('   ✓ name, location, stage, commodities, status, description');
  console.log('   ✓ operator (who runs it)');
  console.log('   ✓ production_rate (annual output)');
  console.log('   ✓ mine_life (expected years)');
  console.log('   ✓ capex (capital cost)');
  console.log('   ✓ project_type (Open Pit, Underground, etc.)');
  console.log('\n   ✗ Removed: resource_estimate, reserve_estimate, ownership_percentage');
  console.log('     (These require parsing PDF technical reports - too complex for web scraping)\n');
}

main();
