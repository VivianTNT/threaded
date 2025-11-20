import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function updateRedLake() {
  console.log('Updating Red Lake project with correct values...\n');

  const { data, error } = await supabase
    .from('projects')
    .update({
      npv: 892.5,      // Was: 892500000, Now: 892.5M
      capex: 445,      // Was: 445000000, Now: 445M
      aisc: 925        // Was: 925.50, Now: 925
    })
    .ilike('name', '%Red Lake%')
    .select();

  if (error) {
    console.error('Error updating project:', error);
    return;
  }

  console.log('✅ Red Lake project updated successfully!\n');
  console.log('Updated values:');
  console.log(`  NPV: ${data?.[0]?.npv}M (was: 892500000)`);
  console.log(`  CAPEX: ${data?.[0]?.capex}M (was: 445000000)`);
  console.log(`  AISC: $${data?.[0]?.aisc}/oz (was: $925.50)`);
  console.log('\nFrontend will now display:');
  console.log('  NPV: $893M');
  console.log('  CAPEX: $445M');
  console.log('  AISC: $925/unit');
}

updateRedLake();
