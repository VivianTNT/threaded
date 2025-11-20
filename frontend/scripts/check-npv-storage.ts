import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkNPV() {
  const { data } = await supabase
    .from('projects')
    .select('name, npv, capex, irr')
    .not('npv', 'is', null)
    .limit(5);

  console.log('=== HOW NPV IS STORED ===');
  data?.forEach(p => {
    console.log(`\n${p.name}`);
    console.log(`  NPV: ${p.npv} (raw value)`);
    console.log(`  CAPEX: ${p.capex} (raw value)`);
    console.log(`  IRR: ${p.irr}%`);
    console.log(`  NPV / 1M = ${(p.npv / 1000000).toFixed(1)}M (current formula)`);
  });
}

checkNPV();
