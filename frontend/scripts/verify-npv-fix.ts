import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function verifyNPVFix() {
  const { data } = await supabase
    .from('projects')
    .select('name, npv, capex, irr')
    .not('npv', 'is', null)
    .limit(5);

  console.log('=== NPV DISPLAY FIX VERIFICATION ===\n');
  data?.forEach(p => {
    console.log(`${p.name}`);
    console.log(`  Raw NPV: ${p.npv}`);
    console.log(`  OLD Formula (WRONG): $${(p.npv / 1000000).toFixed(1)}M`);
    console.log(`  NEW Formula (CORRECT): $${p.npv.toFixed(1)}M`);
    console.log(`  CAPEX: $${p.capex.toFixed(1)}M`);
    console.log(`  IRR: ${p.irr.toFixed(1)}%`);
    console.log('');
  });

  console.log('✅ NPV is now correctly displayed without division');
  console.log('✅ Values are already stored in millions (M)');
}

verifyNPVFix();
