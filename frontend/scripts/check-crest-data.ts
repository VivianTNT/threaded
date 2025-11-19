import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkCrestData() {
  const { data } = await supabase
    .from('projects')
    .select('name, npv, capex, irr')
    .ilike('name', '%Crest Complex%')
    .limit(1)
    .single();

  console.log('Crest Complex data:');
  console.log('  Name:', data?.name);
  console.log('  NPV:', data?.npv, '(type:', typeof data?.npv, ')');
  console.log('  CAPEX:', data?.capex, '(type:', typeof data?.capex, ')');
  console.log('  IRR:', data?.irr, '(type:', typeof data?.irr, ')');
}

checkCrestData();
