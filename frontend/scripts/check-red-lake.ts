import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkRedLake() {
  const { data } = await supabase
    .from('projects')
    .select('id, name, latitude, longitude, npv, capex')
    .ilike('name', '%Red Lake%')
    .limit(1)
    .single();

  console.log('Red Lake Project in Database:');
  console.log('  ID:', data?.id);
  console.log('  Name:', data?.name);
  console.log('  Latitude:', data?.latitude);
  console.log('  Longitude:', data?.longitude);
  console.log('  NPV:', data?.npv);
  console.log('  CAPEX:', data?.capex);
}

checkRedLake();
