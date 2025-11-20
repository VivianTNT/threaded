import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkAISC() {
  // Check total projects
  const { count: total } = await supabase.from('projects').select('*', { count: 'exact', head: true });

  // Check projects with AISC
  const { count: withAISC } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .not('aisc', 'is', null);

  // Check recent projects (last 20)
  const { data: recent } = await supabase
    .from('projects')
    .select('id, name, aisc, commodities, npv, created_at')
    .order('created_at', { ascending: false })
    .limit(20);

  console.log('=== AISC STATUS ===');
  console.log('Total projects:', total);
  console.log('Projects with AISC:', withAISC);
  console.log('Coverage:', ((withAISC / total) * 100).toFixed(1) + '%');
  console.log('\n=== RECENT 20 PROJECTS ===');
  recent?.forEach(p => {
    console.log(`${p.name?.substring(0, 30).padEnd(30)} - AISC: ${p.aisc || 'NULL'} - NPV: ${p.npv || 'NULL'}`);
  });
}

checkAISC();
