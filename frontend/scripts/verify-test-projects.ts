import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function verifyTestProjects() {
  const { data: projects } = await supabase
    .from('projects')
    .select('name, aisc, qualified_persons, latitude, longitude')
    .in('name', [
      'Cortez Hills Gold Project',
      'Kamoa-Kakula Copper Project',
      'Pilbara Lithium Project',
      'Kitimat Molybdenum-Silver Project',
      'Pueblo Viejo Gold-Silver Mine'
    ]);

  console.log('Test Projects:\n');
  projects?.forEach(p => {
    console.log(`${p.name}:`);
    console.log(`  AISC: ${p.aisc}`);
    console.log(`  Qualified Persons: ${p.qualified_persons?.length || 0}`);
    console.log(`  Coordinates: ${p.latitude}, ${p.longitude}`);
    console.log();
  });
}

verifyTestProjects();
