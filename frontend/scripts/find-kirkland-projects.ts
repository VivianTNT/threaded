import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function findKirklandProjects() {
  console.log('--- Searching for Kirkland Lake projects ---\n');

  // Kirkland Lake Gold had several major mines:
  const mineNames = [
    'Macassa',
    'Detour Lake',
    'Fosterville',
    'Holt',
    'Taylor',
    'Kirkland'
  ];

  for (const mineName of mineNames) {
    const { data: projects } = await supabase
      .from('projects')
      .select(`
        id,
        name,
        location,
        company_id,
        commodities,
        stage,
        companies (
          name,
          ticker
        )
      `)
      .ilike('name', `%${mineName}%`);

    if (projects && projects.length > 0) {
      console.log(`\n🔍 Found ${projects.length} project(s) matching "${mineName}":\n`);
      projects.forEach(project => {
        console.log(`  - ${project.name}`);
        console.log(`    Location: ${project.location}`);
        console.log(`    Stage: ${project.stage}`);
        console.log(`    Company: ${project.companies ? (project.companies as any).name : 'Not linked'}`);
        console.log(`    ID: ${project.id}\n`);
      });
    }
  }

  // Also search for projects in Ontario, Canada (where Kirkland Lake mines are)
  console.log('\n--- Projects in Ontario, Canada ---\n');
  const { data: ontarioProjects } = await supabase
    .from('projects')
    .select(`
      id,
      name,
      location,
      company_id,
      commodities,
      stage,
      companies (
        name,
        ticker
      )
    `)
    .or('location.ilike.%ontario%,location.ilike.%canada%')
    .limit(10);

  console.log(`Found ${ontarioProjects?.length || 0} projects in Ontario/Canada\n`);
  ontarioProjects?.forEach(project => {
    console.log(`  - ${project.name} (${project.location})`);
    console.log(`    Company: ${project.companies ? (project.companies as any).name : 'Not linked'}\n`);
  });
}

findKirklandProjects();
