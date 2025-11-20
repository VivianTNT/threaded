import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function fixKirklandProjectLinks() {
  console.log('--- Fixing Kirkland Lake Gold project links ---\n');

  // Get the Kirkland Lake Gold company ID
  const { data: kirklandCompany } = await supabase
    .from('companies')
    .select('id, name, ticker')
    .eq('name', 'Kirkland Lake Gold Ltd.')
    .single();

  if (!kirklandCompany) {
    console.error('Kirkland Lake Gold Ltd. company not found!');
    return;
  }

  console.log(`Found Kirkland Lake Gold Ltd. (ID: ${kirklandCompany.id})\n`);

  // Kirkland Lake Gold's major assets
  const kirklandProjects = [
    'Macassa',
    'Detour Lake',
    'Fosterville'
  ];

  let updated = 0;
  let alreadyLinked = 0;

  for (const projectName of kirklandProjects) {
    // Find all projects with this name
    const { data: projects, error } = await supabase
      .from('projects')
      .select('id, name, location, company_id, companies(name)')
      .ilike('name', `%${projectName}%`);

    if (error) {
      console.error(`Error finding ${projectName} projects:`, error);
      continue;
    }

    console.log(`\n🔍 Found ${projects?.length || 0} projects matching "${projectName}":`);

    for (const project of projects || []) {
      const currentCompany = project.companies as any;

      // Skip if already linked to Kirkland Lake Gold
      if (project.company_id === kirklandCompany.id) {
        console.log(`  ✓ ${project.name} already linked to Kirkland Lake Gold Ltd.`);
        alreadyLinked++;
        continue;
      }

      // Update the project to link to Kirkland Lake Gold
      const { error: updateError } = await supabase
        .from('projects')
        .update({ company_id: kirklandCompany.id })
        .eq('id', project.id);

      if (updateError) {
        console.error(`  ✗ Error updating ${project.name}:`, updateError);
      } else {
        console.log(`  ✓ Updated ${project.name} (was: ${currentCompany?.name || 'None'}) → Kirkland Lake Gold Ltd.`);
        updated++;
      }
    }
  }

  console.log(`\n--- Summary ---`);
  console.log(`Updated: ${updated} projects`);
  console.log(`Already linked: ${alreadyLinked} projects`);

  // Final verification
  console.log('\n--- Verification: All Kirkland Lake Gold projects ---\n');

  const { data: verifyProjects } = await supabase
    .from('projects')
    .select(`
      id,
      name,
      location,
      stage,
      commodities,
      companies!inner (
        name,
        ticker,
        exchange
      )
    `)
    .eq('company_id', kirklandCompany.id);

  console.log(`Total projects linked to Kirkland Lake Gold Ltd.: ${verifyProjects?.length || 0}\n`);

  verifyProjects?.forEach((project, index) => {
    const company = project.companies as any;
    console.log(`${index + 1}. ${project.name}`);
    console.log(`   Location: ${project.location}`);
    console.log(`   Stage: ${project.stage}`);
    console.log(`   Commodities: ${project.commodities}`);
    console.log(`   Company: ${company.name} (${company.ticker})`);
    console.log('');
  });
}

fixKirklandProjectLinks();
