import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function verifyProjects() {
  console.log('--- Checking all Kirkland and Malartic related data ---\n');

  // Check all projects with these names
  const { data: allProjects, error } = await supabase
    .from('projects')
    .select(`
      id,
      name,
      location,
      company_id,
      commodities,
      stage,
      companies (
        id,
        name,
        ticker,
        exchange
      )
    `)
    .or('name.ilike.%kirkland%,name.ilike.%malartic%');

  if (error) {
    console.error('Error:', error);
    return;
  }

  console.log(`Found ${allProjects?.length || 0} projects:\n`);

  allProjects?.forEach((project, index) => {
    console.log(`${index + 1}. Project: ${project.name} (ID: ${project.id})`);
    console.log(`   Location: ${project.location}`);
    console.log(`   Commodities: ${project.commodities}`);
    console.log(`   Stage: ${project.stage}`);
    if (project.companies) {
      const company = project.companies as any;
      console.log(`   ✓ Company: ${company.name}`);
      console.log(`     Ticker: ${company.ticker || 'N/A'}`);
      console.log(`     Exchange: ${company.exchange || 'N/A'}`);
    } else {
      console.log(`   ✗ No company linked`);
    }
    console.log('');
  });

  // Check for any duplicate projects
  const projectNames = allProjects?.map(p => p.name) || [];
  const duplicates = projectNames.filter((name, index) => projectNames.indexOf(name) !== index);

  if (duplicates.length > 0) {
    console.log('⚠️  Warning: Found duplicate project names:', [...new Set(duplicates)]);
  }

  // Check companies
  console.log('\n--- Companies ---\n');
  const { data: companies } = await supabase
    .from('companies')
    .select('*')
    .or('name.ilike.%kirkland%,name.ilike.%malartic%');

  companies?.forEach(company => {
    console.log(`Company: ${company.name}`);
    console.log(`  ID: ${company.id}`);
    console.log(`  Ticker: ${company.ticker || 'N/A'}`);
    console.log(`  Exchange: ${company.exchange || 'N/A'}`);
    console.log(`  Website: ${company.website || 'N/A'}`);
    console.log(`  Description: ${company.description?.substring(0, 100)}...`);
    console.log('');
  });
}

verifyProjects();
