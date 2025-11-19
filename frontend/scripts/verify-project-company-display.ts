import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function verifyDisplay() {
  console.log('=== Verifying Project-Company Display ===\n');

  // Test with specific projects
  const testCases = [
    'Macassa',
    'Canadian Malartic',
    'Fosterville'
  ];

  for (const projectName of testCases) {
    console.log(`\n📋 Testing: ${projectName}`);
    console.log('─'.repeat(60));

    // Fetch project as the hook does it
    const { data: project } = await supabase
      .from('projects')
      .select('id, name, company_id, location, stage, commodities')
      .ilike('name', projectName)
      .limit(1)
      .single();

    if (!project) {
      console.log(`❌ Project not found: ${projectName}\n`);
      continue;
    }

    console.log(`Project ID: ${project.id}`);
    console.log(`Project Name: ${project.name}`);
    console.log(`Company ID: ${project.company_id || 'NULL'}`);

    if (project.company_id) {
      // Fetch company
      const { data: company } = await supabase
        .from('companies')
        .select('id, name, ticker, exchange, website')
        .eq('id', project.company_id)
        .single();

      if (company) {
        console.log(`\n✓ Company Details:`);
        console.log(`  Name: ${company.name}`);
        console.log(`  Ticker: ${company.ticker || 'N/A'}`);
        console.log(`  Exchange: ${company.exchange || 'N/A'}`);
        console.log(`  Website: ${company.website || 'N/A'}`);
      } else {
        console.log(`\n❌ Company not found for ID: ${project.company_id}`);
      }
    } else {
      console.log(`\n⚠️  No company linked to this project`);
    }
  }

  // Test the actual query that the hook uses
  console.log('\n\n=== Testing Hook Query ===\n');
  console.log('Fetching all companies for mapping...');

  const { data: companies } = await supabase
    .from('companies')
    .select('id, name');

  const companiesMap = new Map(companies?.map(c => [c.id, c.name]) || []);
  console.log(`Loaded ${companiesMap.size} companies into map\n`);

  // Test specific company IDs
  const { data: kirklandCompany } = await supabase
    .from('companies')
    .select('id')
    .eq('name', 'Kirkland Lake Gold Ltd.')
    .single();

  if (kirklandCompany) {
    console.log(`Kirkland Lake Gold Ltd. ID: ${kirklandCompany.id}`);
    console.log(`In map: ${companiesMap.has(kirklandCompany.id)}`);
    console.log(`Map value: ${companiesMap.get(kirklandCompany.id)}\n`);
  }

  const { data: malarticCompany } = await supabase
    .from('companies')
    .select('id')
    .eq('name', 'Canadian Malartic GP')
    .single();

  if (malarticCompany) {
    console.log(`Canadian Malartic GP ID: ${malarticCompany.id}`);
    console.log(`In map: ${companiesMap.has(malarticCompany.id)}`);
    console.log(`Map value: ${companiesMap.get(malarticCompany.id)}\n`);
  }

  // Show sample of projects with their company names
  console.log('\n=== Sample Projects with Company Names ===\n');

  const { data: sampleProjects } = await supabase
    .from('projects')
    .select('id, name, company_id, location')
    .or('name.ilike.%macassa%,name.ilike.%malartic%,name.ilike.%fosterville%')
    .limit(5);

  sampleProjects?.forEach(project => {
    const companyName = project.company_id ? (companiesMap.get(project.company_id) || 'Unknown') : 'Unknown';
    console.log(`${project.name} → ${companyName}`);
    console.log(`  Location: ${project.location}`);
    console.log(`  Company ID: ${project.company_id || 'NULL'}\n`);
  });

  console.log('\n✅ Verification complete!\n');
}

verifyDisplay();
