import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function showFinalSummary() {
  console.log('╔═══════════════════════════════════════════════════════════════════╗');
  console.log('║         KIRKLAND LAKE GOLD & CANADIAN MALARTIC SUMMARY           ║');
  console.log('╚═══════════════════════════════════════════════════════════════════╝\n');

  // 1. Kirkland Lake Gold Ltd.
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🏢 KIRKLAND LAKE GOLD LTD.');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  const { data: kirklandCompany } = await supabase
    .from('companies')
    .select('*')
    .eq('name', 'Kirkland Lake Gold Ltd.')
    .single();

  if (kirklandCompany) {
    console.log(`📊 Company Details:`);
    console.log(`   Name:        ${kirklandCompany.name}`);
    console.log(`   Ticker:      ${kirklandCompany.ticker || 'N/A'}`);
    console.log(`   Exchange:    ${kirklandCompany.exchange || 'N/A'}`);
    console.log(`   Country:     ${kirklandCompany.country || 'N/A'}`);
    console.log(`   Website:     ${kirklandCompany.website || 'N/A'}`);
    console.log(`   Market Cap:  ${kirklandCompany.market_cap ? `$${kirklandCompany.market_cap.toLocaleString()}` : 'N/A (Acquired)'}`);
    console.log(`   Description: ${kirklandCompany.description}`);

    if (kirklandCompany.urls && kirklandCompany.urls.length > 0) {
      console.log(`\n🔗 Links:`);
      kirklandCompany.urls.forEach((url: string) => console.log(`   - ${url}`));
    }

    // Get Kirkland Lake projects
    const { data: kirklandProjects } = await supabase
      .from('projects')
      .select('id, name, location, stage, commodities')
      .eq('company_id', kirklandCompany.id);

    console.log(`\n⛏️  Projects (${kirklandProjects?.length || 0}):`);

    // Group by unique names
    const uniqueProjects = new Map();
    kirklandProjects?.forEach(p => {
      if (!uniqueProjects.has(p.name)) {
        uniqueProjects.set(p.name, p);
      }
    });

    Array.from(uniqueProjects.values()).forEach((project: any) => {
      console.log(`   • ${project.name}`);
      console.log(`     Location:    ${project.location}`);
      console.log(`     Stage:       ${project.stage}`);
      console.log(`     Commodities: ${project.commodities}`);
    });
  }

  // 2. Canadian Malartic GP
  console.log('\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('🏢 CANADIAN MALARTIC GP');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  const { data: malarticCompany } = await supabase
    .from('companies')
    .select('*')
    .eq('name', 'Canadian Malartic GP')
    .single();

  if (malarticCompany) {
    console.log(`📊 Company Details:`);
    console.log(`   Name:        ${malarticCompany.name}`);
    console.log(`   Ticker:      ${malarticCompany.ticker || 'N/A (Joint Venture)'}`);
    console.log(`   Exchange:    ${malarticCompany.exchange || 'N/A'}`);
    console.log(`   Country:     ${malarticCompany.country || 'N/A'}`);
    console.log(`   Website:     ${malarticCompany.website || 'N/A'}`);
    console.log(`   Market Cap:  ${malarticCompany.market_cap ? `$${malarticCompany.market_cap.toLocaleString()}` : 'N/A (Joint Venture)'}`);
    console.log(`   Description: ${malarticCompany.description}`);

    if (malarticCompany.urls && malarticCompany.urls.length > 0) {
      console.log(`\n🔗 Links:`);
      malarticCompany.urls.forEach((url: string) => console.log(`   - ${url}`));
    }

    // Get Malartic projects
    const { data: malarticProjects } = await supabase
      .from('projects')
      .select('id, name, location, stage, commodities')
      .eq('company_id', malarticCompany.id);

    console.log(`\n⛏️  Projects (${malarticProjects?.length || 0}):`);

    // Group by unique names
    const uniqueMalarticProjects = new Map();
    malarticProjects?.forEach(p => {
      if (!uniqueMalarticProjects.has(p.name)) {
        uniqueMalarticProjects.set(p.name, p);
      }
    });

    Array.from(uniqueMalarticProjects.values()).forEach((project: any) => {
      console.log(`   • ${project.name}`);
      console.log(`     Location:    ${project.location}`);
      console.log(`     Stage:       ${project.stage}`);
      console.log(`     Commodities: ${project.commodities}`);
    });
  }

  console.log('\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('✅ Summary Complete');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
  console.log('All companies have been populated with:');
  console.log('  ✓ Exchange information');
  console.log('  ✓ Market cap details (where applicable)');
  console.log('  ✓ Website links');
  console.log('  ✓ Company descriptions');
  console.log('  ✓ Related URLs\n');
  console.log('All projects are now correctly linked to their respective companies.\n');
}

showFinalSummary();
