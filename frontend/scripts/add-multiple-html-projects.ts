import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);

async function addMultipleProjects() {
  console.log('🚀 Creating projects with HTML documents...\n');

  // First, get or create companies
  const companies = [
    {
      name: 'Barrick Gold Corporation',
      slug: 'barrick-gold',
      description: 'Major gold mining company with operations worldwide'
    },
    {
      name: 'Freeport-McMoRan Inc.',
      slug: 'freeport-mcmoran',
      description: 'Leading international mining company with primary focus on copper'
    }
  ];

  const companyMap: Record<string, string> = {};

  for (const company of companies) {
    // Check if company exists
    const { data: existing } = await supabase
      .from('companies')
      .select('id')
      .eq('name', company.name)
      .single();

    if (existing) {
      companyMap[company.slug] = existing.id;
      console.log(`✅ Found existing company: ${company.name}`);
    } else {
      // Create company
      const { data: newCompany, error } = await supabase
        .from('companies')
        .insert({
          name: company.name,
          description: company.description
        })
        .select()
        .single();

      if (error) {
        console.error(`Error creating company ${company.name}:`, error);
      } else {
        companyMap[company.slug] = newCompany.id;
        console.log(`✅ Created company: ${company.name}`);
      }
    }
  }

  // Define projects with HTML documents
  const projects = [
    {
      name: 'Barrick Gold - Q1 2025 10-Q Report',
      company_slug: 'barrick-gold',
      description: 'Quarterly report for Barrick Gold Corporation Q1 2025 operations and financial performance',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/barrick-gold/2025/0001193125-25-054505-1.html'],
      location: 'Toronto, Canada',
      commodities: ['Gold', 'Copper'],
      stage: 'Operating',
      npv: 25000,
      irr: 28.5,
      capex: 8500,
      status: 'active'
    },
    {
      name: 'Barrick Gold - Annual Report 2024',
      company_slug: 'barrick-gold',
      description: 'Annual report detailing Barrick Gold\'s 2024 operations across multiple mining projects',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/barrick-gold/2025/0001193125-25-073542-1.html'],
      location: 'Toronto, Canada',
      commodities: ['Gold', 'Copper'],
      stage: 'Operating',
      npv: 30000,
      irr: 32.0,
      capex: 12000,
      status: 'active'
    },
    {
      name: 'Barrick Gold - Q2 2025 Financial Update',
      company_slug: 'barrick-gold',
      description: 'Second quarter 2025 financial results and operational highlights',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/barrick-gold/2025/0001193125-25-114445-1.html'],
      location: 'Toronto, Canada',
      commodities: ['Gold', 'Copper'],
      stage: 'Operating',
      npv: 27000,
      irr: 29.5,
      capex: 9200,
      status: 'active'
    },
    {
      name: 'Barrick Gold - Q3 2025 Earnings Report',
      company_slug: 'barrick-gold',
      description: 'Third quarter 2025 earnings and production metrics',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/barrick-gold/2025/0001193125-25-178085-1.html'],
      location: 'Toronto, Canada',
      commodities: ['Gold', 'Silver', 'Copper'],
      stage: 'Operating',
      npv: 28500,
      irr: 31.0,
      capex: 10500,
      status: 'active'
    },
    {
      name: 'Freeport-McMoRan - Q3 2025 10-Q',
      company_slug: 'freeport-mcmoran',
      description: 'Quarterly report for Freeport-McMoRan copper and gold operations Q3 2025',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/freeport-mcmoran/2025/0000831259-25-000031-1.html'],
      location: 'Phoenix, Arizona, USA',
      commodities: ['Copper', 'Gold', 'Molybdenum'],
      stage: 'Operating',
      npv: 18000,
      irr: 24.5,
      capex: 5500,
      status: 'active'
    }
  ];

  console.log(`\n📊 Creating ${projects.length} projects...\n`);

  for (const project of projects) {
    const companyId = companyMap[project.company_slug];

    if (!companyId) {
      console.error(`❌ Company not found for slug: ${project.company_slug}`);
      continue;
    }

    // Check if project already exists
    const { data: existing } = await supabase
      .from('projects')
      .select('id, name')
      .eq('name', project.name)
      .single();

    if (existing) {
      console.log(`⏭️  Project already exists: ${project.name}`);
      continue;
    }

    // Create project
    const { data: newProject, error } = await supabase
      .from('projects')
      .insert({
        name: project.name,
        company_id: companyId,
        description: project.description,
        urls: project.urls,
        location: project.location,
        commodities: project.commodities,
        stage: project.stage,
        npv: project.npv,
        irr: project.irr,
        capex: project.capex,
        status: project.status
      })
      .select()
      .single();

    if (error) {
      console.error(`❌ Error creating project ${project.name}:`, error);
    } else {
      console.log(`✅ Created project: ${project.name}`);
      console.log(`   Company: ${project.company_slug}`);
      console.log(`   Documents: ${project.urls.length} HTML file(s)`);
      console.log(`   NPV: $${project.npv}M | IRR: ${project.irr}%\n`);
    }
  }

  // Summary
  const { count } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log(`\n✨ Done! Total projects in database: ${count}`);
}

addMultipleProjects().catch(console.error);
