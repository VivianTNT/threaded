import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);

async function addMoreMiningProjects() {
  console.log('🚀 Creating more projects with HTML documents from major mining companies...\n');

  // Define companies
  const companies = [
    {
      name: 'BHP Group Limited',
      slug: 'bhp-group',
      description: 'One of the world\'s largest mining companies, producing iron ore, copper, coal, and nickel'
    },
    {
      name: 'Rio Tinto plc',
      slug: 'rio-tinto',
      description: 'Global mining and metals company specializing in iron ore, aluminum, copper, and diamonds'
    },
    {
      name: 'Vale S.A.',
      slug: 'vale',
      description: 'Brazilian multinational mining corporation, world\'s largest producer of iron ore and nickel'
    },
    {
      name: 'Newmont Corporation',
      slug: 'newmont',
      description: 'World\'s leading gold mining company with operations across five continents'
    },
    {
      name: 'Southern Copper Corporation',
      slug: 'southern-copper',
      description: 'Major integrated copper producer with operations in Peru and Mexico'
    },
    {
      name: 'Teck Resources Limited',
      slug: 'teck-resources',
      description: 'Canadian diversified mining company producing copper, zinc, and steelmaking coal'
    }
  ];

  const companyMap: Record<string, string> = {};

  // Create or get companies
  for (const company of companies) {
    const { data: existing } = await supabase
      .from('companies')
      .select('id')
      .eq('name', company.name)
      .single();

    if (existing) {
      companyMap[company.slug] = existing.id;
      console.log(`✅ Found existing company: ${company.name}`);
    } else {
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
    // BHP Group projects
    {
      name: 'BHP Group - Q4 2024 Results',
      company_slug: 'bhp-group',
      description: 'Fourth quarter 2024 production and financial results for BHP Group',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/bhp-group/2024/0001193125-24-223228-1.html'],
      location: 'Melbourne, Australia',
      commodities: ['Iron Ore', 'Copper', 'Coal', 'Nickel'],
      stage: 'Operating',
      npv: 45000,
      irr: 35.0,
      capex: 15000,
      status: 'active'
    },
    {
      name: 'BHP Group - Annual Report 2024',
      company_slug: 'bhp-group',
      description: 'Full year 2024 annual report covering global mining operations',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/bhp-group/2024/0001193125-24-260761-1.html'],
      location: 'Melbourne, Australia',
      commodities: ['Iron Ore', 'Copper', 'Coal'],
      stage: 'Operating',
      npv: 50000,
      irr: 38.0,
      capex: 18000,
      status: 'active'
    },
    // Rio Tinto projects
    {
      name: 'Rio Tinto - Q1 2025 Production Report',
      company_slug: 'rio-tinto',
      description: 'First quarter 2025 production metrics and operational updates',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/rio-tinto/2025/0001628280-25-016021-1.html'],
      location: 'London, UK',
      commodities: ['Iron Ore', 'Aluminum', 'Copper'],
      stage: 'Operating',
      npv: 42000,
      irr: 33.5,
      capex: 14500,
      status: 'active'
    },
    {
      name: 'Rio Tinto - Oyu Tolgoi Copper-Gold Project Update',
      company_slug: 'rio-tinto',
      description: 'Update on Oyu Tolgoi underground mine in Mongolia',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/rio-tinto/2025/0001628280-25-021425-1.html'],
      location: 'Mongolia',
      commodities: ['Copper', 'Gold'],
      stage: 'Development',
      npv: 38000,
      irr: 31.0,
      capex: 12000,
      status: 'active'
    },
    // Vale projects
    {
      name: 'Vale - Q1 2025 Financial Results',
      company_slug: 'vale',
      description: 'First quarter 2025 financial performance and iron ore production',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/vale/2025/0001292814-25-002865-1.html'],
      location: 'Rio de Janeiro, Brazil',
      commodities: ['Iron Ore', 'Nickel', 'Copper'],
      stage: 'Operating',
      npv: 55000,
      irr: 40.0,
      capex: 20000,
      status: 'active'
    },
    {
      name: 'Vale - Nickel Operations Update',
      company_slug: 'vale',
      description: 'Comprehensive update on Vale\'s global nickel operations',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/vale/2025/0001292814-25-003186-1.html'],
      location: 'Brazil',
      commodities: ['Nickel', 'Copper', 'Cobalt'],
      stage: 'Operating',
      npv: 32000,
      irr: 28.5,
      capex: 11000,
      status: 'active'
    },
    // Newmont projects
    {
      name: 'Newmont - Q1 2025 Gold Production',
      company_slug: 'newmont',
      description: 'First quarter 2025 gold and copper production results',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/newmont/2025/0001164727-25-000020-1.html'],
      location: 'Denver, Colorado, USA',
      commodities: ['Gold', 'Copper', 'Silver'],
      stage: 'Operating',
      npv: 35000,
      irr: 29.0,
      capex: 9500,
      status: 'active'
    },
    {
      name: 'Newmont - Annual Sustainability Report 2024',
      company_slug: 'newmont',
      description: '2024 sustainability and ESG performance across all operations',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/newmont/2025/0001164727-25-000035-1.html'],
      location: 'Multiple Locations',
      commodities: ['Gold', 'Copper'],
      stage: 'Operating',
      npv: 40000,
      irr: 32.0,
      capex: 13000,
      status: 'active'
    },
    // Southern Copper projects
    {
      name: 'Southern Copper - Q4 2024 Results',
      company_slug: 'southern-copper',
      description: 'Fourth quarter 2024 copper production and financial results',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/southern-copper/2024/0001558370-24-010710-1.html'],
      location: 'Peru and Mexico',
      commodities: ['Copper', 'Molybdenum', 'Silver'],
      stage: 'Operating',
      npv: 28000,
      irr: 26.5,
      capex: 8000,
      status: 'active'
    },
    {
      name: 'Southern Copper - Expansion Project Update',
      company_slug: 'southern-copper',
      description: 'Update on Tia Maria and Los Chancas expansion projects',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/southern-copper/2025/0001558370-25-002017-1.html'],
      location: 'Peru',
      commodities: ['Copper', 'Gold'],
      stage: 'Development',
      npv: 22000,
      irr: 24.0,
      capex: 6500,
      status: 'active'
    },
    // Teck Resources projects
    {
      name: 'Teck Resources - Q1 2025 Report',
      company_slug: 'teck-resources',
      description: 'First quarter 2025 copper and zinc production results',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/teck-resources/2025/0000886986-25-000004-1.html'],
      location: 'Vancouver, Canada',
      commodities: ['Copper', 'Zinc', 'Coal'],
      stage: 'Operating',
      npv: 24000,
      irr: 25.5,
      capex: 7500,
      status: 'active'
    },
    {
      name: 'Teck Resources - QB2 Copper Project',
      company_slug: 'teck-resources',
      description: 'Quebrada Blanca Phase 2 copper project operational update',
      urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/teck-resources/2025/0000950142-25-000732-1.html'],
      location: 'Chile',
      commodities: ['Copper', 'Molybdenum'],
      stage: 'Development',
      npv: 19000,
      irr: 22.0,
      capex: 5800,
      status: 'active'
    }
  ];

  console.log(`\n📊 Creating ${projects.length} projects...\n`);

  let created = 0;
  let skipped = 0;

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
      skipped++;
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
      console.log(`✅ Created: ${project.name}`);
      console.log(`   Company: ${project.company_slug} | NPV: $${project.npv}M | IRR: ${project.irr}%\n`);
      created++;
    }
  }

  // Summary
  const { count } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log(`\n✨ Done!`);
  console.log(`   Created: ${created} projects`);
  console.log(`   Skipped: ${skipped} (already exist)`);
  console.log(`   Total in database: ${count}`);
}

addMoreMiningProjects().catch(console.error);
