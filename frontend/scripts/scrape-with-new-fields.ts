import { createClient } from '@supabase/supabase-js';
import Firecrawl from '@mendable/firecrawl-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const firecrawlKey = process.env.FIRECRAWL_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);
const firecrawl = new Firecrawl({ apiKey: firecrawlKey });

// Major mining companies with their key projects
const COMPANY_PROJECTS = [
  { company: "BHP Group", ticker: "BHP", exchange: "NYSE", projects: ["Olympic Dam", "Escondida", "Spence", "Pampa Norte"], commodity: "Copper" },
  { company: "Rio Tinto", ticker: "RIO", exchange: "NYSE", projects: ["Oyu Tolgoi", "Kennecott", "Resolution", "Northparkes"], commodity: "Copper" },
  { company: "Freeport-McMoRan", ticker: "FCX", exchange: "NYSE", projects: ["Grasberg", "Morenci", "Cerro Verde", "El Abra"], commodity: "Copper" },
  { company: "Newmont Corporation", ticker: "NEM", exchange: "NYSE", projects: ["Carlin", "Nevada Gold Mines", "Boddington", "Tanami"], commodity: "Gold" },
  { company: "Barrick Gold", ticker: "GOLD", exchange: "NYSE", projects: ["Cortez", "Goldstrike", "Pueblo Viejo", "Kibali"], commodity: "Gold" },
  { company: "Albemarle Corporation", ticker: "ALB", exchange: "NYSE", projects: ["Silver Peak", "Atacama", "Greenbushes", "Wodgina"], commodity: "Lithium" },
  { company: "Pilbara Minerals", ticker: "PLS", exchange: "ASX", projects: ["Pilgangoora", "Ngungaju"], commodity: "Lithium" },
  { company: "Glencore", ticker: "GLEN", exchange: "LSE", projects: ["Katanga", "Mutanda", "Kamoto", "Antapaccay"], commodity: "Copper" },
  { company: "Vale", ticker: "VALE", exchange: "NYSE", projects: ["Voisey's Bay", "Sudbury", "Thompson", "Onça Puma"], commodity: "Nickel" },
  { company: "Anglo American", ticker: "AAL", exchange: "LSE", projects: ["Los Bronces", "Collahuasi", "Quellaveco", "Minas-Rio"], commodity: "Copper" },
];

let companiesAdded = 0;
let projectsAdded = 0;

async function scrapeProjectDetails(companyName: string, projectName: string, commodity: string) {
  console.log(`  🔍 Scraping: ${projectName}...`);

  try {
    const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(`${companyName} ${projectName} mine production rate capex mine life`)}`;

    const result = await firecrawl.scrapeUrl(searchUrl, {
      formats: ['extract'],
      extract: {
        schema: {
          type: 'object',
          properties: {
            operator: { type: 'string', description: 'Operating company name' },
            location: { type: 'string', description: 'Country and region where mine is located' },
            stage: { type: 'string', description: 'Operating, Development, Exploration, or Closed' },
            production_rate: { type: 'string', description: 'Annual production rate (e.g., "500k tonnes Cu/year", "200k oz Au/year")' },
            mine_life: { type: 'string', description: 'Expected mine life (e.g., "20 years", "2015-2040")' },
            capex: { type: 'string', description: 'Capital expenditure (e.g., "$2.5B", "A$1.2 billion")' },
            project_type: { type: 'string', description: 'Mining method: Open Pit, Underground, Heap Leach, In-Situ Recovery, etc.' },
            first_production: { type: 'string', description: 'Year of first production or expected start year' },
            description: { type: 'string', description: 'Brief description of the project' },
          }
        }
      }
    });

    const data = result.extract;
    if (!data) {
      console.log(`    ⚠️  No data extracted`);
      return null;
    }

    // Parse first production year to date
    let firstProdDate = null;
    if (data.first_production) {
      const year = data.first_production.match(/\d{4}/)?.[0];
      if (year) {
        firstProdDate = `${year}-01-01`;
      }
    }

    return {
      name: projectName,
      operator: data.operator || companyName,
      location: data.location || 'Unknown',
      stage: data.stage || 'Operating',
      commodities: [commodity],
      production_rate: data.production_rate || null,
      mine_life: data.mine_life || null,
      capex: data.capex || null,
      project_type: data.project_type || null,
      first_production: firstProdDate,
      status: data.stage === 'Operating' ? 'Active' : data.stage === 'Closed' ? 'Closed' : 'In Development',
      description: data.description || `${projectName} is a ${commodity} mining project operated by ${companyName}.`,
      urls: [searchUrl],
    };

  } catch (error: any) {
    console.error(`    ❌ Error: ${error.message}`);
    return null;
  }
}

async function scrapeCompanyAndProjects(item: typeof COMPANY_PROJECTS[0]) {
  console.log(`\n🏢 ${item.company} (${item.ticker})`);

  // First, ensure company exists
  let companyId: string;

  const { data: existingCompany } = await supabase
    .from('companies')
    .select('id')
    .eq('name', item.company)
    .single();

  if (existingCompany) {
    companyId = existingCompany.id;
    console.log(`  ⏭️  Company exists`);
  } else {
    // Search for company market cap
    const companySearchUrl = `https://www.google.com/search?q=${encodeURIComponent(item.company + ' ' + item.ticker + ' market cap stock price')}`;

    try {
      const companyResult = await firecrawl.scrapeUrl(companySearchUrl, {
        formats: ['extract'],
        extract: {
          schema: {
            type: 'object',
            properties: {
              market_cap: { type: 'string', description: 'Current market capitalization with units' },
              description: { type: 'string', description: 'Brief company description' },
            }
          }
        }
      });

      let marketCap = null;
      if (companyResult.extract?.market_cap) {
        const match = companyResult.extract.market_cap.match(/([\d,.]+)\s*(billion|million|B|M|bn|mn|trillion)/i);
        if (match) {
          const value = parseFloat(match[1].replace(/,/g, ''));
          const unit = match[2].toLowerCase();
          if (unit.startsWith('t')) {
            marketCap = value * 1000000000000;
          } else if (unit.startsWith('b')) {
            marketCap = value * 1000000000;
          } else {
            marketCap = value * 1000000;
          }
        }
      }

      const { data: newCompany, error } = await supabase
        .from('companies')
        .insert({
          name: item.company,
          ticker: item.ticker,
          exchange: item.exchange,
          market_cap: marketCap,
          description: companyResult.extract?.description || `${item.company} is a major ${item.commodity} mining company.`,
          urls: [companySearchUrl],
        })
        .select('id')
        .single();

      if (error) {
        console.error(`  ❌ Error adding company:`, error.message);
        return;
      }

      companyId = newCompany.id;
      companiesAdded++;
      console.log(`  ✓ Added company${marketCap ? ` (Market cap: $${(marketCap / 1000000000).toFixed(2)}B)` : ''}`);
    } catch (error: any) {
      console.error(`  ❌ Error scraping company:`, error.message);
      return;
    }
  }

  // Now scrape each project
  for (const projectName of item.projects) {
    const projectData = await scrapeProjectDetails(item.company, projectName, item.commodity);

    if (!projectData) continue;

    // Check if project already exists
    const { data: existingProject } = await supabase
      .from('projects')
      .select('id')
      .eq('name', projectName)
      .eq('company_id', companyId)
      .single();

    if (existingProject) {
      console.log(`    ⏭️  Project exists: ${projectName}`);
      continue;
    }

    const { error } = await supabase
      .from('projects')
      .insert({
        company_id: companyId,
        ...projectData,
      });

    if (error) {
      console.error(`    ❌ Error adding project:`, error.message);
    } else {
      projectsAdded++;
      console.log(`    ✓ ${projectName} (${projectData.stage})`);
      if (projectData.production_rate) console.log(`      Production: ${projectData.production_rate}`);
      if (projectData.mine_life) console.log(`      Mine Life: ${projectData.mine_life}`);
      if (projectData.capex) console.log(`      CAPEX: ${projectData.capex}`);
      if (projectData.project_type) console.log(`      Type: ${projectData.project_type}`);
    }

    // Delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

async function main() {
  console.log(`🚀 MINING DATA SCRAPER (Updated Fields)`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);
  console.log(`📋 Companies to process: ${COMPANY_PROJECTS.length}`);
  console.log(`📊 Fields to extract: operator, production_rate, mine_life, capex, project_type`);
  console.log(`🎯 Goal: Populate 1000+ projects with ACTUAL usable data\n`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);

  for (const item of COMPANY_PROJECTS) {
    await scrapeCompanyAndProjects(item);
  }

  console.log(`\n═══════════════════════════════════════════════════════════════`);
  console.log(`✅ INITIAL SCRAPING COMPLETE!`);
  console.log(`═══════════════════════════════════════════════════════════════`);
  console.log(`📊 Session Summary:`);
  console.log(`   Companies added: ${companiesAdded}`);
  console.log(`   Projects added: ${projectsAdded}`);

  const { count: finalProjects } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  const { count: finalCompanies } = await supabase.from('companies').select('*', { count: 'exact', head: true });
  console.log(`\n📊 Database Totals:`);
  console.log(`   Companies: ${finalCompanies}`);
  console.log(`   Projects: ${finalProjects}`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);

  console.log(`💡 Next: Expand to more companies and projects to reach 1000+ target`);
}

main();
