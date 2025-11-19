import { createClient } from '@supabase/supabase-js';
import Firecrawl from '@mendable/firecrawl-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const firecrawlKey = process.env.FIRECRAWL_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);
const firecrawl = new Firecrawl({ apiKey: firecrawlKey });

// Comprehensive list of mining companies to scrape individually
const MINING_COMPANIES = [
  // Major Gold Miners
  { name: "Newmont Corporation", ticker: "NEM", exchange: "NYSE", commodity: "Gold" },
  { name: "Barrick Gold", ticker: "GOLD", exchange: "NYSE", commodity: "Gold" },
  { name: "Agnico Eagle Mines", ticker: "AEM", exchange: "NYSE", commodity: "Gold" },
  { name: "Kinross Gold", ticker: "KGC", exchange: "NYSE", commodity: "Gold" },
  { name: "AngloGold Ashanti", ticker: "AU", exchange: "NYSE", commodity: "Gold" },
  { name: "Gold Fields", ticker: "GFI", exchange: "NYSE", commodity: "Gold" },
  { name: "Polyus", ticker: "PLZL", exchange: "MOEX", commodity: "Gold" },
  { name: "Northern Star Resources", ticker: "NST", exchange: "ASX", commodity: "Gold" },
  { name: "Evolution Mining", ticker: "EVN", exchange: "ASX", commodity: "Gold" },
  { name: "Eldorado Gold", ticker: "EGO", exchange: "NYSE", commodity: "Gold" },

  // Major Copper Miners
  { name: "BHP Group", ticker: "BHP", exchange: "NYSE", commodity: "Copper" },
  { name: "Rio Tinto", ticker: "RIO", exchange: "NYSE", commodity: "Copper" },
  { name: "Freeport-McMoRan", ticker: "FCX", exchange: "NYSE", commodity: "Copper" },
  { name: "Southern Copper", ticker: "SCCO", exchange: "NYSE", commodity: "Copper" },
  { name: "Glencore", ticker: "GLEN", exchange: "LSE", commodity: "Copper" },
  { name: "First Quantum Minerals", ticker: "FM", exchange: "TSX", commodity: "Copper" },
  { name: "Antofagasta", ticker: "ANTO", exchange: "LSE", commodity: "Copper" },
  { name: "Teck Resources", ticker: "TECK", exchange: "TSX", commodity: "Copper" },
  { name: "KGHM Polska Miedź", ticker: "KGH", exchange: "WSE", commodity: "Copper" },
  { name: "Ivanhoe Mines", ticker: "IVN", exchange: "TSX", commodity: "Copper" },

  // Major Lithium Miners
  { name: "Albemarle Corporation", ticker: "ALB", exchange: "NYSE", commodity: "Lithium" },
  { name: "SQM", ticker: "SQM", exchange: "NYSE", commodity: "Lithium" },
  { name: "Livent Corporation", ticker: "LTHM", exchange: "NYSE", commodity: "Lithium" },
  { name: "Pilbara Minerals", ticker: "PLS", exchange: "ASX", commodity: "Lithium" },
  { name: "Mineral Resources", ticker: "MIN", exchange: "ASX", commodity: "Lithium" },
  { name: "Allkem", ticker: "AKE", exchange: "ASX", commodity: "Lithium" },
  { name: "Ganfeng Lithium", ticker: "1772", exchange: "HKEX", commodity: "Lithium" },
  { name: "Tianqi Lithium", ticker: "9696", exchange: "HKEX", commodity: "Lithium" },
  { name: "Lithium Americas", ticker: "LAC", exchange: "NYSE", commodity: "Lithium" },
  { name: "Sigma Lithium", ticker: "SGML", exchange: "NASDAQ", commodity: "Lithium" },

  // Major Nickel Miners
  { name: "Vale", ticker: "VALE", exchange: "NYSE", commodity: "Nickel" },
  { name: "Nornickel", ticker: "GMKN", exchange: "MOEX", commodity: "Nickel" },
  { name: "Jinchuan Group", ticker: null, exchange: null, commodity: "Nickel" },
  { name: "IGO Limited", ticker: "IGO", exchange: "ASX", commodity: "Nickel" },
  { name: "Nickel Mines", ticker: "NIC", exchange: "ASX", commodity: "Nickel" },
  { name: "Mincor Resources", ticker: "MCR", exchange: "ASX", commodity: "Nickel" },
  { name: "Western Areas", ticker: "WSA", exchange: "ASX", commodity: "Nickel" },

  // Rare Earth Miners
  { name: "MP Materials", ticker: "MP", exchange: "NYSE", commodity: "Rare Earths" },
  { name: "Lynas Rare Earths", ticker: "LYC", exchange: "ASX", commodity: "Rare Earths" },
  { name: "China Northern Rare Earth", ticker: "600111", exchange: "SSE", commodity: "Rare Earths" },
  { name: "Iluka Resources", ticker: "ILU", exchange: "ASX", commodity: "Rare Earths" },
  { name: "Energy Fuels", ticker: "UUUU", exchange: "NYSE", commodity: "Rare Earths" },

  // Major Silver Miners
  { name: "Pan American Silver", ticker: "PAAS", exchange: "NYSE", commodity: "Silver" },
  { name: "Fresnillo", ticker: "FRES", exchange: "LSE", commodity: "Silver" },
  { name: "Hecla Mining", ticker: "HL", exchange: "NYSE", commodity: "Silver" },
  { name: "Coeur Mining", ticker: "CDE", exchange: "NYSE", commodity: "Silver" },
  { name: "MAG Silver", ticker: "MAG", exchange: "TSX", commodity: "Silver" },

  // Diversified Miners
  { name: "Anglo American", ticker: "AAL", exchange: "LSE", commodity: "Diversified" },
  { name: "South32", ticker: "S32", exchange: "ASX", commodity: "Diversified" },
  { name: "MMG Limited", ticker: "1208", exchange: "HKEX", commodity: "Diversified" },

  // Junior/Mid-Tier Companies
  { name: "Sandfire Resources", ticker: "SFR", exchange: "ASX", commodity: "Copper" },
  { name: "OZ Minerals", ticker: "OZL", exchange: "ASX", commodity: "Copper" },
  { name: "Lundin Mining", ticker: "LUN", exchange: "TSX", commodity: "Copper" },
  { name: "Hudbay Minerals", ticker: "HBM", exchange: "TSX", commodity: "Copper" },
  { name: "Capstone Copper", ticker: "CS", exchange: "TSX", commodity: "Copper" },
  { name: "Taseko Mines", ticker: "TKO", exchange: "TSX", commodity: "Copper" },
  { name: "Arizona Metals", ticker: "AMC", exchange: "TSXV", commodity: "Copper" },
  { name: "Osisko Gold Royalties", ticker: "OR", exchange: "TSX", commodity: "Gold" },
  { name: "Wheaton Precious Metals", ticker: "WPM", exchange: "NYSE", commodity: "Gold" },
  { name: "Royal Gold", ticker: "RGLD", exchange: "NASDAQ", commodity: "Gold" },
  { name: "B2Gold", ticker: "BTG", exchange: "TSX", commodity: "Gold" },
  { name: "Endeavour Mining", ticker: "EDV", exchange: "TSX", commodity: "Gold" },
  { name: "Kirkland Lake Gold", ticker: "KL", exchange: "TSX", commodity: "Gold" },
  { name: "SSR Mining", ticker: "SSRM", exchange: "NASDAQ", commodity: "Gold" },
  { name: "Alamos Gold", ticker: "AGI", exchange: "TSX", commodity: "Gold" },
  { name: "IAMGOLD", ticker: "IAG", exchange: "TSX", commodity: "Gold" },
  { name: "Centerra Gold", ticker: "CG", exchange: "TSX", commodity: "Gold" },
  { name: "Equinox Gold", ticker: "EQX", exchange: "TSX", commodity: "Gold" },
  { name: "Dundee Precious Metals", ticker: "DPM", exchange: "TSX", commodity: "Gold" },
  { name: "Torex Gold", ticker: "TXG", exchange: "TSX", commodity: "Gold" },
  { name: "Perseus Mining", ticker: "PRU", exchange: "ASX", commodity: "Gold" },
  { name: "Regis Resources", ticker: "RRL", exchange: "ASX", commodity: "Gold" },
  { name: "Silver Lake Resources", ticker: "SLR", exchange: "ASX", commodity: "Gold" },
  { name: "Ramelius Resources", ticker: "RMS", exchange: "ASX", commodity: "Gold" },
  { name: "Resolute Mining", ticker: "RSG", exchange: "ASX", commodity: "Gold" },
  { name: "St Barbara", ticker: "SBM", exchange: "ASX", commodity: "Gold" },
  { name: "Westgold Resources", ticker: "WGX", exchange: "ASX", commodity: "Gold" },
  { name: "Bellevue Gold", ticker: "BGL", exchange: "ASX", commodity: "Gold" },
  { name: "De Grey Mining", ticker: "DEG", exchange: "ASX", commodity: "Gold" },
  { name: "Greatland Gold", ticker: "GGP", exchange: "LSE", commodity: "Gold" },
  { name: "Core Lithium", ticker: "CXO", exchange: "ASX", commodity: "Lithium" },
  { name: "Liontown Resources", ticker: "LTR", exchange: "ASX", commodity: "Lithium" },
  { name: "Lake Resources", ticker: "LKE", exchange: "ASX", commodity: "Lithium" },
  { name: "Patriot Battery Metals", ticker: "PMET", exchange: "TSXV", commodity: "Lithium" },
  { name: "Frontier Lithium", ticker: "FL", exchange: "TSXV", commodity: "Lithium" },
  { name: "American Lithium", ticker: "LI", exchange: "TSXV", commodity: "Lithium" },
  { name: "Ionic Rare Earths", ticker: "IXR", exchange: "ASX", commodity: "Rare Earths" },
  { name: "Arafura Resources", ticker: "ARU", exchange: "ASX", commodity: "Rare Earths" },
  { name: "Hastings Technology Metals", ticker: "HAS", exchange: "ASX", commodity: "Rare Earths" },
  { name: "Northern Minerals", ticker: "NTU", exchange: "ASX", commodity: "Rare Earths" },
  { name: "Rainbow Rare Earths", ticker: "RBW", exchange: "LSE", commodity: "Rare Earths" },
];

let companiesAdded = 0;
let projectsAdded = 0;
let batchNumber = 0;

async function scrapeCompanyAndProjects(company: typeof MINING_COMPANIES[0]) {
  console.log(`\n🏢 Scraping: ${company.name} (${company.ticker || 'N/A'})`);

  try {
    // Search for company information including market cap
    const companySearchUrl = `https://www.google.com/search?q=${encodeURIComponent(company.name + ' ' + (company.ticker || '') + ' market cap stock price')}`;

    const companyResult = await firecrawl.scrapeUrl(companySearchUrl, {
      formats: ['extract'],
      extract: {
        schema: {
          type: 'object',
          properties: {
            market_cap: { type: 'string', description: 'Current market capitalization with units (e.g., $5.2 billion, A$3.1 billion)' },
            stock_price: { type: 'string', description: 'Current stock price' },
            description: { type: 'string', description: 'Brief description of the company' },
            website: { type: 'string', description: 'Company website URL' },
            headquarters: { type: 'string', description: 'Company headquarters location/country' },
          }
        }
      }
    });

    // Parse market cap
    let marketCap = null;
    const companyData = companyResult.extract;
    if (companyData?.market_cap) {
      const match = companyData.market_cap.match(/([\d,.]+)\s*(billion|million|B|M|bn|mn|trillion)/i);
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

    // Insert company
    const { data: existingCompany } = await supabase
      .from('companies')
      .select('id')
      .eq('name', company.name)
      .single();

    let companyId;
    if (existingCompany) {
      companyId = existingCompany.id;
      console.log(`  ⏭️  Company already exists`);
    } else {
      const { data: newCompany, error } = await supabase
        .from('companies')
        .insert({
          name: company.name,
          ticker: company.ticker,
          exchange: company.exchange,
          market_cap: marketCap,
          country: companyData?.headquarters || null,
          website: companyData?.website || null,
          description: companyData?.description || `${company.name} is a mining company focused on ${company.commodity}.`,
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
    }

    // Now search for the company's mining projects
    const projectsSearchUrl = `https://www.google.com/search?q=${encodeURIComponent(company.name + ' mining projects operations assets')}`;

    const projectsResult = await firecrawl.scrapeUrl(projectsSearchUrl, {
      formats: ['extract'],
      extract: {
        schema: {
          type: 'object',
          properties: {
            projects: {
              type: 'array',
              description: 'List of mining projects, operations, or assets owned/operated by this company',
              items: {
                type: 'object',
                properties: {
                  name: { type: 'string', description: 'Project/mine/operation name' },
                  location: { type: 'string', description: 'Country and/or region where project is located' },
                  stage: { type: 'string', description: 'Development stage (Operating, Development, Exploration, etc.)' },
                  commodities: { type: 'array', items: { type: 'string' }, description: 'List of commodities/minerals' },
                  ownership: { type: 'string', description: 'Ownership percentage' },
                  resources: { type: 'string', description: 'Mineral resource estimate with grade and tonnage' },
                  reserves: { type: 'string', description: 'Mineral reserve estimate with grade and tonnage' },
                  description: { type: 'string', description: 'Brief project description' },
                }
              }
            }
          }
        }
      }
    });

    const projects = projectsResult.extract?.projects || [];
    console.log(`  📑 Found ${projects.length} projects`);

    for (const proj of projects) {
      if (!proj.name) continue;

      // Parse ownership percentage
      let ownership = null;
      if (proj.ownership) {
        const match = proj.ownership.match(/([\d.]+)%?/);
        if (match) {
          ownership = parseFloat(match[1]);
        }
      }

      // Check if project exists
      const { data: existingProject } = await supabase
        .from('projects')
        .select('id')
        .eq('name', proj.name)
        .eq('company_id', companyId)
        .single();

      if (existingProject) {
        console.log(`    ⏭️  Project exists: ${proj.name}`);
        continue;
      }

      const { error: projError } = await supabase
        .from('projects')
        .insert({
          company_id: companyId,
          name: proj.name,
          location: proj.location || company.name.includes('Australia') ? 'Australia' : null,
          stage: proj.stage || 'Operating',
          commodities: proj.commodities && proj.commodities.length > 0 ? proj.commodities : [company.commodity],
          resource_estimate: proj.resources || null,
          reserve_estimate: proj.reserves || null,
          ownership_percentage: ownership,
          status: proj.stage === 'Operating' ? 'Active' : proj.stage === 'Exploration' ? 'Exploring' : 'In Development',
          description: proj.description || `${proj.name} is a ${company.commodity} project operated by ${company.name}.`,
          urls: [projectsSearchUrl],
        });

      if (projError) {
        console.error(`    ❌ Error adding project:`, projError.message);
      } else {
        projectsAdded++;
        console.log(`    ✓ ${proj.name} (${proj.stage || 'Operating'})`);
        if (proj.resources) console.log(`      Resources: ${proj.resources}`);
        if (proj.reserves) console.log(`      Reserves: ${proj.reserves}`);
      }
    }

    // Small delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 2000));

  } catch (error: any) {
    console.error(`  ❌ Error scraping ${company.name}:`, error.message);
  }
}

async function main() {
  console.log(`🚀 MANUAL COMPREHENSIVE MINING DATA SCRAPER`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);
  console.log(`📋 Companies to scrape: ${MINING_COMPANIES.length}`);
  console.log(`📊 Strategy: Scrape each company individually with web search`);
  console.log(`🎯 Goal: 1000+ projects with resource/reserve estimates\n`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);

  const BATCH_SIZE = 10;

  for (let i = 0; i < MINING_COMPANIES.length; i++) {
    await scrapeCompanyAndProjects(MINING_COMPANIES[i]);

    // Upload batch every 10 companies
    if ((i + 1) % BATCH_SIZE === 0) {
      batchNumber++;
      console.log(`\n📦 Batch ${batchNumber} complete (${i + 1}/${MINING_COMPANIES.length} companies processed)`);
      console.log(`   Companies added in this session: ${companiesAdded}`);
      console.log(`   Projects added in this session: ${projectsAdded}\n`);

      // Check total counts
      const { count: totalProjects } = await supabase.from('projects').select('*', { count: 'exact', head: true });
      const { count: totalCompanies } = await supabase.from('companies').select('*', { count: 'exact', head: true });
      console.log(`📊 Database totals: ${totalCompanies} companies, ${totalProjects} projects\n`);

      if (totalProjects && totalProjects >= 1000) {
        console.log(`\n🎉 TARGET REACHED: ${totalProjects} projects in database!`);
        break;
      }
    }
  }

  console.log(`\n═══════════════════════════════════════════════════════════════`);
  console.log(`✅ SCRAPING COMPLETE!`);
  console.log(`═══════════════════════════════════════════════════════════════`);
  console.log(`📊 Session Summary:`);
  console.log(`   Companies added: ${companiesAdded}`);
  console.log(`   Projects added: ${projectsAdded}`);

  const { count: finalProjects } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  const { count: finalCompanies } = await supabase.from('companies').select('*', { count: 'exact', head: true });
  console.log(`\n📊 Final Database Totals:`);
  console.log(`   Companies: ${finalCompanies}`);
  console.log(`   Projects: ${finalProjects}`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);
}

main();
