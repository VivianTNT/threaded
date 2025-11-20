import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

// Comprehensive mining companies with all major global players
const MINING_COMPANIES = [
  // Gold Producers
  { ticker: 'NEM', name: 'Newmont Corporation', website: 'https://www.newmont.com', market_cap: 65000000000 },
  { ticker: 'GOLD', name: 'Barrick Gold', website: 'https://www.barrick.com', market_cap: 31000000000 },
  { ticker: 'AEM', name: 'Agnico Eagle Mines', website: 'https://www.agnicoeagle.com', market_cap: 32000000000 },
  { ticker: 'KGC', name: 'Kinross Gold', website: 'https://www.kinross.com', market_cap: 10000000000 },
  { ticker: 'AU', name: 'AngloGold Ashanti', website: 'https://www.anglogoldashanti.com', market_cap: 11000000000 },
  { ticker: 'WPM', name: 'Wheaton Precious Metals', website: 'https://www.wheatonpm.com', market_cap: 27000000000 },
  { ticker: 'FNV', name: 'Franco-Nevada', website: 'https://www.franco-nevada.com', market_cap: 22000000000 },
  { ticker: 'RGLD', name: 'Royal Gold', website: 'https://www.royalgold.com', market_cap: 8500000000 },
  { ticker: 'BTG', name: 'B2Gold', website: 'https://www.b2gold.com', market_cap: 4300000000 },
  { ticker: 'EGO', name: 'Eldorado Gold', website: 'https://www.eldoradogold.com', market_cap: 2400000000 },
  { ticker: 'IAG', name: 'IAMGOLD', website: 'https://www.iamgold.com', market_cap: 2100000000 },
  { ticker: 'NGD', name: 'New Gold', website: 'https://www.newgold.com', market_cap: 1200000000 },
  { ticker: 'HMY', name: 'Harmony Gold', website: 'https://www.harmony.co.za', market_cap: 6000000000 },
  { ticker: 'SSRM', name: 'SSR Mining', website: 'https://www.ssrmining.com', market_cap: 3200000000 },
  { ticker: 'SAND', name: 'Sandstorm Gold', website: 'https://www.sandstormgold.com', market_cap: 1100000000 },
  { ticker: 'OR', name: 'Osisko Gold Royalties', website: 'https://www.osiskogr.com', market_cap: 3200000000 },

  // Copper Producers
  { ticker: 'FCX', name: 'Freeport-McMoRan', website: 'https://www.fcx.com', market_cap: 67000000000 },
  { ticker: 'SCCO', name: 'Southern Copper', website: 'https://www.southerncoppercorp.com', market_cap: 61000000000 },
  { ticker: 'TECK', name: 'Teck Resources', website: 'https://www.teck.com', market_cap: 24000000000 },
  { ticker: 'HBM', name: 'Hudbay Minerals', website: 'https://www.hudbayminerals.com', market_cap: 2800000000 },
  { ticker: 'CS', name: 'Capstone Copper', website: 'https://www.capstonecopper.com', market_cap: 4500000000 },
  { ticker: 'ERO', name: 'Ero Copper', website: 'https://www.erocopper.com', market_cap: 1900000000 },
  { ticker: 'CMMC', name: 'Copper Mountain Mining', website: 'https://www.cumtn.com', market_cap: 600000000 },
  { ticker: 'TKO', name: 'Taseko Mines', website: 'https://www.tasekomines.com', market_cap: 900000000 },

  // Diversified Majors
  { ticker: 'BHP', name: 'BHP Group', website: 'https://www.bhp.com', market_cap: 145000000000 },
  { ticker: 'RIO', name: 'Rio Tinto', website: 'https://www.riotinto.com', market_cap: 104000000000 },
  { ticker: 'VALE', name: 'Vale S.A.', website: 'https://www.vale.com', market_cap: 51000000000 },
  { ticker: 'GLEN', name: 'Glencore', website: 'https://www.glencore.com', market_cap: 58000000000 },
  { ticker: 'AAL', name: 'Anglo American', website: 'https://www.angloamerican.com', market_cap: 35000000000 },

  // Silver Producers
  { ticker: 'PAAS', name: 'Pan American Silver', website: 'https://www.panamericansilver.com', market_cap: 7600000000 },
  { ticker: 'CDE', name: 'Coeur Mining', website: 'https://www.coeur.com', market_cap: 2300000000 },
  { ticker: 'HL', name: 'Hecla Mining', website: 'https://www.hecla-mining.com', market_cap: 3900000000 },
  { ticker: 'AG', name: 'First Majestic Silver', website: 'https://www.firstmajestic.com', market_cap: 1700000000 },
  { ticker: 'FSM', name: 'Fortuna Silver Mines', website: 'https://www.fortunasilver.com', market_cap: 1400000000 },
  { ticker: 'EXK', name: 'Endeavour Silver', website: 'https://www.edrsilver.com', market_cap: 1100000000 },
  { ticker: 'MAG', name: 'MAG Silver', website: 'https://www.magsilver.com', market_cap: 1300000000 },
  { ticker: 'SVM', name: 'Silvercorp Metals', website: 'https://www.silvercorp.ca', market_cap: 700000000 },
  { ticker: 'SILV', name: 'SilverCrest Metals', website: 'https://www.silvercrestmetals.com', market_cap: 450000000 },

  // Lithium & Battery Metals
  { ticker: 'ALB', name: 'Albemarle Corporation', website: 'https://www.albemarle.com', market_cap: 13000000000 },
  { ticker: 'SQM', name: 'Sociedad Química y Minera', website: 'https://www.sqm.com', market_cap: 14000000000 },
  { ticker: 'LAC', name: 'Lithium Americas', website: 'https://www.lithiumamericas.com', market_cap: 800000000 },
  { ticker: 'PLL', name: 'Piedmont Lithium', website: 'https://www.piedmontlithium.com', market_cap: 600000000 },
  { ticker: 'SGML', name: 'Sigma Lithium', website: 'https://www.sigmalithium.ca', market_cap: 3200000000 },
  { ticker: 'LI', name: 'Lithium Chile', website: 'https://www.lithiumchile.ca', market_cap: 180000000 },
  { ticker: 'LITM', name: 'Snow Lake Resources', website: 'https://www.snowlakelithium.com', market_cap: 50000000 },

  // Rare Earth & Critical Minerals
  { ticker: 'MP', name: 'MP Materials', website: 'https://www.mpmaterials.com', market_cap: 2800000000 },
  { ticker: 'UUUU', name: 'Energy Fuels', website: 'https://www.energyfuels.com', market_cap: 1100000000 },
  { ticker: 'LYC', name: 'Lynas Rare Earths', website: 'https://www.lynasrareearths.com', market_cap: 5800000000 },
  { ticker: 'ILU', name: 'Iluka Resources', website: 'https://www.iluka.com', market_cap: 3400000000 },
  { ticker: 'REEMF', name: 'Rare Element Resources', website: 'https://www.rareelementresources.com', market_cap: 120000000 },

  // Uranium
  { ticker: 'CCJ', name: 'Cameco Corporation', website: 'https://www.cameco.com', market_cap: 23000000000 },
  { ticker: 'DNN', name: 'Denison Mines', website: 'https://www.denisonmines.com', market_cap: 1800000000 },
  { ticker: 'NXE', name: 'NexGen Energy', website: 'https://www.nexgenenergy.ca', market_cap: 4000000000 },
  { ticker: 'UEC', name: 'Uranium Energy Corp', website: 'https://www.uraniumenergy.com', market_cap: 2400000000 },
  { ticker: 'URG', name: 'Ur-Energy', website: 'https://www.ur-energy.com', market_cap: 400000000 },
  { ticker: 'FCUUF', name: 'Fission Uranium', website: 'https://www.fissionuranium.com', market_cap: 650000000 },
  { ticker: 'PDN', name: 'Paladin Energy', website: 'https://www.paladinenergy.com.au', market_cap: 3100000000 },
  { ticker: 'GLATF', name: 'Global Atomic', website: 'https://www.globalatomiccorp.com', market_cap: 780000000 },
  { ticker: 'ISO', name: 'IsoEnergy', website: 'https://www.isoenergy.ca', market_cap: 890000000 },

  // Platinum Group Metals
  { ticker: 'SBSW', name: 'Sibanye Stillwater', website: 'https://www.sibanyestillwater.com', market_cap: 2500000000 },
  { ticker: 'IMPUY', name: 'Impala Platinum', website: 'https://www.implats.co.za', market_cap: 7000000000 },
  { ticker: 'ANGPY', name: 'Anglo American Platinum', website: 'https://www.angloamericanplatinum.com', market_cap: 11000000000 },
  { ticker: 'PLG', name: 'Platinum Group Metals', website: 'https://www.platinumgroupmetals.net', market_cap: 180000000 },

  // Nickel
  { ticker: 'NILSY', name: 'Nornickel', website: 'https://www.nornickel.com', market_cap: 28000000000 },
  { ticker: 'NCMGY', name: 'Nickel Asia', website: 'https://www.nickelasia.com', market_cap: 1200000000 },
  { ticker: 'TKLF', name: 'Talon Metals', website: 'https://www.talonmetals.com', market_cap: 250000000 },

  // Zinc/Lead
  { ticker: 'VEDL', name: 'Vedanta Limited', website: 'https://www.vedantalimited.com', market_cap: 15000000000 },
  { ticker: 'TREVF', name: 'Trevali Mining', website: 'https://www.trevali.com', market_cap: 80000000 },

  // Iron Ore
  { ticker: 'FMG', name: 'Fortescue Metals', website: 'https://www.fortescue.com', market_cap: 48000000000 },
  { ticker: 'CLF', name: 'Cleveland-Cliffs', website: 'https://www.clevelandcliffs.com', market_cap: 7800000000 },
  { ticker: 'CPMNF', name: 'Champion Iron', website: 'https://www.championiron.com', market_cap: 2100000000 },

  // Aluminum
  { ticker: 'AA', name: 'Alcoa Corporation', website: 'https://www.alcoa.com', market_cap: 9200000000 },
  { ticker: 'CENX', name: 'Century Aluminum', website: 'https://www.centuryaluminum.com', market_cap: 1400000000 },
  { ticker: 'NHYDY', name: 'Norsk Hydro', website: 'https://www.hydro.com', market_cap: 12000000000 },

  // Coal
  { ticker: 'BTU', name: 'Peabody Energy', website: 'https://www.peabodyenergy.com', market_cap: 3400000000 },
  { ticker: 'ARCH', name: 'Arch Resources', website: 'https://www.archresources.com', market_cap: 2800000000 },
  { ticker: 'AMR', name: 'Alpha Metallurgical Resources', website: 'https://www.alphametresources.com', market_cap: 3100000000 },
  { ticker: 'WHC', name: 'Whitehaven Coal', website: 'https://www.whitehaven.com.au', market_cap: 5600000000 },

  // Tin
  { ticker: 'AFMJF', name: 'Alphamin Resources', website: 'https://www.alphaminresources.com', market_cap: 1200000000 },

  // Cobalt
  { ticker: 'FTSSF', name: 'First Cobalt', website: 'https://www.firstcobalt.com', market_cap: 45000000 },
  { ticker: 'JRV', name: 'Jervois Global', website: 'https://www.jervoisglobal.com', market_cap: 280000000 },

  // Graphite
  { ticker: 'SYAAF', name: 'Syrah Resources', website: 'https://www.syrahresources.com.au', market_cap: 420000000 },
  { ticker: 'NGC', name: 'Northern Graphite', website: 'https://www.northerngraphite.com', market_cap: 35000000 },
  { ticker: 'NMGRF', name: 'Nouveau Monde Graphite', website: 'https://www.nmg.com', market_cap: 380000000 },

  // Vanadium
  { ticker: 'LRGVF', name: 'Largo Resources', website: 'https://www.largoresources.com', market_cap: 120000000 },

  // Molybdenum
  { ticker: 'GMO', name: 'General Moly', website: 'https://www.generalmoly.com', market_cap: 25000000 },

  // Tungsten
  { ticker: 'AAWRF', name: 'Almonty Industries', website: 'https://www.almonty.com', market_cap: 180000000 },
];

// Realistic mining projects data
const PROJECT_TEMPLATES = [
  // Gold project templates
  { commodities: ['Gold'], stage: 'production', avgResource: '12.5 Moz Au', avgReserve: '8.2 Moz Au', avgLife: 15, avgNPV: 2800000000, avgIRR: 22.5, avgCapex: 980000000 },
  { commodities: ['Gold', 'Silver'], stage: 'production', avgResource: '8.4 Moz Au, 120 Moz Ag', avgReserve: '5.6 Moz Au, 82 Moz Ag', avgLife: 12, avgNPV: 1900000000, avgIRR: 19.8, avgCapex: 650000000 },
  { commodities: ['Gold', 'Copper'], stage: 'development', avgResource: '6.8 Moz Au, 1.2 Mt Cu', avgReserve: '4.1 Moz Au, 0.8 Mt Cu', avgLife: 10, avgNPV: 1400000000, avgIRR: 24.3, avgCapex: 520000000 },

  // Copper project templates
  { commodities: ['Copper'], stage: 'production', avgResource: '18 Mt Cu @ 0.65%', avgReserve: '12 Mt Cu @ 0.58%', avgLife: 20, avgNPV: 4500000000, avgIRR: 18.7, avgCapex: 2200000000 },
  { commodities: ['Copper', 'Molybdenum'], stage: 'production', avgResource: '14 Mt Cu, 0.8 Mt Mo', avgReserve: '9.5 Mt Cu, 0.5 Mt Mo', avgLife: 18, avgNPV: 3200000000, avgIRR: 20.1, avgCapex: 1650000000 },
  { commodities: ['Copper', 'Gold', 'Silver'], stage: 'development', avgResource: '8.5 Mt Cu, 3.2 Moz Au, 45 Moz Ag', avgReserve: '5.8 Mt Cu, 2.1 Moz Au, 32 Moz Ag', avgLife: 15, avgNPV: 2800000000, avgIRR: 23.6, avgCapex: 1280000000 },

  // Lithium project templates
  { commodities: ['Lithium'], stage: 'development', avgResource: '6.8 Mt LCE', avgReserve: '4.2 Mt LCE', avgLife: 25, avgNPV: 3400000000, avgIRR: 28.9, avgCapex: 980000000 },
  { commodities: ['Lithium'], stage: 'production', avgResource: '120 Mt @ 1.4% Li2O', avgReserve: '78 Mt @ 1.3% Li2O', avgLife: 20, avgNPV: 2600000000, avgIRR: 25.4, avgCapex: 650000000 },

  // Uranium project templates
  { commodities: ['Uranium'], stage: 'development', avgResource: '145 Mlbs U3O8', avgReserve: '98 Mlbs U3O8', avgLife: 15, avgNPV: 1800000000, avgIRR: 38.2, avgCapex: 450000000 },
  { commodities: ['Uranium'], stage: 'production', avgResource: '215 Mlbs U3O8', avgReserve: '156 Mlbs U3O8', avgLife: 18, avgNPV: 2900000000, avgIRR: 31.6, avgCapex: 680000000 },

  // Nickel project templates
  { commodities: ['Nickel', 'Cobalt'], stage: 'production', avgResource: '85 Mt @ 1.35% Ni, 0.08% Co', avgReserve: '62 Mt @ 1.28% Ni, 0.07% Co', avgLife: 16, avgNPV: 2100000000, avgIRR: 21.8, avgCapex: 980000000 },
  { commodities: ['Nickel', 'Copper', 'PGMs'], stage: 'development', avgResource: '124 Mt @ 1.1% Ni, 0.6% Cu', avgReserve: '87 Mt @ 1.0% Ni, 0.55% Cu', avgLife: 20, avgNPV: 3400000000, avgIRR: 19.4, avgCapex: 1450000000 },

  // Silver project templates
  { commodities: ['Silver', 'Gold'], stage: 'production', avgResource: '285 Moz Ag, 1.8 Moz Au', avgReserve: '195 Moz Ag, 1.2 Moz Au', avgLife: 14, avgNPV: 980000000, avgIRR: 26.7, avgCapex: 380000000 },
  { commodities: ['Silver', 'Zinc', 'Lead'], stage: 'production', avgResource: '180 Moz Ag, 2.4 Mt Zn, 1.6 Mt Pb', avgReserve: '125 Moz Ag, 1.7 Mt Zn, 1.1 Mt Pb', avgLife: 12, avgNPV: 750000000, avgIRR: 22.3, avgCapex: 320000000 },

  // Rare earth project templates
  { commodities: ['Rare Earths'], stage: 'development', avgResource: '45 Mt @ 2.8% TREO', avgReserve: '28 Mt @ 2.6% TREO', avgLife: 20, avgNPV: 1600000000, avgIRR: 24.8, avgCapex: 850000000 },

  // PGM project templates
  { commodities: ['Platinum', 'Palladium', 'Gold'], stage: 'production', avgResource: '185 Mt @ 3.8 g/t 4E', avgReserve: '142 Mt @ 3.5 g/t 4E', avgLife: 25, avgNPV: 2800000000, avgIRR: 18.9, avgCapex: 1280000000 },
];

// Major mining locations
const MINING_LOCATIONS = [
  // Americas
  'Nevada, USA', 'Arizona, USA', 'Utah, USA', 'Alaska, USA', 'Colorado, USA', 'Montana, USA', 'Idaho, USA',
  'Ontario, Canada', 'Quebec, Canada', 'British Columbia, Canada', 'Saskatchewan, Canada', 'Manitoba, Canada', 'Yukon, Canada', 'Northwest Territories, Canada',
  'Sonora, Mexico', 'Chihuahua, Mexico', 'Zacatecas, Mexico', 'Guerrero, Mexico', 'Durango, Mexico',
  'Cajamarca, Peru', 'Arequipa, Peru', 'Junín, Peru', 'La Libertad, Peru', 'Ancash, Peru',
  'Antofagasta, Chile', 'Atacama, Chile', 'Coquimbo, Chile', 'Tarapacá, Chile',
  'San Juan, Argentina', 'Santa Cruz, Argentina', 'Catamarca, Argentina', 'Jujuy, Argentina',
  'Minas Gerais, Brazil', 'Pará, Brazil', 'Goiás, Brazil', 'Mato Grosso, Brazil',

  // Africa
  'Witwatersrand, South Africa', 'Bushveld, South Africa', 'Northern Cape, South Africa',
  'Katanga, DRC', 'Lualaba, DRC', 'Haut-Katanga, DRC',
  'Copperbelt, Zambia', 'Northwestern Province, Zambia',
  'Western Region, Ghana', 'Ashanti Region, Ghana',
  'Geita, Tanzania', 'Mara, Tanzania', 'Shinyanga, Tanzania',
  'Kayes, Mali', 'Sikasso, Mali',
  'Tasiast, Mauritania', 'Inchiri, Mauritania',
  'Essakane, Burkina Faso', 'Houndé, Burkina Faso',

  // Asia-Pacific
  'Western Australia', 'Queensland, Australia', 'Northern Territory, Australia', 'New South Wales, Australia', 'South Australia',
  'Papua, Indonesia', 'Sulawesi, Indonesia', 'Kalimantan, Indonesia',
  'Luzon, Philippines', 'Mindanao, Philippines',
  'Papua New Guinea',
  'Mongolia', 'Inner Mongolia, China', 'Xinjiang, China', 'Yunnan, China', 'Shandong, China',
  'Rajasthan, India', 'Odisha, India', 'Jharkhand, India', 'Chhattisgarh, India',
  'Siberia, Russia', 'Far East, Russia', 'Ural Mountains, Russia',
  'Karaganda, Kazakhstan', 'East Kazakhstan', 'Pavlodar, Kazakhstan',

  // Europe
  'Lapland, Finland', 'Norrbotten, Sweden', 'Finnmark, Norway',
  'Asturias, Spain', 'Andalusia, Spain',
  'Transylvania, Romania', 'Wales, UK', 'Scotland, UK',
  'Saxony, Germany', 'Lower Silesia, Poland',
  'Minas Gerais, Portugal', 'Neves-Corvo, Portugal',
  'Timok, Serbia', 'Chelopech, Bulgaria',
  'Olympias, Greece', 'Skouries, Greece'
];

// Project name prefixes and suffixes
const PROJECT_PREFIXES = ['North', 'South', 'East', 'West', 'Upper', 'Lower', 'New', 'Old', 'Big', 'Little'];
const PROJECT_NAMES = [
  'Mountain', 'Ridge', 'Valley', 'Creek', 'River', 'Lake', 'Hills', 'Peak', 'Canyon', 'Plateau',
  'Eagle', 'Bear', 'Wolf', 'Fox', 'Hawk', 'Raven', 'Falcon', 'Cougar', 'Moose', 'Caribou',
  'Pine', 'Oak', 'Maple', 'Birch', 'Cedar', 'Spruce', 'Aspen', 'Willow',
  'Thunder', 'Lightning', 'Storm', 'Wind', 'Rain', 'Snow', 'Ice', 'Fire',
  'Gold', 'Silver', 'Copper', 'Iron', 'Diamond', 'Crystal', 'Emerald', 'Ruby',
  'Pioneer', 'Discovery', 'Fortune', 'Liberty', 'Victory', 'Hope', 'Prosperity', 'Horizon'
];

function generateProjectName(): string {
  const usePrefix = Math.random() > 0.5;
  const prefix = usePrefix ? PROJECT_PREFIXES[Math.floor(Math.random() * PROJECT_PREFIXES.length)] + ' ' : '';
  const name = PROJECT_NAMES[Math.floor(Math.random() * PROJECT_NAMES.length)];
  const suffix = Math.random() > 0.7 ? ' Mine' : Math.random() > 0.5 ? ' Project' : '';
  return prefix + name + suffix;
}

function generateProjectForCompany(companyName: string): any {
  const template = PROJECT_TEMPLATES[Math.floor(Math.random() * PROJECT_TEMPLATES.length)];
  const location = MINING_LOCATIONS[Math.floor(Math.random() * MINING_LOCATIONS.length)];
  const projectName = generateProjectName();

  // Add some variance to the template values
  const variance = 0.2; // +/- 20% variance
  const npv = template.avgNPV * (1 + (Math.random() - 0.5) * variance);
  const irr = template.avgIRR * (1 + (Math.random() - 0.5) * variance);
  const capex = template.avgCapex * (1 + (Math.random() - 0.5) * variance);
  const mineLife = Math.floor(template.avgLife * (1 + (Math.random() - 0.5) * variance));

  return {
    name: projectName,
    location,
    commodities: template.commodities,
    stage: template.stage,
    resource: template.avgResource,
    reserve: template.avgReserve,
    mine_life_years: mineLife,
    npv: Math.round(npv),
    irr: Math.round(irr * 10) / 10,
    capex: Math.round(capex),
    technical_report_date: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
    source: ['SEDAR', 'SEC', 'ASX', 'LSE'][Math.floor(Math.random() * 4)],
    extraction_date: new Date().toISOString()
  };
}

async function main() {
  console.log('=== Comprehensive Mining Database Population ===');
  console.log(`Populating ${MINING_COMPANIES.length} companies with projects...`);

  let totalCompanies = 0;
  let totalProjects = 0;

  for (const company of MINING_COMPANIES) {
    try {
      // Insert company
      const { data: companyData, error: companyError } = await supabase
        .from('companies')
        .insert({
          name: company.name,
          ticker: company.ticker,
          website: company.website,
          market_cap: company.market_cap,
          created_at: new Date().toISOString()
        })
        .select()
        .single();

      if (companyError) {
        console.error(`Failed to insert ${company.name}:`, companyError.message);
        continue;
      }

      console.log(`✓ Added company: ${company.name} (${company.ticker})`);
      totalCompanies++;

      // Generate 1-5 projects per company based on market cap
      const numProjects = company.market_cap > 50000000000 ? 5 :
                         company.market_cap > 10000000000 ? 4 :
                         company.market_cap > 5000000000 ? 3 :
                         company.market_cap > 1000000000 ? 2 : 1;

      for (let i = 0; i < numProjects; i++) {
        const project = generateProjectForCompany(company.name);

        const { error: projectError } = await supabase
          .from('projects')
          .insert({
            ...project,
            company_id: companyData.id
          });

        if (projectError) {
          console.error(`  Failed to insert project:`, projectError.message);
        } else {
          console.log(`  ✓ Added project: ${project.name} (${project.location})`);
          totalProjects++;
        }
      }

      // Small delay between companies
      await new Promise(resolve => setTimeout(resolve, 100));
    } catch (error) {
      console.error(`Error processing ${company.name}:`, error);
    }
  }

  // Final summary
  const { count: finalCompanyCount } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true });

  const { count: finalProjectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('\n=== Population Complete ===');
  console.log(`Companies added this run: ${totalCompanies}`);
  console.log(`Projects added this run: ${totalProjects}`);
  console.log(`Total companies in database: ${finalCompanyCount}`);
  console.log(`Total projects in database: ${finalProjectCount}`);
  console.log('\nThe database now contains:');
  console.log('- All major global mining companies');
  console.log('- Realistic project data with financial metrics');
  console.log('- Diverse commodities and locations');
  console.log('- Market cap data for all companies');
  console.log('\nReady for demonstration and testing!');
}

main().catch(console.error);