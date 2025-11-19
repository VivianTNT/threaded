import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

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
  'Nevada, USA', 'Arizona, USA', 'Utah, USA', 'Alaska, USA', 'Colorado, USA',
  'Ontario, Canada', 'Quebec, Canada', 'British Columbia, Canada', 'Saskatchewan, Canada',
  'Sonora, Mexico', 'Chihuahua, Mexico', 'Zacatecas, Mexico',
  'Cajamarca, Peru', 'Arequipa, Peru', 'Junín, Peru',
  'Antofagasta, Chile', 'Atacama, Chile',
  'San Juan, Argentina', 'Santa Cruz, Argentina',
  'Minas Gerais, Brazil', 'Pará, Brazil',

  // Africa
  'Witwatersrand, South Africa', 'Bushveld, South Africa',
  'Katanga, DRC', 'Lualaba, DRC',
  'Copperbelt, Zambia',
  'Western Region, Ghana',
  'Geita, Tanzania',
  'Kayes, Mali',

  // Asia-Pacific
  'Western Australia', 'Queensland, Australia', 'Northern Territory, Australia',
  'Papua, Indonesia',
  'Papua New Guinea',
  'Mongolia',
  'Siberia, Russia'
];

// Project name prefixes and suffixes
const PROJECT_NAMES = [
  'Boddington', 'Tanami', 'Peñasquito', 'Cerro Negro', 'Yanacocha',
  'Nevada Gold Mines', 'Kibali', 'Pueblo Viejo', 'Loulo-Gounkoto', 'Veladero',
  'Grasberg', 'Morenci', 'Cerro Verde', 'Bagdad',
  'Olympic Dam', 'Escondida', 'Pampa Norte', 'Antamina',
  'Oyu Tolgoi', 'Kennecott', 'Resolution', 'Winu',
  'Salobo', 'Sudbury', 'Voisey\'s Bay', 'Thompson',
  'Greenbushes', 'Salar de Atacama', 'Thacker Pass', 'Carolina',
  'Cigar Lake', 'McArthur River', 'Arrow', 'Wheeler River',
  'Cortez', 'Carlin', 'Turquoise Ridge', 'Goldstrike', 'Hemlo',
  'Canadian Malartic', 'Detour Lake', 'Musselwhite', 'Rainy River',
  'Fosterville', 'Cadia', 'Telfer', 'Tanami', 'St Ives',
  'Tropicana', 'Sunrise Dam', 'Granny Smith', 'Jundee',
  'Lihir', 'Porgera', 'Ok Tedi', 'Wafi-Golpu',
  'Fruta del Norte', 'Pascua-Lama', 'Caspiche', 'La Coipa',
  'Collahuasi', 'Los Bronces', 'El Teniente', 'Andina',
  'Kumtor', 'Centerra', 'Mount Milligan', 'Öksüt',
  'Kisladag', 'Lamaque', 'Efemçukuru', 'Olympias',
  'Skouries', 'Perama', 'Stratoni', 'Minto',
  'Gibraltar', 'Red Chris', 'Mount Polley', 'Highland Valley',
  'Antapaccay', 'Las Bambas', 'Constancia', 'Tintaya',
  'Quellaveco', 'Cuajone', 'Toquepala', 'La Granja'
];

function generateProjectName(): string {
  return PROJECT_NAMES[Math.floor(Math.random() * PROJECT_NAMES.length)];
}

async function main() {
  console.log('Populating projects for existing companies...');

  // Get all companies
  const { data: companies, error: companiesError } = await supabase
    .from('companies')
    .select('id, name, ticker, market_cap');

  if (companiesError || !companies) {
    console.error('Failed to fetch companies:', companiesError);
    return;
  }

  console.log(`Found ${companies.length} companies`);

  let totalProjects = 0;
  const usedProjectNames = new Set<string>();

  for (const company of companies) {
    try {
      // Determine number of projects based on market cap
      const numProjects = company.market_cap > 50000000000 ? 4 :
                         company.market_cap > 20000000000 ? 3 :
                         company.market_cap > 5000000000 ? 2 : 1;

      for (let i = 0; i < numProjects; i++) {
        const template = PROJECT_TEMPLATES[Math.floor(Math.random() * PROJECT_TEMPLATES.length)];
        const location = MINING_LOCATIONS[Math.floor(Math.random() * MINING_LOCATIONS.length)];

        // Generate unique project name
        let projectName = generateProjectName();
        let attempts = 0;
        while (usedProjectNames.has(projectName) && attempts < 10) {
          projectName = generateProjectName() + ' ' + ['North', 'South', 'East', 'West', 'Extension', 'Deep'][Math.floor(Math.random() * 6)];
          attempts++;
        }
        usedProjectNames.add(projectName);

        // Add variance to template values
        const variance = 0.25;
        const npv = template.avgNPV * (1 + (Math.random() - 0.5) * variance);
        const irr = template.avgIRR * (1 + (Math.random() - 0.5) * variance);
        const capex = template.avgCapex * (1 + (Math.random() - 0.5) * variance);
        const mineLife = Math.floor(template.avgLife * (1 + (Math.random() - 0.5) * variance));

        const projectData = {
          name: projectName,
          company_id: company.id,
          location,
          commodities: template.commodities,
          stage: template.stage,
          resource: template.avgResource,
          reserve: template.avgReserve,
          mine_life_years: mineLife,
          npv: Math.round(npv),
          irr: Math.round(irr * 10) / 10,
          capex: Math.round(capex),
          technical_report_date: new Date(Date.now() - Math.random() * 2 * 365 * 24 * 60 * 60 * 1000).toISOString(),
          source: ['SEDAR', 'SEC', 'ASX', 'LSE'][Math.floor(Math.random() * 4)],
          document_storage_path: `https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/technical-reports/${company.ticker}_${Date.now()}.pdf`
        };

        const { error: projectError } = await supabase
          .from('projects')
          .insert(projectData);

        if (projectError) {
          console.error(`Failed to insert project for ${company.name}:`, projectError.message);
        } else {
          console.log(`✓ Added project: ${projectName} for ${company.name}`);
          totalProjects++;
        }
      }
    } catch (error) {
      console.error(`Error processing company ${company.name}:`, error);
    }
  }

  // Final count
  const { count: finalProjectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('\n=== Project Population Complete ===');
  console.log(`Projects added: ${totalProjects}`);
  console.log(`Total projects in database: ${finalProjectCount}`);
}

main().catch(console.error);