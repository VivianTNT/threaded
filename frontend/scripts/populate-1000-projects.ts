import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  throw new Error('Missing Supabase environment variables');
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false
  }
});

// Real mining companies and their known projects
const miningCompanies = [
  { name: 'Rio Tinto', country: 'UK/Australia' },
  { name: 'BHP Group', country: 'Australia' },
  { name: 'Vale', country: 'Brazil' },
  { name: 'Glencore', country: 'Switzerland' },
  { name: 'Anglo American', country: 'UK' },
  { name: 'Freeport-McMoRan', country: 'USA' },
  { name: 'Newmont Corporation', country: 'USA' },
  { name: 'Barrick Gold', country: 'Canada' },
  { name: 'Albemarle Corporation', country: 'USA' },
  { name: 'SQM', country: 'Chile' },
  { name: 'Ganfeng Lithium', country: 'China' },
  { name: 'Tianqi Lithium', country: 'China' },
  { name: 'Pilbara Minerals', country: 'Australia' },
  { name: 'Mineral Resources', country: 'Australia' },
  { name: 'Liontown Resources', country: 'Australia' },
  { name: 'Arcadium Lithium', country: 'USA' },
  { name: 'Agnico Eagle Mines', country: 'Canada' },
  { name: 'AngloGold Ashanti', country: 'South Africa' },
  { name: 'Zijin Mining', country: 'China' },
  { name: 'Gold Fields', country: 'South Africa' },
  { name: 'Kinross Gold', country: 'Canada' },
  { name: 'Polyus', country: 'Russia' },
  { name: 'Newcrest Mining', country: 'Australia' },
  { name: 'Northern Star Resources', country: 'Australia' },
  { name: 'Evolution Mining', country: 'Australia' },
  { name: 'Harmony Gold', country: 'South Africa' },
  { name: 'CMOC Group', country: 'China' },
  { name: 'Teck Resources', country: 'Canada' },
  { name: 'First Quantum Minerals', country: 'Canada' },
  { name: 'Lundin Mining', country: 'Canada' },
  { name: 'Antofagasta', country: 'Chile' },
  { name: 'Southern Copper', country: 'Peru/Mexico' },
  { name: 'Ivanhoe Mines', country: 'Canada' },
  { name: 'Hudbay Minerals', country: 'Canada' },
  { name: 'Capstone Copper', country: 'Canada' },
  { name: 'Codelco', country: 'Chile' },
  { name: 'Norilsk Nickel', country: 'Russia' },
  { name: 'Nickel 28', country: 'Canada' },
  { name: 'IGO Limited', country: 'Australia' },
  { name: 'Sibanye Stillwater', country: 'South Africa' },
  { name: 'Impala Platinum', country: 'South Africa' },
  { name: 'Anglo American Platinum', country: 'South Africa' },
  { name: 'Lithium Americas', country: 'Canada' },
  { name: 'Piedmont Lithium', country: 'USA' },
  { name: 'Livent Corporation', country: 'USA' },
  { name: 'Patriot Battery Metals', country: 'Canada' },
  { name: 'Core Lithium', country: 'Australia' },
  { name: 'Latin Resources', country: 'Australia' },
  { name: 'Energy Fuels', country: 'USA' },
  { name: 'MP Materials', country: 'USA' },
  { name: 'Lynas Rare Earths', country: 'Australia' },
  { name: 'Torngat Metals', country: 'Canada' },
  { name: 'Syrah Resources', country: 'Australia' },
  { name: 'NextSource Materials', country: 'Canada' },
  { name: 'Graphite One', country: 'USA' },
  { name: 'Nouveau Monde Graphite', country: 'Canada' },
  { name: 'Endeavour Mining', country: 'Côte d\'Ivoire' },
  { name: 'B2Gold', country: 'Canada' },
  { name: 'Centamin', country: 'UK/Egypt' },
  { name: 'Perseus Mining', country: 'Australia' },
  { name: 'Resolute Mining', country: 'Australia' },
  { name: 'West African Resources', country: 'Australia' },
  { name: 'Sandstorm Gold', country: 'Canada' },
  { name: 'Wheaton Precious Metals', country: 'Canada' },
  { name: 'Franco-Nevada', country: 'Canada' },
  { name: 'Royal Gold', country: 'USA' },
  { name: 'Eldorado Gold', country: 'Canada' },
  { name: 'IAMGOLD', country: 'Canada' },
  { name: 'Calibre Mining', country: 'Canada' },
  { name: 'Centerra Gold', country: 'Canada' },
  { name: 'Pan American Silver', country: 'Canada' },
  { name: 'Hecla Mining', country: 'USA' },
  { name: 'Coeur Mining', country: 'USA' },
  { name: 'Fortuna Silver Mines', country: 'Canada' },
  { name: 'MAG Silver', country: 'Canada' },
  { name: 'Hochschild Mining', country: 'Peru' },
  { name: 'Fresnillo', country: 'Mexico' },
  { name: 'Polymetal International', country: 'Russia' },
  { name: 'Buenaventura', country: 'Peru' },
  { name: 'Alamos Gold', country: 'Canada' },
  { name: 'OceanaGold', country: 'Australia/New Zealand' },
  { name: 'Equinox Gold', country: 'Canada' },
  { name: 'Osisko Gold Royalties', country: 'Canada' },
  { name: 'Sandfire Resources', country: 'Australia' },
  { name: '29Metals', country: 'Australia' },
  { name: 'Aeris Resources', country: 'Australia' },
  { name: 'Oz Minerals', country: 'Australia' },
  { name: 'Copper Mountain Mining', country: 'Canada' },
  { name: 'Taseko Mines', country: 'Canada' },
  { name: 'Ero Copper', country: 'Canada' },
  { name: 'Trilogy Metals', country: 'USA' },
  { name: 'NovaCopper', country: 'USA' },
  { name: 'Antipa Minerals', country: 'Australia' },
  { name: 'Western Areas', country: 'Australia' },
  { name: 'Mincor Resources', country: 'Australia' },
  { name: 'Independence Group', country: 'Australia' },
  { name: 'St Barbara', country: 'Australia' },
  { name: 'Regis Resources', country: 'Australia' },
  { name: 'Silver Lake Resources', country: 'Australia' },
  { name: 'Ramelius Resources', country: 'Australia' },
  { name: 'Emerald Resources', country: 'Australia' },
  { name: 'De Grey Mining', country: 'Australia' },
];

const commoditiesList = [
  ['Lithium'],
  ['Copper'],
  ['Gold'],
  ['Nickel'],
  ['Cobalt'],
  ['Silver'],
  ['Zinc'],
  ['Lead'],
  ['Iron Ore'],
  ['Rare Earths'],
  ['Graphite'],
  ['Uranium'],
  ['Platinum'],
  ['Palladium'],
  ['Manganese'],
  ['Vanadium'],
  ['Molybdenum'],
  ['Tungsten'],
  ['Tin'],
  ['Bauxite'],
  ['Lithium', 'Tantalum'],
  ['Copper', 'Gold'],
  ['Copper', 'Cobalt'],
  ['Nickel', 'Cobalt'],
  ['Gold', 'Silver'],
  ['Gold', 'Copper'],
  ['Zinc', 'Lead'],
  ['Zinc', 'Silver'],
  ['PGMs', 'Nickel'],
  ['Rare Earths', 'Uranium'],
];

const locations = [
  'Western Australia', 'Queensland', 'New South Wales', 'Northern Territory', 'South Australia', 'Victoria',
  'Ontario', 'Quebec', 'British Columbia', 'Saskatchewan', 'Manitoba', 'Yukon', 'Northwest Territories',
  'Nevada', 'Arizona', 'Alaska', 'Utah', 'Montana', 'Idaho', 'California', 'Wyoming', 'New Mexico',
  'Chile', 'Peru', 'Brazil', 'Argentina', 'Mexico', 'Colombia', 'Ecuador', 'Bolivia',
  'DRC', 'South Africa', 'Ghana', 'Mali', 'Burkina Faso', 'Tanzania', 'Zambia', 'Zimbabwe', 'Namibia', 'Botswana',
  'China', 'Mongolia', 'Kazakhstan', 'Russia', 'Philippines', 'Indonesia', 'Papua New Guinea',
  'Finland', 'Sweden', 'Norway', 'Poland', 'Serbia', 'Spain', 'Portugal', 'Turkey',
];

const stages = [
  'Exploration',
  'Advanced Exploration',
  'Pre-Feasibility',
  'Feasibility',
  'Development',
  'Construction',
  'Production',
  'Care & Maintenance',
];

const statuses = [
  'Active',
  'Active',
  'Active',
  'Active',
  'On Hold',
  'Suspended',
];

// Project name prefixes and suffixes
const projectPrefixes = [
  'Mount', 'Lake', 'River', 'Peak', 'Valley', 'Creek', 'Ridge', 'Hill', 'East', 'West', 'North', 'South', 'Central',
  'Upper', 'Lower', 'New', 'Old', 'Big', 'Little', 'Great', 'High', 'Deep', 'Twin', 'Triple', 'Silver', 'Golden',
  'Red', 'Blue', 'Green', 'Black', 'White', 'Copper', 'Iron', 'Star', 'Crown', 'Royal', 'Eagle', 'Phoenix',
];

const projectNames = [
  'Aurora', 'Phoenix', 'Apex', 'Summit', 'Horizon', 'Pinnacle', 'Vista', 'Cascade', 'Thunder', 'Lightning',
  'Sunrise', 'Sunset', 'Diamond', 'Ruby', 'Emerald', 'Sapphire', 'Crystal', 'Titan', 'Atlas', 'Olympus',
  'Marathon', 'Victory', 'Fortune', 'Prosperity', 'Liberty', 'Independence', 'Pioneer', 'Frontier', 'Explorer',
  'Discovery', 'Strike', 'Bonanza', 'Eureka', 'El Dorado', 'Klondike', 'Comstock', 'Carlin', 'Cortez',
  'Escondida', 'Chuquicamata', 'Oyu Tolgoi', 'Grasberg', 'Bingham', 'Olympic Dam', 'Boddington', 'Newmont',
  'Pilgangoora', 'Wodgina', 'Greenbushes', 'Marion', 'Kathleen Valley', 'Finniss', 'Goulamina', 'Ewoyaa',
  'Arcadia', 'Bikita', 'Manono', 'Bald Hill', 'Whabouchi', 'Separation Rapids', 'Georgia Lake', 'Moblan',
  'James Bay', 'Rose', 'Authier', 'Corvette', 'CV13', 'CV5', 'Delta', 'Spodumene Gabbro',
];

function generateResourceEstimate(commodities: string[], stage: string): string {
  if (stage === 'Exploration' || stage === 'Advanced Exploration') {
    return Math.random() < 0.3 ? `${(Math.random() * 50 + 10).toFixed(1)} Mt @ ${(Math.random() * 3 + 0.5).toFixed(2)}% ${commodities[0]}` : '';
  }
  const tonnage = (Math.random() * 500 + 20).toFixed(1);
  const grade = (Math.random() * 4 + 0.3).toFixed(2);
  return `${tonnage} Mt @ ${grade}% ${commodities[0]} (Indicated & Inferred)`;
}

function generateReserveEstimate(stage: string, resourceEstimate: string): string {
  if (stage === 'Exploration' || stage === 'Advanced Exploration' || stage === 'Pre-Feasibility' || !resourceEstimate) {
    return '';
  }
  const match = resourceEstimate.match(/([\d.]+)\s*Mt\s*@\s*([\d.]+)%/);
  if (!match) return '';
  const tonnage = (parseFloat(match[1]) * (Math.random() * 0.4 + 0.4)).toFixed(1);
  const grade = (parseFloat(match[2]) * (Math.random() * 0.2 + 0.9)).toFixed(2);
  return `${tonnage} Mt @ ${grade}% (Proven & Probable)`;
}

function generateDescription(name: string, commodities: string[], location: string, stage: string): string {
  const templates = [
    `The ${name} project is a ${commodities.join('-')} project located in ${location}. ${stage === 'Production' ? 'Currently in production with' : stage === 'Development' || stage === 'Construction' ? 'Under development with' : 'Exploration focused on'} world-class ${commodities[0].toLowerCase()} deposits.`,
    `${name} represents a significant ${commodities.join('/')} opportunity in ${location}. The project is at ${stage.toLowerCase()} stage with promising ${commodities[0].toLowerCase()} mineralization.`,
    `Located in ${location}, ${name} is a ${commodities.join(' and ')} ${stage.toLowerCase()} project with strong resource potential and favorable infrastructure.`,
    `${name} project in ${location} targets ${commodities.join(', ')} mineralization. ${stage === 'Production' ? 'Operating mine with' : 'Project features'} excellent metallurgy and jurisdiction support.`,
    `Advanced ${commodities.join('-')} project situated in the ${location} region. Currently in ${stage.toLowerCase()} with comprehensive drilling programs and resource modeling.`,
  ];
  return templates[Math.floor(Math.random() * templates.length)];
}

function generateOwnershipPercentage(): number {
  const percentages = [100, 100, 100, 80, 70, 60, 51, 50, 49, 40, 30, 25, 20];
  return percentages[Math.floor(Math.random() * percentages.length)];
}

async function getCompanyIdByName(companyName: string): Promise<string | null> {
  const { data } = await supabase
    .from('companies')
    .select('id')
    .ilike('name', companyName)
    .limit(1)
    .single();

  return data?.id || null;
}

async function generateProjects(count: number, startIndex: number = 0): Promise<any[]> {
  const projects: any[] = [];

  for (let i = startIndex; i < startIndex + count; i++) {
    const company = miningCompanies[i % miningCompanies.length];
    const prefix = projectPrefixes[Math.floor(Math.random() * projectPrefixes.length)];
    const baseName = projectNames[Math.floor(Math.random() * projectNames.length)];
    const name = Math.random() > 0.5 ? `${prefix} ${baseName}` : baseName;

    const commodities = commoditiesList[Math.floor(Math.random() * commoditiesList.length)];
    const location = locations[Math.floor(Math.random() * locations.length)];
    const stage = stages[Math.floor(Math.random() * stages.length)];
    const status = statuses[Math.floor(Math.random() * statuses.length)];

    const resourceEstimate = generateResourceEstimate(commodities, stage);
    const reserveEstimate = generateReserveEstimate(stage, resourceEstimate);
    const ownership = generateOwnershipPercentage();
    const description = generateDescription(name, commodities, location, stage);

    // Get company_id
    const companyId = await getCompanyIdByName(company.name);

    projects.push({
      company_id: companyId,
      name: `${name} Project`,
      location,
      stage,
      commodities,
      resource_estimate: resourceEstimate || null,
      reserve_estimate: reserveEstimate || null,
      ownership_percentage: ownership,
      status,
      description,
      urls: [],
      watchlist: false,
    });
  }

  return projects;
}

async function uploadProjects() {
  console.log('🚀 Starting to generate and upload 1000+ mining projects...\n');

  const BATCH_SIZE = 100;
  const TOTAL_PROJECTS = 1100;
  let totalUploaded = 0;

  try {
    // Get current count
    const { count: existingCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true });

    console.log(`📊 Existing projects in database: ${existingCount}\n`);

    for (let i = 0; i < TOTAL_PROJECTS; i += BATCH_SIZE) {
      const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
      const batchSize = Math.min(BATCH_SIZE, TOTAL_PROJECTS - i);

      console.log(`📦 Generating batch ${batchNumber} (${batchSize} projects)...`);
      const projects = await generateProjects(batchSize, i);

      console.log(`💾 Uploading batch ${batchNumber} to Supabase...`);
      const { data, error } = await supabase
        .from('projects')
        .insert(projects)
        .select();

      if (error) {
        console.error(`❌ Error uploading batch ${batchNumber}:`, error);
        throw error;
      }

      totalUploaded += data?.length || 0;
      console.log(`✅ Batch ${batchNumber} uploaded successfully! (${data?.length} projects)`);
      console.log(`   Progress: ${totalUploaded}/${TOTAL_PROJECTS} (${((totalUploaded / TOTAL_PROJECTS) * 100).toFixed(1)}%)\n`);

      // Small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Get final count
    const { count: finalCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true });

    console.log('\n✅ All projects uploaded successfully!');
    console.log('═══════════════════════════════════════');
    console.log(`📊 Total projects in database: ${finalCount}`);
    console.log(`📈 New projects added: ${totalUploaded}`);
    console.log('═══════════════════════════════════════\n');

  } catch (error) {
    console.error('❌ Error during upload:', error);
    throw error;
  }
}

// Run the upload
uploadProjects()
  .then(() => {
    console.log('🎉 Project population complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('💥 Failed:', error);
    process.exit(1);
  });
