#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Helper functions
function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min
}

function randomFloat(min: number, max: number, decimals: number = 2): number {
  return Number((Math.random() * (max - min) + min).toFixed(decimals))
}

function randomElement<T>(arr: T[]): T {
  return arr[randomInt(0, arr.length - 1)]
}

function randomElements<T>(arr: T[], min: number = 1, max: number = 3): T[] {
  const count = randomInt(min, Math.min(max, arr.length))
  const shuffled = [...arr].sort(() => Math.random() - 0.5)
  return shuffled.slice(0, count)
}

// Data definitions
const countries = [
  'Australia', 'Canada', 'United States', 'Chile', 'Peru', 'Brazil',
  'South Africa', 'Democratic Republic of Congo', 'Zambia', 'Ghana',
  'Tanzania', 'Mexico', 'Argentina', 'China', 'Russia', 'Kazakhstan',
  'Mongolia', 'Indonesia', 'Philippines', 'Papua New Guinea', 'Finland',
  'Sweden', 'Norway', 'Spain', 'Portugal', 'Poland', 'Turkey', 'Saudi Arabia',
  'Burkina Faso', 'Mali', 'Ivory Coast', 'Botswana', 'Namibia', 'Guinea'
]

const exchanges = ['NYSE', 'TSX', 'ASX', 'LSE', 'TSXV', 'AIM', 'JSE', 'SGX', 'HKE', 'NSX']

const commodities = [
  'Gold', 'Copper', 'Silver', 'Lithium', 'Nickel', 'Cobalt', 'Zinc', 'Lead',
  'Iron Ore', 'Platinum', 'Palladium', 'Tin', 'Tungsten', 'Rare Earth Elements',
  'Uranium', 'Graphite', 'Manganese', 'Vanadium', 'Molybdenum', 'Aluminum'
]

const stages = [
  'Exploration', 'Pre-Feasibility', 'Feasibility', 'Development',
  'Construction', 'Production', 'On Hold'
]

const statuses = ['Active', 'On Hold', 'Under Review', 'Expanding', 'Operational']

const companyPrefixes = [
  'Global', 'International', 'Pan', 'Trans', 'Continental', 'United',
  'Imperial', 'Royal', 'Pacific', 'Atlantic', 'Northern', 'Southern',
  'Eastern', 'Western', 'Central', 'Neo', 'Apex', 'Zenith', 'Quantum',
  'Stellar', 'Horizon', 'Pinnacle', 'Summit', 'Premier'
]

const companySuffixes = [
  'Mining', 'Resources', 'Minerals', 'Metals', 'Industries', 'Group',
  'Corporation', 'Holdings', 'Exploration', 'Development', 'Operations',
  'Ventures', 'Enterprises', 'Partners', 'Capital'
]

const projectNames = [
  'Eagle', 'Phoenix', 'Dragon', 'Falcon', 'Thunder', 'Lightning',
  'Mountain', 'Valley', 'River', 'Lake', 'Summit', 'Ridge', 'Peak',
  'Canyon', 'Mesa', 'Plateau', 'Highland', 'Lowland', 'North Star',
  'South Cross', 'Crown', 'Royal', 'King', 'Queen', 'Emperor', 'Atlas',
  'Titan', 'Olympus', 'Victoria', 'Aurora', 'Genesis', 'Omega', 'Alpha',
  'Beta', 'Gamma', 'Delta', 'Echo', 'Foxtrot', 'Sierra', 'Tango'
]

const locations = [
  'Red Lake District', 'Pilbara Region', 'Atacama Desert', 'Witwatersrand Basin',
  'Golden Triangle', 'Ring of Fire', 'Great Dyke', 'Copperbelt', 'Carlin Trend',
  'Yilgarn Craton', 'Superior Province', 'Fraser Range', 'Lachlan Fold Belt',
  'Andean Belt', 'Great Basin', 'Abitibi Belt', 'Kalgoorlie Region',
  'Kimberley Region', 'Bowen Basin', 'Hunter Valley', 'Queensland Outback',
  'Northern Territory', 'Saskatchewan', 'British Columbia', 'Nevada',
  'Arizona', 'New Mexico', 'Ontario', 'Quebec', 'Yukon', 'Nunavut'
]

// Generate Companies
async function generateCompanies(count: number) {
  console.log(`\n🏢 Generating ${count} companies...\n`)

  const companies = []

  for (let i = 0; i < count; i++) {
    const prefix = randomElement(companyPrefixes)
    const suffix = randomElement(companySuffixes)
    const name = `${prefix} ${suffix}`
    const country = randomElement(countries)
    const exchange = randomElement(exchanges)

    // Generate ticker based on company name
    const ticker = name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase() + randomInt(0, 99).toString().padStart(2, '0')

    const marketCap = randomFloat(10, 50000, 0) // $10M to $50B

    const company = {
      name,
      ticker,
      exchange,
      country,
      website: `https://www.${name.toLowerCase().replace(/\s+/g, '')}.com`,
      description: `${name} is a ${country}-based mining and minerals exploration company focused on ${randomElements(commodities, 1, 3).join(', ')} production and development.`,
      market_cap: marketCap,
      urls: [
        `https://www.${name.toLowerCase().replace(/\s+/g, '')}.com`,
        `https://investors.${name.toLowerCase().replace(/\s+/g, '')}.com`
      ],
      watchlist: Math.random() > 0.8,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    companies.push(company)

    if ((i + 1) % 50 === 0) {
      console.log(`   Generated ${i + 1}/${count} companies...`)
    }
  }

  console.log(`\n📤 Inserting ${companies.length} companies into database...\n`)

  // Insert in batches of 100
  const batchSize = 100
  const insertedCompanies = []

  for (let i = 0; i < companies.length; i += batchSize) {
    const batch = companies.slice(i, i + batchSize)

    const { data, error } = await supabase
      .from('companies')
      .insert(batch)
      .select('id, name')

    if (error) {
      console.error(`❌ Error inserting companies batch ${i / batchSize + 1}:`, error.message)
    } else {
      insertedCompanies.push(...(data || []))
      console.log(`   ✅ Inserted batch ${i / batchSize + 1} (${batch.length} companies)`)
    }
  }

  console.log(`\n✅ Successfully inserted ${insertedCompanies.length} companies\n`)
  return insertedCompanies
}

// Generate Projects
async function generateProjects(count: number, companies: Array<{ id: string; name: string }>) {
  console.log(`\n⛏️  Generating ${count} projects...\n`)

  const projects = []

  for (let i = 0; i < count; i++) {
    const company = randomElement(companies)
    const projectName = `${randomElement(projectNames)} ${randomElement(['Mine', 'Project', 'Deposit', 'Complex', 'Operation'])}`
    const location = randomElement(locations)
    const country = randomElement(countries)
    const stage = randomElement(stages)
    const status = randomElement(statuses)
    const projectCommodities = randomElements(commodities, 1, 3)

    // Financial metrics based on stage
    let npv = null
    let irr = null
    let capex = null

    if (['Feasibility', 'Development', 'Construction', 'Production'].includes(stage)) {
      npv = randomFloat(50, 5000, 0) // $50M to $5B
      irr = randomFloat(8, 35, 1) // 8% to 35%
      capex = randomFloat(100, 3000, 0) // $100M to $3B
    }

    // Resource and reserve estimates
    let resource = null
    let reserve = null

    if (['Pre-Feasibility', 'Feasibility', 'Development', 'Construction', 'Production'].includes(stage)) {
      const resourceTonnes = randomFloat(1, 500, 1)
      const resourceGrade = projectCommodities[0] === 'Iron Ore'
        ? randomFloat(45, 68, 1)
        : randomFloat(0.5, 8, 2)
      resource = `${resourceTonnes}Mt @ ${resourceGrade}${projectCommodities[0] === 'Iron Ore' ? '% Fe' : 'g/t'}`

      if (['Feasibility', 'Development', 'Construction', 'Production'].includes(stage)) {
        const reserveTonnes = randomFloat(0.5, resourceTonnes * 0.7, 1)
        const reserveGrade = projectCommodities[0] === 'Iron Ore'
          ? randomFloat(50, 65, 1)
          : randomFloat(0.8, 6, 2)
        reserve = `${reserveTonnes}Mt @ ${reserveGrade}${projectCommodities[0] === 'Iron Ore' ? '% Fe' : 'g/t'}`
      }
    }

    const project = {
      company_id: company.id,
      name: projectName,
      location,
      stage,
      commodities: projectCommodities,
      status,
      description: `The ${projectName} is a ${stage.toLowerCase()} stage ${projectCommodities.join('-')} project located in ${location}, ${country}. The project is operated by ${company.name}.`,
      urls: [
        `https://www.${company.name.toLowerCase().replace(/\s+/g, '')}.com/projects/${projectName.toLowerCase().replace(/\s+/g, '-')}`,
        `https://investors.${company.name.toLowerCase().replace(/\s+/g, '')}.com/${projectName.toLowerCase().replace(/\s+/g, '-')}`
      ],
      watchlist: Math.random() > 0.85,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      npv,
      irr,
      capex,
      financial_metrics_updated_at: npv ? new Date().toISOString() : null,
      resource,
      reserve,
      user_id: null,
      is_private: false,
      uploaded_at: null,
      document_storage_path: null
    }

    projects.push(project)

    if ((i + 1) % 50 === 0) {
      console.log(`   Generated ${i + 1}/${count} projects...`)
    }
  }

  console.log(`\n📤 Inserting ${projects.length} projects into database...\n`)

  // Insert in batches of 100
  const batchSize = 100
  let insertedCount = 0

  for (let i = 0; i < projects.length; i += batchSize) {
    const batch = projects.slice(i, i + batchSize)

    const { data, error } = await supabase
      .from('projects')
      .insert(batch)
      .select('id')

    if (error) {
      console.error(`❌ Error inserting projects batch ${i / batchSize + 1}:`, error.message)
    } else {
      insertedCount += data?.length || 0
      console.log(`   ✅ Inserted batch ${i / batchSize + 1} (${batch.length} projects)`)
    }
  }

  console.log(`\n✅ Successfully inserted ${insertedCount} projects\n`)
}

// Main execution
async function main() {
  console.log('🚀 Starting synthetic data generation...')
  console.log('=' .repeat(60))

  try {
    // Clear existing data
    console.log('\n🗑️  Clearing existing data...')
    await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')
    await supabase.from('companies').delete().neq('id', '00000000-0000-0000-0000-000000000000')
    console.log('✅ Existing data cleared\n')

    // Generate companies
    const companies = await generateCompanies(300)

    // Generate projects
    await generateProjects(400, companies)

    console.log('=' .repeat(60))
    console.log('🎉 Data generation complete!')
    console.log('\n📊 Summary:')
    console.log(`   Companies: ${companies.length}`)
    console.log(`   Projects: 400`)
    console.log('\n✅ All data successfully populated!')

  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message)
    process.exit(1)
  }
}

main()
