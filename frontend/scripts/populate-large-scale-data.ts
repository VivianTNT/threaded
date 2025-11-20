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
  const value = Math.random() * (max - min) + min
  return Number(value.toFixed(decimals))
}

function randomElement<T>(arr: T[]): T {
  return arr[randomInt(0, arr.length - 1)]
}

function randomElements<T>(arr: T[], min: number = 1, max: number = 3): T[] {
  const count = randomInt(min, max)
  const shuffled = [...arr].sort(() => Math.random() - 0.5)
  return shuffled.slice(0, count)
}

async function main() {
  console.log('🚀 Starting LARGE-SCALE database population...\n')
  console.log('⚠️  This will CLEAR all existing data first!\n')
  console.log('📊 Target: ~2,000 companies and ~14,000 projects (1:7 ratio)\n')

  try {
    // Step 1: Clear existing data
    console.log('🗑️  Clearing existing data...')
    await supabase.from('pdf_highlights').delete().neq('id', '00000000-0000-0000-0000-000000000000')
    await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')
    await supabase.from('companies').delete().neq('id', '00000000-0000-0000-0000-000000000000')
    console.log('✅ Data cleared\n')

    // Step 2: Fetch available PDF files
    console.log('📁 Fetching available PDF files...')
    const { data: pdfFiles, error: pdfError } = await supabase.storage
      .from('mining-documents')
      .list('', { limit: 100 })

    if (pdfError || !pdfFiles || pdfFiles.length === 0) {
      throw new Error('No PDF files found in mining-documents bucket')
    }

    console.log(`✅ Found ${pdfFiles.length} PDF files\n`)

    // Step 3: Generate companies (2,137 - uneven number)
    const numCompanies = 2137
    console.log(`🏢 Generating ${numCompanies.toLocaleString()} companies...\n`)

    const exchanges = ['NYSE', 'NASDAQ', 'TSX', 'TSX-V', 'ASX', 'LSE', 'AIM', 'JSE', 'HKEX', 'SGX', 'BSE', 'NSE']
    const countries = ['Canada', 'Australia', 'USA', 'UK', 'South Africa', 'Peru', 'Chile', 'Brazil', 'Mexico', 'Argentina', 'China', 'Russia', 'Kazakhstan', 'Mongolia', 'DRC', 'Zambia', 'Tanzania', 'Ghana', 'Saudi Arabia', 'Guinea', 'Mali', 'Burkina Faso', 'Indonesia', 'Philippines', 'Finland', 'Sweden', 'Poland', 'Spain', 'Turkey', 'Namibia', 'Botswana', 'Zimbabwe']

    const companyPrefixes = ['Global', 'Northern', 'Southern', 'Eastern', 'Western', 'Central', 'Pacific', 'Atlantic', 'Mountain', 'Valley', 'Peak', 'Summit', 'Frontier', 'Pioneer', 'Premier', 'Imperial', 'Royal', 'Continental', 'International', 'United', 'Advanced', 'Superior', 'Elite', 'Prime', 'Noble', 'Apex', 'Titan', 'Omega', 'Alpha', 'Delta', 'Sigma', 'Phoenix', 'Dragon', 'Eagle', 'Lion', 'Tiger', 'Wolf', 'Bear', 'Falcon', 'Hawk', 'Thunder', 'Lightning', 'Storm', 'Blaze', 'Inferno', 'Aurora', 'Horizon', 'Zenith', 'Pinnacle', 'Crest']
    const companySuffixes = ['Mining', 'Resources', 'Minerals', 'Gold', 'Copper', 'Metals', 'Group', 'Corporation', 'Holdings', 'Industries', 'Ventures', 'Exploration', 'Development', 'Commodities', 'Materials', 'Assets', 'Capital', 'Investments', 'Partners', 'Operations']

    let companiesCreated = 0
    let batch = []
    const batchSize = 100

    for (let i = 0; i < numCompanies; i++) {
      const prefix = randomElement(companyPrefixes)
      const suffix = randomElement(companySuffixes)
      const name = `${prefix} ${suffix}`
      const ticker = name.split(' ').map(w => w[0]).join('').toUpperCase() + randomInt(10, 999)

      batch.push({
        name: `${name} ${i > 500 ? 'Corp' : ''}`.trim(),
        ticker,
        exchange: randomElement(exchanges),
        country: randomElement(countries),
        website: `https://www.${name.toLowerCase().replace(/\s+/g, '')}-${i}.com`,
        description: `${name} is a ${randomElement(['diversified', 'focused', 'leading', 'emerging', 'established', 'junior', 'mid-tier', 'major'])} mining company specializing in ${randomElements(['precious metals', 'base metals', 'critical minerals', 'rare earth elements', 'industrial minerals', 'energy minerals'], 1, 3).join(', ')}.`,
        market_cap: randomFloat(10, 75000, 0),
        urls: [`https://www.${name.toLowerCase().replace(/\s+/g, '')}-${i}.com`],
        watchlist: Math.random() < 0.05
      })

      if (batch.length === batchSize || i === numCompanies - 1) {
        const { error } = await supabase.from('companies').insert(batch)
        if (!error) {
          companiesCreated += batch.length
          console.log(`   ✅ Inserted companies ${companiesCreated - batch.length + 1}-${companiesCreated} / ${numCompanies}`)
        } else {
          console.error(`   ⚠️ Error inserting batch: ${error.message}`)
        }
        batch = []
      }
    }

    // Fetch inserted companies
    const { data: insertedCompanies, error: fetchError } = await supabase
      .from('companies')
      .select('id, name')

    if (fetchError || !insertedCompanies) throw new Error('Failed to fetch companies')
    console.log(`\n✅ ${insertedCompanies.length} companies created\n`)

    // Step 4: Generate projects (14,783 - uneven number, ~7x companies)
    const numProjects = 14783
    console.log(`⛏️  Generating ${numProjects.toLocaleString()} projects...\n`)

    const projectPrefixes = ['Mount', 'Lake', 'River', 'Valley', 'Peak', 'Mesa', 'Canyon', 'Ridge', 'Summit', 'Highland', 'Lowland', 'Eagle', 'Dragon', 'Phoenix', 'Thunder', 'Lightning', 'Star', 'Crown', 'Emperor', 'King', 'Queen', 'Aurora', 'Horizon', 'Frontier', 'Pioneer', 'Vista', 'Sierra', 'Plateau', 'Basin', 'Creek', 'Spring', 'Wells', 'Butte', 'Bluff', 'Crest', 'Heights', 'Forge', 'Boulder', 'Crystal', 'Diamond', 'Ruby', 'Emerald', 'Sapphire', 'Opal', 'Jade', 'Pearl', 'Golden', 'Silver', 'Copper', 'Iron']
    const projectSuffixes = ['Mine', 'Project', 'Deposit', 'Complex', 'Operation', 'District', 'Belt', 'Zone', 'Property', 'Prospect', 'Field', 'Reserve', 'Formation']

    const locations = ['Pilbara Region', 'Kalgoorlie', 'Golden Triangle', 'Red Lake', 'Timmins', 'Sudbury Basin', 'Nevada', 'Arizona', 'Alaska', 'Yukon', 'British Columbia', 'Saskatchewan', 'Ontario', 'Quebec', 'Newfoundland', 'Atacama Desert', 'Andes Mountains', 'Amazon Basin', 'Copperbelt', 'Great Dyke', 'Bushveld Complex', 'Witwatersrand Basin', 'Yilgarn Craton', 'Hunter Valley', 'Bowen Basin', 'Sichuan Province', 'Inner Mongolia', 'Siberia', 'Urals', 'Caucasus', 'Northern Territory', 'Western Australia', 'Queensland', 'New South Wales']
    const countries2 = ['Australia', 'Canada', 'USA', 'Chile', 'Peru', 'Brazil', 'Argentina', 'South Africa', 'DRC', 'Zambia', 'Zimbabwe', 'Botswana', 'Namibia', 'Tanzania', 'Ghana', 'Mali', 'Guinea', 'Burkina Faso', 'China', 'Russia', 'Kazakhstan', 'Mongolia', 'Indonesia', 'Philippines', 'Mexico', 'Saudi Arabia', 'Turkey', 'Finland', 'Sweden', 'Poland', 'Spain']

    const stages = ['Exploration', 'Pre-Feasibility', 'Feasibility', 'Development', 'Construction', 'Production']
    const commodities = ['Gold', 'Copper', 'Lithium', 'Nickel', 'Cobalt', 'Zinc', 'Lead', 'Silver', 'Platinum', 'Palladium', 'Rare Earth Elements', 'Iron Ore', 'Molybdenum', 'Tin', 'Tungsten', 'Graphite', 'Uranium', 'Vanadium', 'Manganese', 'Chromium', 'Titanium']
    const statuses = ['Active', 'Under Development', 'Operational', 'Feasibility Stage', 'Exploration Stage', 'Permitting', 'On Hold', 'Advancing']

    let projectsCreated = 0
    let projectBatch = []

    for (let i = 0; i < numProjects; i++) {
      const company = randomElement(insertedCompanies)
      const projectName = `${randomElement(projectPrefixes)} ${randomElement(projectSuffixes)}`
      const location = randomElement(locations)
      const country = randomElement(countries2)
      const stage = randomElement(stages)
      const projectCommodities = randomElements(commodities, 1, 3)
      const primaryCommodity = projectCommodities[0]

      // Financial metrics scaled by stage
      let npv: number, irr: number, capex: number
      switch (stage) {
        case 'Exploration':
          npv = randomFloat(15, 250, 0)
          irr = randomFloat(9, 19, 1)
          capex = randomFloat(40, 350, 0)
          break
        case 'Pre-Feasibility':
          npv = randomFloat(120, 900, 0)
          irr = randomFloat(11, 23, 1)
          capex = randomFloat(200, 850, 0)
          break
        case 'Feasibility':
          npv = randomFloat(500, 2800, 0)
          irr = randomFloat(14, 29, 1)
          capex = randomFloat(600, 2000, 0)
          break
        case 'Development':
          npv = randomFloat(1300, 4500, 0)
          irr = randomFloat(16, 32, 1)
          capex = randomFloat(1000, 2800, 0)
          break
        case 'Construction':
          npv = randomFloat(2200, 5500, 0)
          irr = randomFloat(19, 37, 1)
          capex = randomFloat(1800, 3800, 0)
          break
        case 'Production':
          npv = randomFloat(1800, 7000, 0)
          irr = randomFloat(21, 42, 1)
          capex = randomFloat(1500, 4500, 0)
          break
        default:
          npv = 100
          irr = 15
          capex = 200
      }

      // Resource and reserve estimates
      const resourceTonnes = randomFloat(3, 600, 1)
      let resourceGrade: number
      let unit: string

      if (primaryCommodity === 'Iron Ore') {
        resourceGrade = randomFloat(48, 70, 1)
        unit = '% Fe'
      } else if (['Copper', 'Nickel', 'Zinc', 'Lead'].includes(primaryCommodity)) {
        resourceGrade = randomFloat(0.6, 5.2, 2)
        unit = '% Cu'
      } else if (primaryCommodity === 'Lithium') {
        resourceGrade = randomFloat(0.8, 2.5, 2)
        unit = '% Li2O'
      } else {
        resourceGrade = randomFloat(0.9, 12.5, 2)
        unit = 'g/t'
      }

      const resource = `${resourceTonnes}Mt @ ${resourceGrade}${unit}`
      const reserveTonnes = randomFloat(resourceTonnes * 0.4, resourceTonnes * 0.85, 1)
      const reserveGrade = resourceGrade * randomFloat(1.03, 1.28, 2)
      const reserve = `${reserveTonnes}Mt @ ${reserveGrade}${unit}`

      // Assign PDF (rotate through available files)
      const pdfFile = pdfFiles[i % pdfFiles.length]
      const pdfUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/mining-documents/${pdfFile.name}`

      projectBatch.push({
        company_id: company.id,
        name: projectName,
        location: `${location}, ${country}`,
        stage,
        commodities: projectCommodities,
        status: randomElement(statuses),
        description: `The ${projectName} is a ${stage.toLowerCase()} ${projectCommodities.join('/')} project located in ${location}, ${country}. ${stage === 'Production' ? 'Currently in operation.' : stage === 'Exploration' ? 'Early-stage exploration activities ongoing.' : 'Advancing towards development.'}`,
        urls: [pdfUrl],
        watchlist: Math.random() < 0.03,
        npv,
        irr,
        capex,
        resource,
        reserve,
        document_storage_path: pdfUrl
      })

      if (projectBatch.length === batchSize || i === numProjects - 1) {
        const { error } = await supabase.from('projects').insert(projectBatch)
        if (!error) {
          projectsCreated += projectBatch.length
          if (projectsCreated % 500 === 0 || i === numProjects - 1) {
            console.log(`   ✅ Inserted projects ${projectsCreated - projectBatch.length + 1}-${projectsCreated} / ${numProjects}`)
          }
        } else {
          console.error(`   ⚠️ Error inserting batch: ${error.message}`)
        }
        projectBatch = []
      }
    }

    console.log(`\n✅ ${projectsCreated} projects created\n`)

    // Final verification
    console.log('🔍 Verifying final state...')
    const { count: companyCount } = await supabase
      .from('companies')
      .select('*', { count: 'exact', head: true })

    const { count: projectCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true })

    const { count: projectsWithPdfs } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true })
      .not('document_storage_path', 'is', null)

    const ratio = projectCount && companyCount ? (projectCount / companyCount).toFixed(2) : '0'

    console.log('\n' + '='.repeat(60))
    console.log('✅ LARGE-SCALE DATABASE POPULATED SUCCESSFULLY!')
    console.log(`   Companies: ${companyCount?.toLocaleString()}`)
    console.log(`   Projects: ${projectCount?.toLocaleString()}`)
    console.log(`   Projects with PDFs: ${projectsWithPdfs?.toLocaleString()}`)
    console.log(`   Project:Company Ratio: ${ratio}:1`)
    console.log('='.repeat(60))

  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message)
    process.exit(1)
  }
}

main()
