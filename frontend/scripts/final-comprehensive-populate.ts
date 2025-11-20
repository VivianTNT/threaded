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
  console.log('🚀 Starting comprehensive database population...\n')
  console.log('⚠️  This will CLEAR all existing data first!\n')

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

    // Step 3: Generate companies (273 - uneven number)
    console.log('🏢 Generating 273 companies...')

    const exchanges = ['NYSE', 'TSX', 'ASX', 'LSE', 'JSE', 'HKEX', 'TSX-V', 'AIM']
    const countries = ['Canada', 'Australia', 'USA', 'UK', 'South Africa', 'Peru', 'Chile', 'Brazil', 'Mexico', 'China', 'Russia', 'Kazakhstan', 'Mongolia', 'DRC', 'Zambia', 'Tanzania', 'Ghana', 'Saudi Arabia', 'Guinea', 'Mali', 'Burkina Faso', 'Indonesia', 'Philippines', 'Argentina', 'Finland', 'Sweden', 'Poland', 'Spain']

    const companyPrefixes = ['Global', 'Northern', 'Southern', 'Eastern', 'Western', 'Central', 'Pacific', 'Atlantic', 'Mountain', 'Valley', 'Peak', 'Summit', 'Frontier', 'Pioneer', 'Premier', 'Imperial', 'Royal', 'Continental', 'International', 'United']
    const companySuffixes = ['Mining', 'Resources', 'Minerals', 'Gold', 'Copper', 'Metals', 'Group', 'Corporation', 'Holdings', 'Industries', 'Ventures']

    const companies = []
    for (let i = 0; i < 273; i++) {
      const prefix = randomElement(companyPrefixes)
      const suffix = randomElement(companySuffixes)
      const name = `${prefix} ${suffix}`
      const ticker = name.split(' ').map(w => w[0]).join('') + randomInt(10, 99)

      companies.push({
        name,
        ticker,
        exchange: randomElement(exchanges),
        country: randomElement(countries),
        website: `https://www.${name.toLowerCase().replace(/\s+/g, '')}.com`,
        description: `${name} is a ${randomElement(['diversified', 'focused', 'leading', 'emerging', 'established'])} mining company specializing in ${randomElements(['precious metals', 'base metals', 'critical minerals', 'rare earth elements', 'industrial minerals'], 1, 3).join(', ')}.`,
        market_cap: randomFloat(50, 50000, 0),
        urls: [`https://www.${name.toLowerCase().replace(/\s+/g, '')}.com`],
        watchlist: false
      })
    }

    // Insert companies in batches
    for (let i = 0; i < companies.length; i += 100) {
      const batch = companies.slice(i, i + 100)
      const { error } = await supabase.from('companies').insert(batch)
      if (error) console.error(`Error inserting companies batch ${i / 100 + 1}:`, error.message)
      else console.log(`   ✅ Inserted companies ${i + 1}-${Math.min(i + 100, companies.length)}`)
    }

    // Fetch inserted companies
    const { data: insertedCompanies, error: fetchError } = await supabase
      .from('companies')
      .select('id, name')

    if (fetchError || !insertedCompanies) throw new Error('Failed to fetch companies')
    console.log(`✅ ${insertedCompanies.length} companies created\n`)

    // Step 4: Generate projects (417 - uneven number)
    console.log('⛏️  Generating 417 projects...')

    const projectPrefixes = ['Mount', 'Lake', 'River', 'Valley', 'Peak', 'Mesa', 'Canyon', 'Ridge', 'Summit', 'Highland', 'Lowland', 'Eagle', 'Dragon', 'Phoenix', 'Thunder', 'Lightning', 'Star', 'Crown', 'Emperor', 'King', 'Queen', 'Aurora', 'Horizon', 'Frontier', 'Pioneer']
    const projectSuffixes = ['Mine', 'Project', 'Deposit', 'Complex', 'Operation', 'District', 'Belt']

    const locations = ['Pilbara Region', 'Kalgoorlie', 'Golden Triangle', 'Red Lake', 'Timmins', 'Sudbury Basin', 'Nevada', 'Arizona', 'Alaska', 'Yukon', 'British Columbia', 'Saskatchewan', 'Ontario', 'Quebec', 'Atacama Desert', 'Andes Mountains', 'Amazon Basin', 'Copperbelt', 'Great Dyke', 'Bushveld Complex', 'Witwatersrand Basin', 'Yilgarn Craton', 'Hunter Valley', 'Bowen Basin', 'Sichuan Province', 'Inner Mongolia', 'Siberia', 'Urals', 'Caucasus', 'Northern Territory']
    const countries2 = ['Australia', 'Canada', 'USA', 'Chile', 'Peru', 'Brazil', 'South Africa', 'DRC', 'Zambia', 'Zimbabwe', 'China', 'Russia', 'Kazakhstan', 'Mongolia', 'Indonesia', 'Philippines', 'Mexico', 'Argentina', 'Finland', 'Sweden', 'Poland', 'Spain', 'Turkey', 'Saudi Arabia', 'Guinea', 'Mali', 'Burkina Faso', 'Ghana', 'Tanzania', 'Namibia']

    const stages = ['Exploration', 'Pre-Feasibility', 'Feasibility', 'Development', 'Construction', 'Production']
    const commodities = ['Gold', 'Copper', 'Lithium', 'Nickel', 'Cobalt', 'Zinc', 'Lead', 'Silver', 'Platinum', 'Palladium', 'Rare Earth Elements', 'Iron Ore', 'Molybdenum', 'Tin', 'Tungsten', 'Graphite', 'Uranium']
    const statuses = ['Active', 'Under Development', 'Operational', 'Feasibility Stage', 'Exploration Stage']

    const projects = []
    for (let i = 0; i < 417; i++) {
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
          npv = randomFloat(20, 200, 0)
          irr = randomFloat(10, 18, 1)
          capex = randomFloat(50, 300, 0)
          break
        case 'Pre-Feasibility':
          npv = randomFloat(150, 800, 0)
          irr = randomFloat(12, 22, 1)
          capex = randomFloat(250, 800, 0)
          break
        case 'Feasibility':
          npv = randomFloat(600, 2500, 0)
          irr = randomFloat(15, 28, 1)
          capex = randomFloat(700, 1800, 0)
          break
        case 'Development':
          npv = randomFloat(1500, 4000, 0)
          irr = randomFloat(17, 30, 1)
          capex = randomFloat(1200, 2500, 0)
          break
        case 'Construction':
          npv = randomFloat(2500, 5000, 0)
          irr = randomFloat(20, 35, 1)
          capex = randomFloat(2000, 3500, 0)
          break
        case 'Production':
          npv = randomFloat(2000, 6000, 0)
          irr = randomFloat(22, 40, 1)
          capex = randomFloat(2000, 4000, 0)
          break
        default:
          npv = 100
          irr = 15
          capex = 200
      }

      // Resource and reserve estimates
      const resourceTonnes = randomFloat(5, 500, 1)
      let resourceGrade: number
      let unit: string

      if (primaryCommodity === 'Iron Ore') {
        resourceGrade = randomFloat(52, 68, 1)
        unit = '% Fe'
      } else if (['Copper', 'Nickel', 'Zinc', 'Lead'].includes(primaryCommodity)) {
        resourceGrade = randomFloat(0.8, 4.5, 2)
        unit = '% Cu'
      } else {
        resourceGrade = randomFloat(1.2, 8.5, 2)
        unit = 'g/t'
      }

      const resource = `${resourceTonnes}Mt @ ${resourceGrade}${unit}`
      const reserveTonnes = randomFloat(resourceTonnes * 0.5, resourceTonnes * 0.8, 1)
      const reserveGrade = resourceGrade * randomFloat(1.05, 1.25, 2)
      const reserve = `${reserveTonnes}Mt @ ${reserveGrade}${unit}`

      // Assign PDF (rotate through available files)
      const pdfFile = pdfFiles[i % pdfFiles.length]
      const pdfUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/mining-documents/${pdfFile.name}`

      projects.push({
        company_id: company.id,
        name: projectName,
        location: `${location}, ${country}`,
        stage,
        commodities: projectCommodities,
        status: randomElement(statuses),
        description: `The ${projectName} is a ${stage.toLowerCase()} ${projectCommodities.join('/')} project located in ${location}, ${country}.`,
        urls: [pdfUrl],
        watchlist: false,
        npv,
        irr,
        capex,
        resource,
        reserve,
        document_storage_path: pdfUrl
      })
    }

    // Insert projects in batches
    for (let i = 0; i < projects.length; i += 100) {
      const batch = projects.slice(i, i + 100)
      const { error } = await supabase.from('projects').insert(batch)
      if (error) console.error(`Error inserting projects batch ${i / 100 + 1}:`, error.message)
      else console.log(`   ✅ Inserted projects ${i + 1}-${Math.min(i + 100, projects.length)}`)
    }

    console.log(`✅ ${projects.length} projects created\n`)

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

    console.log('\n' + '='.repeat(60))
    console.log('✅ Database populated successfully!')
    console.log(`   Companies: ${companyCount}`)
    console.log(`   Projects: ${projectCount}`)
    console.log(`   Projects with PDFs: ${projectsWithPdfs}`)
    console.log('='.repeat(60))

  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message)
    process.exit(1)
  }
}

main()
