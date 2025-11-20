#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('🔍 Verifying synthetic data population...\n')

  // Count companies
  const { count: companyCount, error: companyError } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true })

  if (companyError) {
    console.error('❌ Error counting companies:', companyError.message)
  } else {
    console.log(`✅ Companies: ${companyCount}`)
  }

  // Count projects
  const { count: projectCount, error: projectError } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  if (projectError) {
    console.error('❌ Error counting projects:', projectError.message)
  } else {
    console.log(`✅ Projects: ${projectCount}`)
  }

  // Sample companies
  console.log('\n📋 Sample Companies:')
  const { data: companies, error: companiesError } = await supabase
    .from('companies')
    .select('name, ticker, exchange, country, market_cap')
    .limit(5)

  if (companiesError) {
    console.error('❌ Error fetching companies:', companiesError.message)
  } else {
    companies?.forEach((company, i) => {
      console.log(`   ${i + 1}. ${company.name} (${company.ticker}) - ${company.exchange}`)
      console.log(`      ${company.country} | Market Cap: $${company.market_cap}M`)
    })
  }

  // Sample projects
  console.log('\n📋 Sample Projects:')
  const { data: projects, error: projectsError } = await supabase
    .from('projects')
    .select(`
      name,
      location,
      stage,
      commodities,
      npv,
      irr,
      capex,
      resource,
      reserve,
      companies (name)
    `)
    .limit(5)

  if (projectsError) {
    console.error('❌ Error fetching projects:', projectsError.message)
  } else {
    projects?.forEach((project: any, i: number) => {
      console.log(`   ${i + 1}. ${project.name}`)
      console.log(`      Company: ${project.companies?.name || 'N/A'}`)
      console.log(`      Location: ${project.location}`)
      console.log(`      Stage: ${project.stage}`)
      console.log(`      Commodities: ${project.commodities?.join(', ')}`)
      if (project.npv) console.log(`      NPV: $${project.npv}M | IRR: ${project.irr}% | CAPEX: $${project.capex}M`)
      if (project.resource) console.log(`      Resource: ${project.resource}`)
      if (project.reserve) console.log(`      Reserve: ${project.reserve}`)
      console.log('')
    })
  }

  // Statistics by stage
  console.log('📊 Projects by Stage:')
  const { data: stageStats, error: stageError } = await supabase
    .from('projects')
    .select('stage')

  if (stageError) {
    console.error('❌ Error fetching stage stats:', stageError.message)
  } else {
    const stageCounts = stageStats?.reduce((acc: any, project: any) => {
      acc[project.stage] = (acc[project.stage] || 0) + 1
      return acc
    }, {})

    Object.entries(stageCounts || {}).forEach(([stage, count]) => {
      console.log(`   ${stage}: ${count}`)
    })
  }

  // Statistics by commodity
  console.log('\n📊 Top 10 Commodities:')
  const { data: commodityProjects, error: commodityError } = await supabase
    .from('projects')
    .select('commodities')

  if (commodityError) {
    console.error('❌ Error fetching commodity stats:', commodityError.message)
  } else {
    const commodityCounts: { [key: string]: number } = {}
    commodityProjects?.forEach((project: any) => {
      project.commodities?.forEach((commodity: string) => {
        commodityCounts[commodity] = (commodityCounts[commodity] || 0) + 1
      })
    })

    const topCommodities = Object.entries(commodityCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)

    topCommodities.forEach(([commodity, count]) => {
      console.log(`   ${commodity}: ${count}`)
    })
  }

  console.log('\n✅ Verification complete!')
}

main()
