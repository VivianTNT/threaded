#!/usr/bin/env node

/**
 * Populate database with comprehensive list of mining projects
 * Includes 120+ real critical minerals projects from major mining companies
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)

interface ProjectData {
  company_name: string
  name: string
  location: string
  stage: string
  commodities: string[]
  resource_estimate: string | null
  reserve_estimate: string | null
  ownership_percentage: number
  status: string
  description: string
  urls: string[]
}

async function main() {
  console.log('='.repeat(70))
  console.log('POPULATING MINING PROJECTS DATABASE')
  console.log('='.repeat(70))

  // Load projects data
  const dataPath = path.join(__dirname, '..', 'data', 'mining-projects-comprehensive.json')
  console.log(`\n📂 Loading data from: ${dataPath}`)

  if (!fs.existsSync(dataPath)) {
    console.error(`❌ Error: File not found: ${dataPath}`)
    process.exit(1)
  }

  const projectsData: ProjectData[] = JSON.parse(fs.readFileSync(dataPath, 'utf-8'))
  console.log(`✅ Loaded ${projectsData.length} projects`)

  // Check if table exists and has data
  const { count: existingCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  console.log(`\n📊 Current database status:`)
  console.log(`   Existing projects: ${existingCount}`)

  if (existingCount && existingCount > 0) {
    console.log(`\n⚠️  Warning: Database already contains ${existingCount} projects`)
    console.log(`   This script will add new projects (duplicates will be skipped)`)
  }

  // Get all companies for matching
  const { data: companies, error: companiesError } = await supabase
    .from('companies')
    .select('id, name, ticker, exchange')

  if (companiesError || !companies || companies.length === 0) {
    console.error('❌ Error: Could not load companies from database')
    console.error('   Please populate companies first using populate-mining-companies.ts')
    process.exit(1)
  }

  console.log(`✅ Loaded ${companies.length} companies for matching`)

  // Create a map for faster company lookups
  const companyMap = new Map<string, string>()
  companies.forEach(c => {
    companyMap.set(c.name.toLowerCase(), c.id)
    // Also add variations without common suffixes
    const cleanName = c.name
      .toLowerCase()
      .replace(/\s+(limited|ltd|corporation|corp|inc|plc|s\.a\.|pty)\.?$/i, '')
      .trim()
    companyMap.set(cleanName, c.id)
  })

  // Statistics
  let inserted = 0
  let skipped = 0
  let errors = 0
  let companyNotFound = 0

  console.log('\n📥 Inserting projects...')
  console.log('='.repeat(70))

  for (const project of projectsData) {
    try {
      // Find company_id by matching company_name
      const companyNameLower = project.company_name.toLowerCase()
      const cleanCompanyName = companyNameLower
        .replace(/\s+(limited|ltd|corporation|corp|inc|plc|s\.a\.|pty)\.?$/i, '')
        .trim()

      let companyId = companyMap.get(companyNameLower) || companyMap.get(cleanCompanyName)

      if (!companyId) {
        console.log(`   ⚠️  Company not found: ${project.company_name} (skipping ${project.name})`)
        companyNotFound++
        continue
      }

      // Check if project already exists (by name and company_id)
      const { data: existing } = await supabase
        .from('projects')
        .select('id')
        .eq('name', project.name)
        .eq('company_id', companyId)
        .single()

      if (existing) {
        console.log(`   ⏭️  Skipped: ${project.name} (${project.company_name}) - already exists`)
        skipped++
        continue
      }

      // Insert project
      const { error } = await supabase.from('projects').insert({
        company_id: companyId,
        name: project.name,
        location: project.location,
        stage: project.stage,
        commodities: project.commodities,
        resource_estimate: project.resource_estimate,
        reserve_estimate: project.reserve_estimate,
        ownership_percentage: project.ownership_percentage,
        status: project.status,
        description: project.description,
        urls: project.urls,
        watchlist: false
      })

      if (error) {
        console.log(`   ❌ Error inserting ${project.name}: ${error.message}`)
        errors++
      } else {
        const commodityStr = project.commodities.join(', ')
        console.log(`   ✅ Inserted: ${project.name} (${commodityStr}) - ${project.stage}`)
        inserted++
      }

    } catch (error: any) {
      console.log(`   ❌ Error processing ${project.name}: ${error.message}`)
      errors++
    }
  }

  // Get final statistics
  const { count: finalCount, data: allProjects } = await supabase
    .from('projects')
    .select('*, companies(name, ticker)', { count: 'exact' })

  console.log('\n' + '='.repeat(70))
  console.log('POPULATION COMPLETE')
  console.log('='.repeat(70))
  console.log(`\n📊 Summary:`)
  console.log(`   Projects inserted: ${inserted}`)
  console.log(`   Projects skipped: ${skipped}`)
  console.log(`   Companies not found: ${companyNotFound}`)
  console.log(`   Errors: ${errors}`)
  console.log(`   Total in database: ${finalCount}`)

  // Breakdown by commodity
  if (allProjects && allProjects.length > 0) {
    const commodityCounts: Record<string, number> = {}
    const stageCounts: Record<string, number> = {}
    const countryCounts: Record<string, number> = {}

    allProjects.forEach(p => {
      // Count commodities
      p.commodities?.forEach((c: string) => {
        commodityCounts[c] = (commodityCounts[c] || 0) + 1
      })

      // Count stages
      if (p.stage) {
        stageCounts[p.stage] = (stageCounts[p.stage] || 0) + 1
      }

      // Count countries
      if (p.location) {
        // Extract country from location (assumes format like "State, Country" or "Country")
        const parts = p.location.split(',')
        const country = parts.length > 1 ? parts[parts.length - 1].trim() : p.location.trim()
        countryCounts[country] = (countryCounts[country] || 0) + 1
      }
    })

    console.log(`\n🔬 Projects by Commodity:`)
    Object.entries(commodityCounts)
      .sort((a, b) => b[1] - a[1])
      .forEach(([commodity, count]) => {
        console.log(`   ${commodity}: ${count}`)
      })

    console.log(`\n📈 Projects by Stage:`)
    Object.entries(stageCounts)
      .sort((a, b) => b[1] - a[1])
      .forEach(([stage, count]) => {
        console.log(`   ${stage}: ${count}`)
      })

    console.log(`\n🌍 Top 10 Countries by Project Count:`)
    Object.entries(countryCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .forEach(([country, count]) => {
        console.log(`   ${country}: ${count}`)
      })
  }

  console.log('\n✅ Database populated successfully!')
  console.log('='.repeat(70))
}

main().catch(error => {
  console.error('Fatal error:', error)
  process.exit(1)
})
