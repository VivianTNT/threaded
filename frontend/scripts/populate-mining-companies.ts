#!/usr/bin/env node

/**
 * Populate database with comprehensive list of mining companies
 * Includes 120+ critical minerals companies from ASX, TSX, NYSE, LSE, etc.
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)

async function main() {
  console.log('='.repeat(70))
  console.log('POPULATING MINING COMPANIES DATABASE')
  console.log('='.repeat(70))

  // Load companies data
  const dataPath = path.join(__dirname, '..', 'data', 'mining-companies-comprehensive.json')
  console.log(`\n📂 Loading data from: ${dataPath}`)

  if (!fs.existsSync(dataPath)) {
    console.error(`❌ Error: File not found: ${dataPath}`)
    process.exit(1)
  }

  const companiesData = JSON.parse(fs.readFileSync(dataPath, 'utf-8'))
  console.log(`✅ Loaded ${companiesData.length} companies`)

  // Check if table exists and has data
  const { count: existingCount } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true })

  console.log(`\n📊 Current database status:`)
  console.log(`   Existing companies: ${existingCount}`)

  // Ask user if they want to clear existing data (in production, add confirmation)
  if (existingCount && existingCount > 0) {
    console.log(`\n⚠️  Warning: Database already contains ${existingCount} companies`)
    console.log(`   This script will add new companies (duplicates will be skipped)`)
  }

  // Statistics
  let inserted = 0
  let skipped = 0
  let errors = 0

  console.log('\n📥 Inserting companies...')
  console.log('=' .repeat(70))

  for (const company of companiesData) {
    try {
      // Check if company already exists (by ticker and exchange)
      const { data: existing } = await supabase
        .from('companies')
        .select('id')
        .eq('ticker', company.ticker)
        .eq('exchange', company.exchange)
        .single()

      if (existing) {
        console.log(`   ⏭️  Skipped: ${company.name} (${company.ticker}.${company.exchange}) - already exists`)
        skipped++
        continue
      }

      // Insert company
      const { error } = await supabase.from('companies').insert({
        name: company.name,
        ticker: company.ticker,
        exchange: company.exchange,
        country: company.country,
        website: company.website,
        description: company.description,
        market_cap: company.market_cap,
        urls: company.urls,
        watchlist: false
      })

      if (error) {
        console.log(`   ❌ Error inserting ${company.name}: ${error.message}`)
        errors++
      } else {
        console.log(`   ✅ Inserted: ${company.name} (${company.ticker}.${company.exchange})`)
        inserted++
      }

    } catch (error: any) {
      console.log(`   ❌ Error processing ${company.name}: ${error.message}`)
      errors++
    }
  }

  // Get final statistics
  const { count: finalCount, data: allCompanies } = await supabase
    .from('companies')
    .select('*', { count: 'exact' })
    .order('market_cap', { ascending: false })

  console.log('\n' + '='.repeat(70))
  console.log('POPULATION COMPLETE')
  console.log('='.repeat(70))
  console.log(`\n📊 Summary:`)
  console.log(`   Companies inserted: ${inserted}`)
  console.log(`   Companies skipped: ${skipped}`)
  console.log(`   Errors: ${errors}`)
  console.log(`   Total in database: ${finalCount}`)

  // Breakdown by exchange
  if (allCompanies && allCompanies.length > 0) {
    const exchangeCounts: Record<string, number> = {}
    allCompanies.forEach(c => {
      exchangeCounts[c.exchange] = (exchangeCounts[c.exchange] || 0) + 1
    })

    console.log(`\n📈 Companies by Exchange:`)
    Object.entries(exchangeCounts)
      .sort((a, b) => b[1] - a[1])
      .forEach(([exchange, count]) => {
        console.log(`   ${exchange}: ${count}`)
      })

    // Top 10 by market cap
    console.log(`\n💰 Top 10 Companies by Market Cap:`)
    allCompanies
      .slice(0, 10)
      .forEach((c, i) => {
        const marketCapB = c.market_cap / 1e9
        console.log(`   ${i + 1}. ${c.name} (${c.ticker}.${c.exchange}): $${marketCapB.toFixed(1)}B`)
      })

    // Commodity breakdown
    const commodityCounts: Record<string, number> = {
      lithium: 0,
      copper: 0,
      nickel: 0,
      cobalt: 0,
      'rare earths': 0,
      graphite: 0,
      gold: 0,
      uranium: 0
    }

    allCompanies.forEach(c => {
      const desc = c.description?.toLowerCase() || ''
      if (desc.includes('lithium')) commodityCounts.lithium++
      if (desc.includes('copper')) commodityCounts.copper++
      if (desc.includes('nickel')) commodityCounts.nickel++
      if (desc.includes('cobalt')) commodityCounts.cobalt++
      if (desc.includes('rare earth')) commodityCounts['rare earths']++
      if (desc.includes('graphite')) commodityCounts.graphite++
      if (desc.includes('gold')) commodityCounts.gold++
      if (desc.includes('uranium')) commodityCounts.uranium++
    })

    console.log(`\n🔬 Companies by Primary Commodity:`)
    Object.entries(commodityCounts)
      .filter(([_, count]) => count > 0)
      .sort((a, b) => b[1] - a[1])
      .forEach(([commodity, count]) => {
        console.log(`   ${commodity}: ${count}`)
      })
  }

  console.log('\n✅ Database populated successfully!')
  console.log('=' .repeat(70))
}

main().catch(error => {
  console.error('Fatal error:', error)
  process.exit(1)
})
