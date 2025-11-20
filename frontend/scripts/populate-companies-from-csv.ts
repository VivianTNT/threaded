import { createClient } from '@supabase/supabase-js'
import fs from 'fs'
import path from 'path'
import { parse } from 'csv-parse/sync'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
)

interface MiningCompany {
  Exchange: string
  Ticker: string
  Name: string
  ISIN: string
  Country: string
  Currency: string
  Sector: string
  Industry: string
  SubIndustry: string
  NAICS: string
  GICS: string
  TRBC: string
  CommodityFocus: string
  Source: string
  SourceDate: string
  Is_Mining: string
  ExchangeTicker: string
  Name_Normal: string
  Dedupe_Key: string
}

// Generate a basic description for mining companies
function generateMiningDescription(company: MiningCompany): string {
  const { Name, Sector, CommodityFocus, Country } = company

  let desc = `${Name} is a`

  if (CommodityFocus) {
    desc += ` ${CommodityFocus}`
  } else if (Sector && Sector.toLowerCase().includes('material')) {
    desc += ` materials and mining`
  } else {
    desc += ` mining`
  }

  desc += ` company`

  if (Country) {
    desc += ` based in ${Country}`
  }

  desc += `.`

  return desc
}

async function main() {
  console.log('🚀 Populating companies table from CSV...\n')

  // Read CSV file
  const csvPath = path.join(process.cwd(), 'Mining Ticker List.csv')
  const csvContent = fs.readFileSync(csvPath, 'utf-8')

  const records: MiningCompany[] = parse(csvContent, {
    columns: true,
    skip_empty_lines: true
  })

  console.log(`📊 Found ${records.length} mining companies in CSV\n`)

  let successCount = 0
  let failureCount = 0
  let skippedCount = 0

  // Batch insert for better performance
  const batchSize = 50
  const batches: any[][] = []

  for (let i = 0; i < records.length; i++) {
    const company = records[i]

    // Skip if no ticker or exchange
    if (!company.Ticker || !company.Exchange) {
      skippedCount++
      continue
    }

    const companyData = {
      name: company.Name,
      ticker: company.Ticker,
      exchange: company.Exchange,
      country: company.Country || null,
      description: generateMiningDescription(company),
      market_cap: null, // Will be populated later via API
      website: null, // Will be populated later via API
      updated_at: new Date().toISOString()
    }

    if (batches.length === 0 || batches[batches.length - 1].length >= batchSize) {
      batches.push([])
    }
    batches[batches.length - 1].push(companyData)
  }

  console.log(`Processing ${batches.length} batches of up to ${batchSize} companies each...\n`)

  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i]
    console.log(`[Batch ${i + 1}/${batches.length}] Inserting ${batch.length} companies...`)

    const { data, error } = await supabase
      .from('companies')
      .upsert(batch, {
        onConflict: 'ticker,exchange',
        ignoreDuplicates: false
      })

    if (error) {
      console.error(`  ❌ Batch failed:`, error.message)
      failureCount += batch.length
    } else {
      console.log(`  ✅ Batch successful`)
      successCount += batch.length
    }

    // Progress update
    if ((i + 1) % 10 === 0 || i === batches.length - 1) {
      console.log(`\n📈 Progress: ${i + 1}/${batches.length} batches`)
      console.log(`   ✅ Success: ${successCount}`)
      console.log(`   ❌ Failed: ${failureCount}`)
      console.log(`   ⏭️  Skipped: ${skippedCount}\n`)
    }
  }

  console.log('\n🎉 Company population complete!')
  console.log(`\n📊 Final Statistics:`)
  console.log(`   Total companies in CSV: ${records.length}`)
  console.log(`   ✅ Successfully inserted/updated: ${successCount}`)
  console.log(`   ❌ Failed: ${failureCount}`)
  console.log(`   ⏭️  Skipped (missing data): ${skippedCount}`)
}

main().catch(console.error)
