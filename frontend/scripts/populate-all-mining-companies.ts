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

interface CompanyData {
  name: string
  ticker: string
  exchange: string
  country: string
  currency?: string
  sector?: string
  market_cap?: number
  description?: string
  website?: string
}

// FactSet API configuration
const FACTSET_USERNAME = process.env.FACTSET_USERNAME || ''
const FACTSET_API_KEY = process.env.FACTSET_API_KEY || ''
const FACTSET_AUTH = Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Rate limiting
async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Fetch company data from FactSet Entity API
async function fetchFactSetCompanyData(ticker: string, exchange: string): Promise<Partial<CompanyData> | null> {
  try {
    // Construct FactSet identifier (exchange:ticker format)
    const factsetId = `${exchange}:${ticker}`

    // FactSet Entity API endpoint for company profile
    const url = `https://api.factset.com/content/factset-entity/v1/entity-profile?ids=${factsetId}`

    const response = await fetch(url, {
      headers: {
        'Authorization': `Basic ${FACTSET_AUTH}`,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }
    })

    if (!response.ok) {
      console.log(`FactSet API error for ${factsetId}: ${response.status}`)
      return null
    }

    const data = await response.json()

    if (data.data && data.data.length > 0) {
      const profile = data.data[0]
      return {
        description: profile.description || profile.businessDescription,
        website: profile.website,
        market_cap: profile.marketCap
      }
    }

    return null
  } catch (error) {
    console.error(`Error fetching FactSet data for ${ticker}:`, error)
    return null
  }
}

// Fetch market cap from FactSet Fundamentals API
async function fetchMarketCap(ticker: string, exchange: string): Promise<number | null> {
  try {
    const factsetId = `${exchange}:${ticker}`
    const url = `https://api.factset.com/content/factset-fundamentals/v2/fundamentals`

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Basic ${FACTSET_AUTH}`,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        ids: [factsetId],
        metrics: ['MKT_VAL'],
        periodicity: 'LTM'
      })
    })

    if (!response.ok) {
      return null
    }

    const data = await response.json()
    if (data.data && data.data.length > 0 && data.data[0].value) {
      return parseFloat(data.data[0].value)
    }

    return null
  } catch (error) {
    console.error(`Error fetching market cap for ${ticker}:`, error)
    return null
  }
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

// Upsert company to Supabase
async function upsertCompany(companyData: CompanyData) {
  const { error } = await supabase
    .from('companies')
    .upsert({
      name: companyData.name,
      ticker: companyData.ticker,
      exchange: companyData.exchange,
      country: companyData.country,
      market_cap: companyData.market_cap,
      description: companyData.description,
      website: companyData.website,
      updated_at: new Date().toISOString()
    }, {
      onConflict: 'ticker,exchange'
    })

  if (error) {
    console.error(`Error upserting ${companyData.ticker}:`, error)
    return false
  }

  return true
}

async function main() {
  console.log('🚀 Starting comprehensive mining company population...\n')

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

  for (let i = 0; i < records.length; i++) {
    const company = records[i]
    const progress = `[${i + 1}/${records.length}]`

    console.log(`${progress} Processing: ${company.Name} (${company.ExchangeTicker})`)

    // Skip if no ticker or exchange
    if (!company.Ticker || !company.Exchange) {
      console.log(`  ⏭️  Skipped - Missing ticker or exchange\n`)
      skippedCount++
      continue
    }

    try {
      // Prepare base company data
      const companyData: CompanyData = {
        name: company.Name,
        ticker: company.Ticker,
        exchange: company.Exchange,
        country: company.Country,
        currency: company.Currency,
        sector: company.Sector,
        description: generateMiningDescription(company)
      }

      // Try to fetch additional data from FactSet
      if (FACTSET_USERNAME && FACTSET_API_KEY) {
        console.log(`  🔍 Fetching FactSet data...`)

        const factsetData = await fetchFactSetCompanyData(company.Ticker, company.Exchange)
        if (factsetData) {
          if (factsetData.description) companyData.description = factsetData.description
          if (factsetData.website) companyData.website = factsetData.website
          if (factsetData.market_cap) companyData.market_cap = factsetData.market_cap
          console.log(`  ✅ Found FactSet data`)
        } else {
          // Try to get market cap separately
          const marketCap = await fetchMarketCap(company.Ticker, company.Exchange)
          if (marketCap) {
            companyData.market_cap = marketCap
            console.log(`  💰 Found market cap: $${(marketCap / 1000000).toFixed(0)}M`)
          }
        }

        // Rate limiting for FactSet API
        await sleep(200) // 5 requests per second max
      }

      // Upsert to database
      const success = await upsertCompany(companyData)

      if (success) {
        successCount++
        console.log(`  ✅ Successfully saved to database\n`)
      } else {
        failureCount++
        console.log(`  ❌ Failed to save to database\n`)
      }

    } catch (error) {
      console.error(`  ❌ Error processing company:`, error)
      failureCount++
      console.log()
    }

    // Progress update every 50 companies
    if ((i + 1) % 50 === 0) {
      console.log(`\n📈 Progress Update: ${i + 1}/${records.length} companies processed`)
      console.log(`   ✅ Success: ${successCount}`)
      console.log(`   ❌ Failed: ${failureCount}`)
      console.log(`   ⏭️  Skipped: ${skippedCount}\n`)
    }
  }

  console.log('\n🎉 Company population complete!')
  console.log(`\n📊 Final Statistics:`)
  console.log(`   Total companies: ${records.length}`)
  console.log(`   ✅ Successfully processed: ${successCount}`)
  console.log(`   ❌ Failed: ${failureCount}`)
  console.log(`   ⏭️  Skipped: ${skippedCount}`)
}

main().catch(console.error)
