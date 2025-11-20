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
  market_cap?: number | null
  description?: string
  website?: string | null
}

// Financial Modeling Prep API
const FMP_API_KEY = process.env.FMP_API_KEY || 'your_api_key_here'

// Rate limiting
async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Convert ALL CAPS to Title Case
function toTitleCase(str: string): string {
  if (!str) return str

  // List of words that should remain lowercase (unless at start)
  const lowerWords = new Set(['and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'])

  // List of abbreviations that should remain uppercase
  const upperWords = new Set(['USA', 'UK', 'NV', 'PLC', 'LLC', 'INC', 'CORP', 'LTD', 'AG', 'SA', 'NL', 'SE', 'AB'])

  return str
    .toLowerCase()
    .split(' ')
    .map((word, index) => {
      // Remove trailing punctuation for checking
      const cleanWord = word.replace(/[.,;:!?]$/, '')
      const punctuation = word.slice(cleanWord.length)

      // Keep uppercase abbreviations
      if (upperWords.has(cleanWord.toUpperCase())) {
        return cleanWord.toUpperCase() + punctuation
      }

      // Keep lowercase words (except at start of sentence)
      if (index > 0 && lowerWords.has(cleanWord)) {
        return cleanWord + punctuation
      }

      // Capitalize first letter
      return cleanWord.charAt(0).toUpperCase() + cleanWord.slice(1) + punctuation
    })
    .join(' ')
}

// Fetch company profile from Financial Modeling Prep
async function fetchFMPCompanyProfile(ticker: string, exchange: string): Promise<{ marketCap: number | null, website: string | null }> {
  try {
    // Try different ticker formats
    const tickerFormats = [
      ticker,
      `${ticker}.${exchange}`,
      `${exchange}:${ticker}`
    ]

    for (const tickerFormat of tickerFormats) {
      try {
        const url = `https://financialmodelingprep.com/api/v3/profile/${tickerFormat}?apikey=${FMP_API_KEY}`
        const response = await fetch(url)

        if (response.ok) {
          const data = await response.json()
          if (data && data.length > 0) {
            const profile = data[0]
            return {
              marketCap: profile.mktCap || null,
              website: profile.website || null
            }
          }
        }
      } catch (err) {
        // Try next format
        continue
      }
    }

    return { marketCap: null, website: null }
  } catch (error) {
    console.error(`Error fetching FMP data for ${ticker}:`, error)
    return { marketCap: null, website: null }
  }
}

// Generate a basic description for mining companies
function generateMiningDescription(company: MiningCompany): string {
  const { Name, Sector, CommodityFocus, Country } = company

  let desc = `${toTitleCase(Name)} is a`

  if (CommodityFocus) {
    desc += ` ${CommodityFocus.toLowerCase()}`
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
async function upsertCompany(companyData: CompanyData): Promise<boolean> {
  try {
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
      console.error(`  ❌ Database error for ${companyData.ticker}:`, error.message)
      return false
    }

    return true
  } catch (error) {
    console.error(`  ❌ Upsert error for ${companyData.ticker}:`, error)
    return false
  }
}

async function main() {
  console.log('🚀 Starting enhanced mining company population...\n')
  console.log('📋 Features:')
  console.log('   ✅ Title Case name conversion')
  console.log('   ✅ Market cap fetching (FMP API)')
  console.log('   ✅ Website URL fetching')
  console.log('   ✅ Rate limiting for API calls\n')

  // Read CSV file
  const csvPath = path.join(process.cwd(), 'Mining Ticker List.csv')

  if (!fs.existsSync(csvPath)) {
    console.error('❌ Mining Ticker List.csv not found!')
    return
  }

  const csvContent = fs.readFileSync(csvPath, 'utf-8')

  const records: MiningCompany[] = parse(csvContent, {
    columns: true,
    skip_empty_lines: true
  })

  console.log(`📊 Found ${records.length} mining companies in CSV\n`)

  // Check if FMP API key is available
  const useFMP = FMP_API_KEY && FMP_API_KEY !== 'your_api_key_here'
  if (useFMP) {
    console.log('✅ FMP API key detected - will fetch market cap and website data')
  } else {
    console.log('⚠️  No FMP API key - will skip market cap and website fetching')
    console.log('   Set FMP_API_KEY environment variable to enable data enrichment\n')
  }

  let successCount = 0
  let failureCount = 0
  let skippedCount = 0
  let enrichedCount = 0

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
        name: toTitleCase(company.Name),
        ticker: company.Ticker,
        exchange: company.Exchange,
        country: company.Country,
        description: generateMiningDescription(company)
      }

      // Try to fetch additional data from FMP
      if (useFMP) {
        console.log(`  🔍 Fetching FMP data...`)

        const fmpData = await fetchFMPCompanyProfile(company.Ticker, company.Exchange)

        if (fmpData.marketCap || fmpData.website) {
          enrichedCount++
          if (fmpData.marketCap) {
            companyData.market_cap = fmpData.marketCap
            console.log(`  💰 Market cap: $${(fmpData.marketCap / 1000000).toFixed(0)}M`)
          }
          if (fmpData.website) {
            companyData.website = fmpData.website
            console.log(`  🌐 Website: ${fmpData.website}`)
          }
        } else {
          console.log(`  ℹ️  No FMP data found`)
        }

        // Rate limiting - 5 requests per second max
        await sleep(200)
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
      console.log(`   ⏭️  Skipped: ${skippedCount}`)
      console.log(`   💎 Enriched with data: ${enrichedCount}\n`)
    }
  }

  console.log('\n🎉 Company population complete!')
  console.log(`\n📊 Final Statistics:`)
  console.log(`   Total companies: ${records.length}`)
  console.log(`   ✅ Successfully processed: ${successCount}`)
  console.log(`   ❌ Failed: ${failureCount}`)
  console.log(`   ⏭️  Skipped: ${skippedCount}`)
  console.log(`   💎 Enriched with market/website data: ${enrichedCount}`)
  console.log(`   📈 Enrichment rate: ${((enrichedCount / successCount) * 100).toFixed(1)}%`)
}

main().catch(console.error)
