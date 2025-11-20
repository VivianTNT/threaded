#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!

// Mining companies with their FactSet identifiers
const MINING_COMPANIES = [
  { ticker: 'BHP-AU', name: 'BHP Group' },
  { ticker: 'RIO-GB', name: 'Rio Tinto' },
  { ticker: 'VALE-US', name: 'Vale' },
  { ticker: 'FCX-US', name: 'Freeport-McMoRan' },
  { ticker: 'NEM-US', name: 'Newmont' },
  { ticker: 'GOLD-US', name: 'Barrick Gold' },
  { ticker: 'GLEN-GB', name: 'Glencore' },
  { ticker: 'SCCO-US', name: 'Southern Copper' },
  { ticker: 'TECK-CA', name: 'Teck Resources' },
  { ticker: 'FM-CA', name: 'First Quantum Minerals' }
]

async function fetchFactSetDocuments() {
  console.log('🏭 FETCHING TECHNICAL DOCUMENTS FROM FACTSET API')
  console.log('='.repeat(60))

  for (const company of MINING_COMPANIES) {
    console.log(`\n📊 Fetching documents for ${company.name} (${company.ticker})...`)

    try {
      // FactSet Filings Documents API endpoint
      const response = await fetch('https://api.factset.com/content/factset-global-filings/v1/filings', {
        method: 'POST',
        headers: {
          'Authorization': 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64'),
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          ids: [company.ticker],
          startDate: '2023-01-01',
          endDate: '2024-12-31',
          categories: ['AR'], // Annual Reports
          limit: 5
        })
      })

      if (response.ok) {
        const data = await response.json()
        console.log(`   Response received:`, JSON.stringify(data, null, 2).substring(0, 500))

        if (data.data && Array.isArray(data.data)) {
          console.log(`   ✅ Found ${data.data.length} documents`)

          // Process each document
          for (const doc of data.data) {
            if (doc.url && doc.url.includes('.pdf')) {
              console.log(`      📄 ${doc.title || doc.formType}: ${doc.url.substring(0, 80)}...`)
            }
          }
        } else {
          console.log(`   ⚠️ No documents found in response`)
        }
      } else {
        const errorText = await response.text()
        console.log(`   ❌ API Error ${response.status}: ${errorText.substring(0, 200)}`)
      }
    } catch (error: any) {
      console.log(`   ❌ Request failed: ${error.message}`)
    }

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 1000))
  }

  console.log('\n' + '='.repeat(60))
  console.log('✅ FactSet document fetch complete')
  console.log('\nNote: FactSet API requires proper subscription and access rights.')
  console.log('The API may return different results based on your subscription level.')
}

// Alternative: Try FactSet Entity API for company information
async function fetchFactSetEntities() {
  console.log('\n📊 Trying FactSet Entity API...')
  console.log('='.repeat(60))

  try {
    const response = await fetch('https://api.factset.com/content/factset-entity/v1/entities', {
      method: 'POST',
      headers: {
        'Authorization': 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64'),
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        ids: ['BHP-AU', 'RIO-GB', 'VALE-US'],
        fields: ['name', 'entityType', 'website', 'description']
      })
    })

    if (response.ok) {
      const data = await response.json()
      console.log('Entity API Response:', JSON.stringify(data, null, 2))
    } else {
      console.log(`Entity API Error ${response.status}: ${await response.text()}`)
    }
  } catch (error: any) {
    console.log(`Entity API failed: ${error.message}`)
  }
}

// Alternative: Try FactSet Fundamentals API
async function fetchFactSetFundamentals() {
  console.log('\n💰 Trying FactSet Fundamentals API...')
  console.log('='.repeat(60))

  try {
    const response = await fetch('https://api.factset.com/content/factset-fundamentals/v2/fundamentals', {
      method: 'POST',
      headers: {
        'Authorization': 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64'),
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({
        ids: ['BHP-AU', 'RIO-GB'],
        metrics: ['SALES', 'EBITDA', 'TOTAL_ASSETS'],
        periodicity: 'ANN',
        fiscalPeriodStart: '2023-01-01',
        fiscalPeriodEnd: '2023-12-31'
      })
    })

    if (response.ok) {
      const data = await response.json()
      console.log('Fundamentals API Response:', JSON.stringify(data, null, 2))
    } else {
      console.log(`Fundamentals API Error ${response.status}: ${await response.text()}`)
    }
  } catch (error: any) {
    console.log(`Fundamentals API failed: ${error.message}`)
  }
}

// Run all attempts
async function main() {
  console.log('🔍 Testing FactSet API Access')
  console.log('API Key:', FACTSET_API_KEY ? `${FACTSET_API_KEY.substring(0, 10)}...` : 'MISSING')
  console.log('Username:', FACTSET_USERNAME || 'MISSING')
  console.log('')

  await fetchFactSetDocuments()
  await fetchFactSetEntities()
  await fetchFactSetFundamentals()

  console.log('\n' + '='.repeat(60))
  console.log('🏁 All FactSet API tests complete')
  console.log('Check the output above to see which endpoints are accessible')
  console.log('with your current subscription level.')
}

main().catch(console.error)