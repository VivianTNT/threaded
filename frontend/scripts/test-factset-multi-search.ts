#!/usr/bin/env npx tsx

/**
 * Test FactSet API with multiple company IDs to see response format
 */

import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

async function testMultipleIds() {
  console.log('🧪 Testing FactSet API with Multiple Company IDs\n')

  // Test with 3 major mining companies
  const testCompanies = [
    { name: 'Barrick Gold', ticker: 'GOLD-US' },
    { name: 'Newmont', ticker: 'NEM-US' },
    { name: 'Freeport-McMoRan', ticker: 'FCX-US' }
  ]

  console.log('Test companies:')
  testCompanies.forEach(c => console.log(`  - ${c.name} (${c.ticker})`))
  console.log()

  // METHOD 1: Search with comma-separated IDs (typical batch approach)
  console.log('=' .repeat(80))
  console.log('METHOD 1: Comma-separated IDs in single request')
  console.log('='.repeat(80))

  const multiIds = testCompanies.map(c => c.ticker).join(',')
  const params1 = new URLSearchParams({
    ids: multiIds, // "GOLD-US,NEM-US,FCX-US"
    sources: 'EDG',
    startDate: '20230101',
    endDate: '20250201',
    searchText: 'mining',
    _paginationLimit: '5'
  })

  const url1 = `https://api.factset.com/content/global-filings/v2/search?${params1.toString()}`
  console.log('\nRequest URL:', url1)
  console.log('\nSending request...\n')

  try {
    const response = await fetch(url1, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/json'
      }
    })

    if (!response.ok) {
      console.error(`❌ Failed: ${response.status} ${response.statusText}`)
      const errorText = await response.text()
      console.error('Error:', errorText.substring(0, 500))
    } else {
      const data = await response.json()
      console.log('✅ Success!')
      console.log('\nResponse structure:')
      console.log(JSON.stringify(data, null, 2))

      // Analyze response
      if (data.data && Array.isArray(data.data)) {
        console.log(`\n📊 Analysis:`)
        console.log(`   Total result groups: ${data.data.length}`)
        data.data.forEach((result: any, idx: number) => {
          console.log(`\n   Group ${idx + 1}:`)
          console.log(`     Request ID: ${result.requestId}`)
          console.log(`     Documents: ${result.documents?.length || 0}`)
          if (result.documents && result.documents.length > 0) {
            console.log(`     Sample doc: ${result.documents[0].headline.substring(0, 60)}...`)
          }
        })
      }

      if (data.meta?.pagination) {
        console.log(`\n   Total available: ${data.meta.pagination.total}`)
      }
    }
  } catch (error: any) {
    console.error('❌ Error:', error.message)
  }

  // METHOD 2: Individual requests for comparison
  console.log('\n\n' + '='.repeat(80))
  console.log('METHOD 2: Individual requests (current implementation)')
  console.log('='.repeat(80))

  for (const company of testCompanies.slice(0, 1)) { // Just test first one
    console.log(`\n\n🔍 ${company.name} (${company.ticker})`)

    const params2 = new URLSearchParams({
      ids: company.ticker,
      sources: 'EDG',
      startDate: '20230101',
      endDate: '20250201',
      searchText: 'mining',
      _paginationLimit: '3'
    })

    const url2 = `https://api.factset.com/content/global-filings/v2/search?${params2.toString()}`

    try {
      const response = await fetch(url2, {
        headers: {
          'Authorization': authHeader,
          'Accept': 'application/json'
        }
      })

      if (response.ok) {
        const data = await response.json()
        console.log('   ✅ Success')
        console.log(`   Documents found: ${data.data?.[0]?.documents?.length || 0}`)

        if (data.data?.[0]?.documents?.[0]) {
          const doc = data.data[0].documents[0]
          console.log(`   Sample: ${doc.headline.substring(0, 80)}...`)
          console.log(`   Size: ${doc.filingSize}`)
          console.log(`   Form: ${doc.formTypes?.[0]}`)
        }
      } else {
        console.log(`   ❌ Failed: ${response.status}`)
      }
    } catch (error: any) {
      console.log(`   ❌ Error: ${error.message}`)
    }
  }
}

testMultipleIds().catch(console.error)
