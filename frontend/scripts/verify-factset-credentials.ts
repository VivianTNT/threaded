#!/usr/bin/env node

/**
 * Verify FactSet API Credentials
 *
 * Tests basic authentication against FactSet's meta endpoints
 * which should be accessible with valid credentials.
 */

import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME
const FACTSET_API_KEY = process.env.FACTSET_API_KEY

async function testEndpoint(name: string, url: string) {
  console.log(`\n🔍 Testing: ${name}`)
  console.log(`   URL: ${url}`)

  const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
  console.log(`   Auth: ${authHeader.substring(0, 30)}...`)

  try {
    const response = await fetch(url, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/json'
      }
    })

    console.log(`   Status: ${response.status} ${response.statusText}`)

    if (response.ok) {
      const data = await response.json()
      console.log(`   ✅ SUCCESS`)
      console.log(`   Response preview: ${JSON.stringify(data).substring(0, 200)}...`)
      return true
    } else {
      const errorText = await response.text()
      console.log(`   ❌ FAILED: ${errorText}`)
      return false
    }
  } catch (error: any) {
    console.log(`   ❌ ERROR: ${error.message}`)
    return false
  }
}

async function main() {
  console.log('='.repeat(70))
  console.log('FACTSET CREDENTIALS VERIFICATION')
  console.log('='.repeat(70))

  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    console.error('\n❌ ERROR: Credentials not configured')
    console.error('\nPlease set in .env.local:')
    console.error('  FACTSET_USERNAME=your-username')
    console.error('  FACTSET_API_KEY=your-api-key')
    process.exit(1)
  }

  console.log(`\n📋 Credentials found:`)
  console.log(`   Username: ${FACTSET_USERNAME}`)
  console.log(`   API Key: ${FACTSET_API_KEY.substring(0, 10)}${'*'.repeat(Math.max(0, FACTSET_API_KEY.length - 10))}`)

  console.log('\n' + '='.repeat(70))
  console.log('TESTING API ENDPOINTS')
  console.log('='.repeat(70))

  const baseUrl = 'https://api.factset.com/content/global-filings/v2'

  const results = []

  // Test 1: Meta/Sources endpoint (should work with valid credentials)
  results.push(await testEndpoint(
    'Sources Metadata',
    `${baseUrl}/meta/sources`
  ))

  // Test 2: Meta/Timezones endpoint
  results.push(await testEndpoint(
    'Timezones Metadata',
    `${baseUrl}/meta/time-zones`
  ))

  // Test 3: Meta/Categories endpoint
  results.push(await testEndpoint(
    'Categories Metadata',
    `${baseUrl}/meta/categories`
  ))

  // Test 4: Simple search
  results.push(await testEndpoint(
    'Simple Search',
    `${baseUrl}/search?ids=IVN-CA&sources=SDRP&startDate=20240101&_paginationLimit=1`
  ))

  console.log('\n' + '='.repeat(70))
  console.log('VERIFICATION RESULTS')
  console.log('='.repeat(70))

  const successCount = results.filter(r => r).length
  const totalTests = results.length

  console.log(`\n📊 Tests passed: ${successCount}/${totalTests}`)

  if (successCount === totalTests) {
    console.log(`\n✅ ALL TESTS PASSED!`)
    console.log(`Your FactSet credentials are valid and the API is accessible.`)
  } else if (successCount > 0) {
    console.log(`\n⚠️  PARTIAL SUCCESS`)
    console.log(`Some endpoints work but others don't. This may indicate:`)
    console.log(`   - Permission/access level issues`)
    console.log(`   - Incorrect API parameters for some endpoints`)
  } else {
    console.log(`\n❌ ALL TESTS FAILED`)
    console.log(`\nPossible issues:`)
    console.log(`   1. Incorrect username or API key`)
    console.log(`   2. Account doesn't have API access enabled`)
    console.log(`   3. Network/firewall issues`)
    console.log(`\n📧 Contact FactSet support: misa.kobayashi@factset.com`)
  }

  console.log('\n' + '='.repeat(70))
}

main().catch(console.error)
