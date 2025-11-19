#!/usr/bin/env node

/**
 * Simple FactSet Document Download Test
 *
 * Tests downloading a specific document using the URL from your email.
 * This tests the authentication issue directly.
 */

import { config } from 'dotenv'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME
const FACTSET_API_KEY = process.env.FACTSET_API_KEY
const OUTPUT_DIR = path.join(__dirname, '..', 'downloads', 'factset-test')

// The URL from your email that gave the CACCESS error
const TEST_FILING_URL = 'https://api.factset.com/global-filings/v2/filings?report=story&timezone=America/New_York&key=U2FsdGVkX19SDntKS7laMA+hhV0VaFmCPdcEfQaUjL8cJBsmy4Xzw5eOsd/z98Nr6/rDhgDHpiFDCE5iVf3JbxKdkj1PkSKsQxU7Rzuh0SFLeHnjPZmUO8Q14FnIhiMd'

async function createAuthHeader(): Promise<string> {
  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    throw new Error('FactSet credentials not configured')
  }
  return 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
}

async function testDownload() {
  console.log('='.repeat(70))
  console.log('FACTSET SIMPLE DOWNLOAD TEST')
  console.log('='.repeat(70))

  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    console.error('\n❌ ERROR: FactSet credentials not configured')
    console.error('Please set FACTSET_USERNAME and FACTSET_API_KEY in .env.local')
    process.exit(1)
  }

  console.log(`\n✅ Credentials configured`)
  console.log(`🔑 Username: ${FACTSET_USERNAME}`)

  // Create output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  }

  console.log(`\n📥 Testing download from your email's URL...`)
  console.log(`🔗 URL: ${TEST_FILING_URL}`)

  try {
    const authHeader = await createAuthHeader()
    console.log(`\n🔐 Auth Header: ${authHeader.substring(0, 20)}...`)

    console.log(`\n📡 Making request...`)
    const response = await fetch(TEST_FILING_URL, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf,application/html,application/json,*/*',
        'User-Agent': 'Lithos-AI/1.0'
      }
    })

    console.log(`\n📊 Response Status: ${response.status} ${response.statusText}`)
    console.log(`📋 Response Headers:`)
    response.headers.forEach((value, key) => {
      console.log(`   ${key}: ${value}`)
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`\n❌ Download failed (${response.status}): ${errorText}`)

      if (response.status === 403 && errorText.includes('CACCESS')) {
        console.log(`\n⚠️  CACCESS ERROR DETECTED`)
        console.log(`This suggests your account doesn't have the required permissions.`)
        console.log(`\nPossible solutions:`)
        console.log(`1. Contact FactSet support to enable CACCESS for your account`)
        console.log(`2. Verify you're using the machine account credentials (not personal)`)
        console.log(`3. Check if there are additional API setup steps required`)
      }

      return
    }

    // Success! Save the file
    const contentType = response.headers.get('content-type') || ''
    let fileExtension = '.pdf'

    if (contentType.includes('pdf')) {
      fileExtension = '.pdf'
    } else if (contentType.includes('html')) {
      fileExtension = '.html'
    } else if (contentType.includes('json')) {
      fileExtension = '.json'
    }

    const buffer = await response.arrayBuffer()
    const filename = `ivanhoe_test_document${fileExtension}`
    const filepath = path.join(OUTPUT_DIR, filename)

    fs.writeFileSync(filepath, Buffer.from(buffer))

    const fileSizeKB = Math.round(buffer.byteLength / 1024)
    console.log(`\n✅ SUCCESS! Document downloaded`)
    console.log(`📄 Content-Type: ${contentType}`)
    console.log(`💾 File size: ${fileSizeKB} KB`)
    console.log(`📁 Saved to: ${filepath}`)
    console.log(`\n💡 To view the file:`)
    console.log(`   open "${filepath}"`)

  } catch (error: any) {
    console.error(`\n❌ Error: ${error.message}`)
    console.error(error.stack)
  }

  console.log('\n' + '='.repeat(70))
}

testDownload().catch(console.error)
