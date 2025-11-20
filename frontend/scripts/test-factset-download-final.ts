#!/usr/bin/env node

/**
 * FactSet Document Download Test - Final Version
 *
 * Simple REST API approach to test document downloads.
 * Focuses on showing document metadata and sample downloads.
 */

import { config } from 'dotenv'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME
const FACTSET_API_KEY = process.env.FACTSET_API_KEY
const BASE_URL = 'https://api.factset.com/content/global-filings/v2'
const OUTPUT_DIR = path.join(__dirname, '..', 'downloads', 'factset-test')

interface Filing {
  headline: string
  filingsDateTime: string
  formTypes?: string[]
  documentId: string
  filingsLink: string
  filingSize?: string
}

function createAuthHeader(): string {
  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    throw new Error('Credentials not configured')
  }
  return 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
}

async function searchFilings(ticker: string): Promise<Filing[]> {
  const params = new URLSearchParams({
    ids: ticker,
    sources: 'SDR,SDRP', // Both old and new SEDAR
    startDate: '20240101',
    _paginationLimit: '10',
    timeZone: 'America/Toronto',
    _sort: '-filingsDateTime'
  })

  const url = `${BASE_URL}/search?${params.toString()}`

  const response = await fetch(url, {
    headers: {
      'Authorization': createAuthHeader(),
      'Accept': 'application/json'
    }
  })

  if (!response.ok) {
    throw new Error(`Search failed (${response.status}): ${await response.text()}`)
  }

  const data = await response.json()
  return data.data?.[0]?.documents || []
}

async function downloadDocument(filing: Filing, outputPath: string): Promise<{ success: boolean; message: string; size?: number }> {
  try {
    console.log(`      🔗 URL: ${filing.filingsLink}`)

    const response = await fetch(filing.filingsLink, {
      headers: {
        'Authorization': createAuthHeader(),
        'Accept': '*/*'
      }
    })

    const status = `${response.status} ${response.statusText}`
    const contentType = response.headers.get('content-type') || 'unknown'

    if (!response.ok) {
      const errorText = await response.text()

      if (errorText.includes('CACCESS')) {
        return {
          success: false,
          message: `❌ CACCESS Permission Required - Contact FactSet support`
        }
      }

      return {
        success: false,
        message: `❌ HTTP ${status}: ${errorText.substring(0, 100)}`
      }
    }

    // Successful download
    const buffer = await response.arrayBuffer()

    // Determine file extension from content type
    let ext = '.pdf'
    if (contentType.includes('html')) ext = '.html'
    else if (contentType.includes('xml')) ext = '.xml'
    else if (contentType.includes('json')) ext = '.json'

    const finalPath = outputPath.replace(/\.[^.]+$/, ext)
    fs.writeFileSync(finalPath, Buffer.from(buffer))

    const sizeKB = Math.round(buffer.byteLength / 1024)

    return {
      success: true,
      message: `✅ Downloaded ${sizeKB} KB (${contentType})`,
      size: sizeKB
    }

  } catch (error: any) {
    return {
      success: false,
      message: `❌ Error: ${error.message}`
    }
  }
}

async function main() {
  console.log('='.repeat(70))
  console.log('FACTSET DOCUMENT DOWNLOAD TEST - Sample Extraction')
  console.log('='.repeat(70))

  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    console.error('\n❌ ERROR: FactSet credentials not configured')
    console.error('Set FACTSET_USERNAME and FACTSET_API_KEY in .env.local')
    process.exit(1)
  }

  console.log(`\n✅ Credentials: ${FACTSET_USERNAME}`)

  // Create output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  }
  console.log(`📁 Output: ${OUTPUT_DIR}`)

  const testCompanies = [
    { name: 'Ivanhoe Mines', ticker: 'IVN-CA' },
    { name: 'Teck Resources', ticker: 'TECK-CA' },
    { name: 'First Quantum', ticker: 'FM-CA' }
  ]

  console.log('\n' + '='.repeat(70))
  console.log('SEARCHING AND DOWNLOADING DOCUMENTS')
  console.log('='.repeat(70))

  let totalDocs = 0
  let successfulDownloads = 0
  let cacccessErrors = 0

  for (const company of testCompanies) {
    console.log(`\n📊 ${company.name} (${company.ticker})`)
    console.log('-'.repeat(70))

    try {
      const filings = await searchFilings(company.ticker)

      if (filings.length === 0) {
        console.log(`   ℹ️  No filings found`)
        continue
      }

      console.log(`   ✅ Found ${filings.length} filing(s)`)
      totalDocs += filings.length

      // Show details and attempt download for first 3 filings
      for (let i = 0; i < Math.min(filings.length, 3); i++) {
        const filing = filings[i]

        console.log(`\n   📄 Filing ${i + 1}:`)
        console.log(`      Title: ${filing.headline.substring(0, 80)}${filing.headline.length > 80 ? '...' : ''}`)
        console.log(`      Date: ${new Date(filing.filingsDateTime).toLocaleDateString()}`)
        console.log(`      Form: ${filing.formTypes?.join(', ') || 'N/A'}`)
        console.log(`      ID: ${filing.documentId}`)
        console.log(`      Size: ${filing.filingSize || 'Unknown'}`)

        // Attempt download
        console.log(`\n      📥 Downloading...`)
        const filename = `${company.ticker.replace('-CA', '')}_${i + 1}_${filing.documentId}.pdf`
        const filepath = path.join(OUTPUT_DIR, filename)

        const result = await downloadDocument(filing, filepath)
        console.log(`      ${result.message}`)

        if (result.success) {
          successfulDownloads++
          console.log(`      💾 Saved: ${filename}`)
        } else if (result.message.includes('CACCESS')) {
          cacccessErrors++
        }

        await new Promise(resolve => setTimeout(resolve, 300))
      }

    } catch (error: any) {
      console.error(`   ❌ Error: ${error.message}`)
    }
  }

  // Summary
  console.log('\n' + '='.repeat(70))
  console.log('SUMMARY')
  console.log('='.repeat(70))
  console.log(`\n📊 Results:`)
  console.log(`   Total documents found: ${totalDocs}`)
  console.log(`   Successful downloads: ${successfulDownloads}`)
  console.log(`   CACCESS errors: ${cacccessErrors}`)

  console.log(`\n📁 Files location: ${OUTPUT_DIR}`)

  if (successfulDownloads > 0) {
    console.log(`\n✅ SUCCESS! ${successfulDownloads} document(s) downloaded and ready to view.`)
    console.log(`\n💡 To view files:`)
    console.log(`   open "${OUTPUT_DIR}"`)
  } else if (cacccessErrors > 0) {
    console.log(`\n⚠️  CACCESS PERMISSION REQUIRED`)
    console.log(`\nYour FactSet account can access the API and find documents,`)
    console.log(`but needs CACCESS permissions to download the actual content.`)
    console.log(`\n📧 Contact: misa.kobayashi@factset.com`)
    console.log(`Request: Enable CACCESS permissions for account ${FACTSET_USERNAME}`)
  } else {
    console.log(`\n⚠️  No documents were downloaded. Check errors above.`)
  }

  console.log('\n' + '='.repeat(70))
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error.message)
  process.exit(1)
})
