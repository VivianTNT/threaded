#!/usr/bin/env node

/**
 * Test FactSet Document Download
 *
 * This script tests downloading and viewing FactSet filing documents.
 * Based on FactSet's feedback, we need to provide authentication in BOTH:
 * 1. Configuration (when initializing the API client)
 * 2. Authorization (when downloading the documents)
 *
 * Run with: npx tsx scripts/test-factset-document-download.ts
 */

import { config } from 'dotenv'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

// ============================
// Configuration
// ============================

const FACTSET_USERNAME = process.env.FACTSET_USERNAME
const FACTSET_API_KEY = process.env.FACTSET_API_KEY
const FACTSET_BASE_URL = 'https://api.factset.com/content/global-filings/v2'
const OUTPUT_DIR = path.join(__dirname, '..', 'downloads', 'factset-test')

// Test companies (Canadian mining companies known to file with SEDAR)
const TEST_COMPANIES = [
  { name: 'Ivanhoe Mines', ticker: 'IVN-CA' },
  { name: 'Teck Resources', ticker: 'TECK-CA' },
  { name: 'First Quantum Minerals', ticker: 'FM-CA' }
]

// ============================
// Type Definitions
// ============================

interface FactSetFilingDocument {
  headline: string
  source: string
  filingsDateTime: string
  filingsLink: string
  documentId: string
  filingSize?: string
  formTypes?: string[]
}

interface FactSetSearchResponse {
  data: Array<{
    requestId: string
    documents?: FactSetFilingDocument[]
    error?: {
      code: string
      title: string
      detail: string
    }
  }>
}

// ============================
// Utility Functions
// ============================

/**
 * Create Basic Auth header
 */
function createAuthHeader(): string {
  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    throw new Error('FactSet credentials not configured')
  }
  return 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
}

/**
 * Search for filings
 */
async function searchFilings(ticker: string, limit: number = 5): Promise<FactSetFilingDocument[]> {
  const params = new URLSearchParams({
    ids: ticker,
    sources: 'SDRP', // SEDAR+ for recent filings
    startDate: '20250101',
    _paginationLimit: String(limit),
    _paginationOffset: '0',
    timeZone: 'America/Toronto',
    _sort: '-filingsDateTime'
  })

  const url = `${FACTSET_BASE_URL}/search?${params.toString()}`

  console.log(`   🔍 Searching: ${url}`)

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Authorization': createAuthHeader(),
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    }
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`API error (${response.status}): ${errorText}`)
  }

  const data: FactSetSearchResponse = await response.json()

  if (!data.data || data.data.length === 0) {
    return []
  }

  const result = data.data[0]
  if (result.error) {
    console.warn(`   ⚠️  API Error: ${result.error.detail}`)
    return []
  }

  return result.documents || []
}

/**
 * Download a filing document
 *
 * IMPORTANT: This function provides authentication in BOTH:
 * - The Authorization header (as required by FactSet)
 * - The request configuration
 */
async function downloadDocument(
  filing: FactSetFilingDocument,
  outputPath: string
): Promise<boolean> {
  try {
    console.log(`   📥 Downloading from: ${filing.filingsLink}`)

    // According to FactSet documentation, we need to pass the document through
    // the /filings endpoint with proper authentication
    const response = await fetch(filing.filingsLink, {
      method: 'GET',
      headers: {
        'Authorization': createAuthHeader(), // CRITICAL: Auth in header
        'Accept': 'application/pdf,application/octet-stream,*/*',
      }
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error(`   ❌ Download failed (${response.status}): ${errorText}`)
      return false
    }

    // Get the content type to determine file extension
    const contentType = response.headers.get('content-type') || ''
    let fileExtension = '.pdf'

    if (contentType.includes('pdf')) {
      fileExtension = '.pdf'
    } else if (contentType.includes('html')) {
      fileExtension = '.html'
    } else if (contentType.includes('xml')) {
      fileExtension = '.xml'
    }

    // Save the file
    const buffer = await response.arrayBuffer()
    const finalPath = outputPath.replace(/\.[^.]+$/, fileExtension)
    fs.writeFileSync(finalPath, Buffer.from(buffer))

    const fileSizeKB = Math.round(buffer.byteLength / 1024)
    console.log(`   ✅ Downloaded: ${path.basename(finalPath)} (${fileSizeKB} KB)`)
    console.log(`   📄 Content-Type: ${contentType}`)
    console.log(`   💾 Saved to: ${finalPath}`)

    return true
  } catch (error: any) {
    console.error(`   ❌ Download error: ${error.message}`)
    return false
  }
}

/**
 * Get document metadata (without downloading)
 */
async function getDocumentMetadata(filing: FactSetFilingDocument): Promise<void> {
  try {
    console.log(`   🔍 Fetching metadata from: ${filing.filingsLink}`)

    const response = await fetch(filing.filingsLink, {
      method: 'HEAD',
      headers: {
        'Authorization': createAuthHeader(),
        'Accept': '*/*',
      }
    })

    console.log(`   📊 Status: ${response.status} ${response.statusText}`)
    console.log(`   📋 Headers:`)
    response.headers.forEach((value, key) => {
      console.log(`      ${key}: ${value}`)
    })
  } catch (error: any) {
    console.error(`   ❌ Metadata error: ${error.message}`)
  }
}

// ============================
// Main Process
// ============================

async function main() {
  console.log('='.repeat(70))
  console.log('FACTSET DOCUMENT DOWNLOAD TEST')
  console.log('='.repeat(70))

  // Validate credentials
  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    console.error('\n❌ ERROR: FactSet credentials not configured')
    console.error('Please set FACTSET_USERNAME and FACTSET_API_KEY in .env.local')
    process.exit(1)
  }

  console.log(`\n✅ Credentials configured`)
  console.log(`🔑 Username: ${FACTSET_USERNAME}`)
  console.log(`📡 API URL: ${FACTSET_BASE_URL}`)

  // Create output directory
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true })
  }
  console.log(`📁 Output directory: ${OUTPUT_DIR}`)

  let totalFilings = 0
  let totalDownloads = 0
  let successfulDownloads = 0
  let failedDownloads = 0

  console.log('\n' + '='.repeat(70))
  console.log('TESTING COMPANIES')
  console.log('='.repeat(70))

  // Test each company
  for (const company of TEST_COMPANIES) {
    console.log(`\n📊 ${company.name} (${company.ticker})`)
    console.log('-'.repeat(70))

    try {
      // Search for filings
      const filings = await searchFilings(company.ticker, 3)

      if (filings.length === 0) {
        console.log('   ℹ️  No recent filings found')
        continue
      }

      totalFilings += filings.length
      console.log(`   ✅ Found ${filings.length} filing(s)`)

      // Display filing details and attempt download
      for (let i = 0; i < filings.length; i++) {
        const filing = filings[i]
        console.log(`\n   📄 Filing ${i + 1}/${filings.length}:`)
        console.log(`      Headline: ${filing.headline}`)
        console.log(`      Date: ${new Date(filing.filingsDateTime).toLocaleDateString()}`)
        console.log(`      Form: ${filing.formTypes?.join(', ') || 'N/A'}`)
        console.log(`      Document ID: ${filing.documentId}`)
        console.log(`      Size: ${filing.filingSize || 'Unknown'}`)

        // Get metadata first
        await getDocumentMetadata(filing)

        // Attempt download
        const filename = `${company.ticker.replace('-CA', '')}_${filing.documentId}.pdf`
        const outputPath = path.join(OUTPUT_DIR, filename)

        totalDownloads++
        const success = await downloadDocument(filing, outputPath)

        if (success) {
          successfulDownloads++
        } else {
          failedDownloads++
        }

        // Small delay between downloads
        await new Promise(resolve => setTimeout(resolve, 500))
      }

    } catch (error: any) {
      console.error(`   ❌ Error processing ${company.name}: ${error.message}`)
      failedDownloads++
    }
  }

  // Summary
  console.log('\n' + '='.repeat(70))
  console.log('TEST COMPLETE')
  console.log('='.repeat(70))
  console.log(`\n📊 Summary:`)
  console.log(`   Total filings found: ${totalFilings}`)
  console.log(`   Download attempts: ${totalDownloads}`)
  console.log(`   Successful downloads: ${successfulDownloads}`)
  console.log(`   Failed downloads: ${failedDownloads}`)
  console.log(`\n📁 Downloaded files saved to: ${OUTPUT_DIR}`)

  if (successfulDownloads > 0) {
    console.log(`\n✅ Success! You can now view the downloaded documents.`)
    console.log(`\n💡 To view the files:`)
    console.log(`   cd ${OUTPUT_DIR}`)
    console.log(`   open .  # macOS`)
  } else {
    console.log(`\n⚠️  No documents were successfully downloaded.`)
    console.log(`\n🔍 Troubleshooting steps:`)
    console.log(`   1. Verify your FactSet credentials are correct`)
    console.log(`   2. Check that your account has the required CACCESS permissions`)
    console.log(`   3. Contact FactSet support if the issue persists`)
  }

  console.log('\n' + '='.repeat(70))
}

// Execute
main().catch(error => {
  console.error('\n❌ Fatal error:', error)
  process.exit(1)
})
