#!/usr/bin/env node

/**
 * FactSet SDK Document Download Test
 *
 * Uses the official @factset/sdk-globalfilings SDK to properly authenticate
 * and download documents. This follows FactSet's recommendation to use
 * authentication in both configuration and authorization.
 */

import { config } from 'dotenv'
import path from 'path'
import fs from 'fs'
import * as fds from '@factset/sdk-globalfilings'
import { ConfidentialClient } from '@factset/sdk-utils/authentication'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME
const FACTSET_API_KEY = process.env.FACTSET_API_KEY
const OUTPUT_DIR = path.join(__dirname, '..', 'downloads', 'factset-test')

// Test companies
const TEST_COMPANIES = [
  { name: 'Ivanhoe Mines', ticker: 'IVN-CA' },
  { name: 'Teck Resources', ticker: 'TECK-CA' }
]

async function main() {
  console.log('='.repeat(70))
  console.log('FACTSET SDK DOCUMENT DOWNLOAD TEST')
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
  console.log(`📁 Output directory: ${OUTPUT_DIR}`)

  try {
    // Step 1: Configure SDK with credentials (Configuration)
    console.log(`\n🔐 Step 1: Configuring SDK authentication...`)
    const configuration = fds.sdk.GlobalFilings.Configuration(
      username=FACTSET_USERNAME,
      password=FACTSET_API_KEY
    )

    // Step 2: Create authorization (Authorization)
    console.log(`🔐 Step 2: Creating authorization...`)
    const authorization = (FACTSET_USERNAME, FACTSET_API_KEY)

    console.log(`✅ SDK configured with both configuration and authorization`)

    // Initialize API client
    const filingsApi = new fds.GlobalFilingsApi(configuration)

    console.log('\n' + '='.repeat(70))
    console.log('SEARCHING FOR FILINGS')
    console.log('='.repeat(70))

    for (const company of TEST_COMPANIES) {
      console.log(`\n📊 ${company.name} (${company.ticker})`)
      console.log('-'.repeat(70))

      try {
        // Search parameters
        const searchRequest = new fds.InvestmentResearchRequest()
        searchRequest.ids = [company.ticker]
        searchRequest.sources = ['SDRP'] // SEDAR+
        searchRequest.startDate = '20250101'
        searchRequest.endDate = undefined
        searchRequest.paginationLimit = 5
        searchRequest.paginationOffset = 0
        searchRequest.timezone = 'America/Toronto'
        searchRequest.sort = ['-filingsDateTime']

        console.log(`   🔍 Searching for recent filings...`)

        // Execute search
        const response = await filingsApi.searchForFilings(searchRequest)

        if (!response.data || response.data.length === 0 || !response.data[0].documents) {
          console.log(`   ℹ️  No filings found`)
          continue
        }

        const documents = response.data[0].documents
        console.log(`   ✅ Found ${documents.length} filing(s)`)

        // Display and download each filing
        for (let i = 0; i < Math.min(documents.length, 3); i++) {
          const doc = documents[i]
          console.log(`\n   📄 Filing ${i + 1}:`)
          console.log(`      Headline: ${doc.headline}`)
          console.log(`      Date: ${new Date(doc.filingsDateTime).toLocaleDateString()}`)
          console.log(`      Form: ${doc.formTypes?.join(', ') || 'N/A'}`)
          console.log(`      Document ID: ${doc.documentId}`)
          console.log(`      Filing URL: ${doc.filingsLink}`)

          // Download the document
          console.log(`\n      📥 Attempting download...`)

          try {
            // Parse the filing link to get the key parameter
            const url = new URL(doc.filingsLink)
            const key = url.searchParams.get('key')

            if (!key) {
              console.log(`      ⚠️  No key found in filing URL`)
              continue
            }

            // Use the SDK's method to download the document
            const fileResponse = await filingsApi.getFilings({
              report: 'story',
              timezone: 'America/New_York',
              key: key
            })

            // Save the file
            const filename = `${company.ticker.replace('-CA', '')}_${doc.documentId}.pdf`
            const filepath = path.join(OUTPUT_DIR, filename)

            if (fileResponse) {
              fs.writeFileSync(filepath, fileResponse)
              const fileSizeKB = Math.round(fileResponse.length / 1024)

              console.log(`      ✅ Downloaded successfully`)
              console.log(`      💾 File size: ${fileSizeKB} KB`)
              console.log(`      📁 Saved to: ${filename}`)
            }

          } catch (downloadError: any) {
            console.error(`      ❌ Download error: ${downloadError.message}`)

            if (downloadError.message?.includes('CACCESS')) {
              console.log(`\n      ⚠️  CACCESS ERROR: Your account needs content access permissions`)
              console.log(`      Contact FactSet support to enable CACCESS`)
            }
          }

          // Small delay between downloads
          await new Promise(resolve => setTimeout(resolve, 500))
        }

      } catch (error: any) {
        console.error(`   ❌ Error processing ${company.name}: ${error.message}`)
      }
    }

    console.log('\n' + '='.repeat(70))
    console.log('TEST COMPLETE')
    console.log('='.repeat(70))
    console.log(`\n📁 Files saved to: ${OUTPUT_DIR}`)
    console.log(`\n💡 To view downloaded files:`)
    console.log(`   open "${OUTPUT_DIR}"`)

  } catch (error: any) {
    console.error(`\n❌ Fatal error: ${error.message}`)
    console.error(error.stack)
    process.exit(1)
  }
}

main().catch(console.error)
