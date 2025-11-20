#!/usr/bin/env node

/**
 * FactSet SDK Document Download Test (v2)
 *
 * Uses the official @factset/sdk-globalfilings SDK to search for and
 * download filing documents from FactSet.
 */

import { config } from 'dotenv'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

// Import FactSet SDK
const { ApiClient, FilingsAPIApi } = require('@factset/sdk-globalfilings')

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
  console.log('FACTSET SDK DOCUMENT DOWNLOAD TEST (V2)')
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

  // Configure API Client
  console.log(`\n🔐 Configuring FactSet API client...`)
  const apiClient = ApiClient.instance

  // Use Basic authentication
  const FactSetApiKey = apiClient.authentications['FactSetApiKey']
  FactSetApiKey.username = FACTSET_USERNAME
  FactSetApiKey.password = FACTSET_API_KEY

  console.log(`✅ API client configured`)

  // Create API instance
  const apiInstance = new FilingsAPIApi()

  console.log('\n' + '='.repeat(70))
  console.log('SEARCHING FOR FILINGS')
  console.log('='.repeat(70))

  let totalFilings = 0
  let successfulDownloads = 0

  for (const company of TEST_COMPANIES) {
    console.log(`\n📊 ${company.name} (${company.ticker})`)
    console.log('-'.repeat(70))

    try {
      const opts = {
        'startDate': '20240101', // Expanded date range
        'timeZone': 'America/Toronto',
        'paginationLimit': 5,
        'paginationOffset': 0,
        '_sort': ['-filingsDateTime']
      }

      console.log(`   🔍 Searching for filings...`)

      // Call the getFilings API endpoint
      const response = await new Promise((resolve, reject) => {
        apiInstance.getFilings(
          [company.ticker], // ids
          ['SDRP'], // sources (SEDAR+)
          opts,
          (error: any, data: any, response: any) => {
            if (error) {
              console.error(`   ❌ API Error:`, error)
              reject(error)
            } else {
              resolve(data)
            }
          }
        )
      }).catch((err: any) => {
        console.error(`   ❌ Promise rejected:`, err)
        throw err
      })

      console.log(`   📦 Response received (first 500 chars):`, JSON.stringify(response).substring(0, 500))

      // Parse response
      const responseData = response as any
      if (!responseData.data || !Array.isArray(responseData.data) || responseData.data.length === 0) {
        console.log(`   ℹ️  No filings found`)
        continue
      }

      const documents = responseData.data[0]?.documents || []
      if (documents.length === 0) {
        console.log(`   ℹ️  No documents in response`)
        continue
      }

      totalFilings += documents.length
      console.log(`   ✅ Found ${documents.length} filing(s)`)

      // Display first few documents
      for (let i = 0; i < Math.min(documents.length, 3); i++) {
        const doc = documents[i]
        console.log(`\n   📄 Filing ${i + 1}:`)
        console.log(`      Headline: ${doc.headline}`)
        console.log(`      Date: ${doc.filingsDateTime}`)
        console.log(`      Form: ${doc.formTypes?.join(', ') || 'N/A'}`)
        console.log(`      Document ID: ${doc.documentId}`)
        console.log(`      Filing Link: ${doc.filingsLink}`)

        // Extract the encrypted key from the filing link
        if (doc.filingsLink) {
          const url = new URL(doc.filingsLink)
          const key = url.searchParams.get('key')

          if (key) {
            console.log(`      🔑 Extracted key: ${key.substring(0, 20)}...`)

            // Try to download using the filing link directly
            console.log(`\n      📥 Attempting to download document...`)

            try {
              const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

              const downloadResponse = await fetch(doc.filingsLink, {
                method: 'GET',
                headers: {
                  'Authorization': authHeader,
                  'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml,*/*',
                  'User-Agent': 'Lithos-AI/1.0'
                }
              })

              console.log(`      📊 Download Response: ${downloadResponse.status} ${downloadResponse.statusText}`)

              if (!downloadResponse.ok) {
                const errorText = await downloadResponse.text()
                console.error(`      ❌ Download failed: ${errorText}`)

                if (errorText.includes('CACCESS')) {
                  console.log(`\n      ⚠️  CACCESS Permission Required`)
                  console.log(`      Your account needs content access permissions.`)
                  console.log(`      Contact FactSet support at misa.kobayashi@factset.com`)
                }
              } else {
                // Success! Save the file
                const buffer = await downloadResponse.arrayBuffer()
                const contentType = downloadResponse.headers.get('content-type') || ''

                let fileExtension = '.pdf'
                if (contentType.includes('html')) fileExtension = '.html'
                else if (contentType.includes('xml')) fileExtension = '.xml'

                const filename = `${company.ticker.replace('-CA', '')}_${doc.documentId}${fileExtension}`
                const filepath = path.join(OUTPUT_DIR, filename)

                fs.writeFileSync(filepath, Buffer.from(buffer))

                const fileSizeKB = Math.round(buffer.byteLength / 1024)
                console.log(`      ✅ Downloaded successfully!`)
                console.log(`      💾 File size: ${fileSizeKB} KB`)
                console.log(`      📁 Saved to: ${filename}`)

                successfulDownloads++
              }

            } catch (downloadError: any) {
              console.error(`      ❌ Download error: ${downloadError.message}`)
            }
          }
        }

        // Delay between downloads
        await new Promise(resolve => setTimeout(resolve, 500))
      }

    } catch (error: any) {
      console.error(`   ❌ Error: ${error.message}`)
      if (error.response) {
        console.error(`   📊 Response Status: ${error.response.status}`)
        console.error(`   📋 Response Body:`, error.response.body)
      }
    }
  }

  console.log('\n' + '='.repeat(70))
  console.log('TEST COMPLETE')
  console.log('='.repeat(70))
  console.log(`\n📊 Summary:`)
  console.log(`   Total filings found: ${totalFilings}`)
  console.log(`   Successful downloads: ${successfulDownloads}`)
  console.log(`\n📁 Files saved to: ${OUTPUT_DIR}`)

  if (successfulDownloads > 0) {
    console.log(`\n✅ Success! Documents downloaded.`)
    console.log(`\n💡 To view files:`)
    console.log(`   open "${OUTPUT_DIR}"`)
  } else {
    console.log(`\n⚠️  No documents were downloaded.`)
    console.log(`\nIf you see CACCESS errors, contact FactSet to enable content access.`)
  }
}

main().catch(error => {
  console.error('\n❌ Fatal error:', error)
  process.exit(1)
})
