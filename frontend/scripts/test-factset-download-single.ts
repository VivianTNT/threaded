#!/usr/bin/env npx tsx

/**
 * Test FactSet document download with a single company
 * to verify the API approach before running exhaustively
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import * as crypto from 'crypto'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

async function testSingleCompany() {
  console.log('🧪 Testing FactSet Document Download\n')
  console.log('Credentials:', FACTSET_USERNAME)
  console.log('Auth header:', authHeader.substring(0, 30) + '...\n')

  // Test with Barrick Gold (BHP-US ticker)
  const testTicker = 'GOLD-US' // Barrick Gold on NYSE
  const testCompanyName = 'Barrick Gold Corporation'

  console.log(`Testing with: ${testCompanyName} (${testTicker})\n`)

  // Step 1: Search for documents
  console.log('Step 1: Searching for documents...')

  const searchParams = new URLSearchParams({
    ids: testTicker,
    sources: 'EDG', // EDGAR for US companies
    startDate: '20220101',
    endDate: '20250201',
    searchText: 'mining',
    _paginationLimit: '5'
  })

  const searchUrl = `https://api.factset.com/content/global-filings/v2/search?${searchParams.toString()}`

  console.log('Search URL:', searchUrl, '\n')

  const searchResponse = await fetch(searchUrl, {
    headers: {
      'Authorization': authHeader,
      'Accept': 'application/json'
    }
  })

  if (!searchResponse.ok) {
    console.error('❌ Search failed:', searchResponse.status, searchResponse.statusText)
    const errorText = await searchResponse.text()
    console.error('Error details:', errorText)
    return
  }

  const searchData = await searchResponse.json()
  console.log('✅ Search successful!')
  console.log(JSON.stringify(searchData, null, 2))

  // Step 2: Try to download first document
  if (searchData.data && searchData.data[0]?.documents?.[0]) {
    const firstDoc = searchData.data[0].documents[0]
    console.log('\n\nStep 2: Attempting to download first document...')
    console.log('Document ID:', firstDoc.documentId)
    console.log('Headline:', firstDoc.headline)

    // Try different download approaches
    const downloadUrls = [
      `https://api.factset.com/content/documents-distributor/v1/single-document?id=${firstDoc.documentId}`,
      `https://api.factset.com/content/documents-distributor/v1/documents/${firstDoc.documentId}`,
      firstDoc.filingsLink // Direct link from search results
    ]

    for (const downloadUrl of downloadUrls) {
      console.log(`\n   Trying: ${downloadUrl}`)

      try {
        const downloadResponse = await fetch(downloadUrl, {
          headers: {
            'Authorization': authHeader,
            'Accept': 'application/pdf'
          }
        })

        console.log(`   Status: ${downloadResponse.status} ${downloadResponse.statusText}`)

        if (downloadResponse.ok) {
          const buffer = Buffer.from(await downloadResponse.arrayBuffer())
          console.log(`   ✅ SUCCESS! Downloaded ${(buffer.length / 1024).toFixed(1)} KB`)

          // Test upload to storage
          const hash = crypto.createHash('md5').update(firstDoc.documentId).digest('hex')
          const storagePath = `test/${hash}.pdf`

          const { error: uploadError } = await supabase.storage
            .from('factset-documents')
            .upload(storagePath, buffer, {
              contentType: 'application/pdf',
              upsert: true
            })

          if (uploadError) {
            console.log(`   ❌ Upload failed: ${uploadError.message}`)
          } else {
            console.log(`   ✅ Uploaded to storage: ${storagePath}`)

            const { data: urlData } = supabase.storage
              .from('factset-documents')
              .getPublicUrl(storagePath)

            console.log(`   Public URL: ${urlData.publicUrl}`)
          }

          return
        } else {
          const errorText = await downloadResponse.text()
          console.log(`   ❌ Failed: ${errorText.substring(0, 200)}`)
        }
      } catch (error: any) {
        console.log(`   ❌ Error: ${error.message}`)
      }
    }
  } else {
    console.log('\n❌ No documents found in search results')
  }
}

testSingleCompany().catch(console.error)
