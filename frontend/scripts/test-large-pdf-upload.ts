#!/usr/bin/env npx tsx

/**
 * Test Large PDF Upload Strategies
 *
 * This script tests different approaches to upload 20-25 MB PDFs to Supabase Storage
 */

import { createClient } from '@supabase/supabase-js'
import * as https from 'https'
import * as http from 'http'
import * as fs from 'fs'
import * as path from 'path'
import * as crypto from 'crypto'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Test with a known large document from FactSet
const TEST_DOCUMENT_ID = '0022320240515067709' // Example large document

/**
 * Strategy 1: Direct buffer upload with extended timeout
 */
async function strategy1_bufferWithTimeout(buffer: Buffer, filename: string): Promise<boolean> {
  console.log('\n🧪 Strategy 1: Direct buffer upload with extended timeout')

  try {
    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(`test/${filename}`, buffer, {
        contentType: 'application/pdf',
        upsert: true
      })

    if (error) {
      console.log(`   ❌ Failed: ${error.message}`)
      return false
    }

    console.log(`   ✅ Success: ${data.path}`)
    return true
  } catch (err: any) {
    console.log(`   ❌ Exception: ${err.message}`)
    return false
  }
}

/**
 * Strategy 2: Upload via signed URL (bypasses some SDK limitations)
 */
async function strategy2_signedUrl(buffer: Buffer, filename: string): Promise<boolean> {
  console.log('\n🧪 Strategy 2: Upload via signed URL')

  try {
    // Create a signed upload URL
    const { data: urlData, error: urlError } = await supabase.storage
      .from('factset-documents')
      .createSignedUploadUrl(`test/${filename}`)

    if (urlError || !urlData) {
      console.log(`   ❌ Failed to create signed URL: ${urlError?.message}`)
      return false
    }

    // Upload directly to the signed URL
    const uploadUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/upload/sign/factset-documents/${urlData.path}?token=${urlData.token}`

    const response = await fetch(uploadUrl, {
      method: 'PUT',
      body: buffer,
      headers: {
        'Content-Type': 'application/pdf',
        'x-upsert': 'true'
      }
    })

    if (!response.ok) {
      console.log(`   ❌ Upload failed: ${response.status} ${response.statusText}`)
      return false
    }

    console.log(`   ✅ Success via signed URL`)
    return true
  } catch (err: any) {
    console.log(`   ❌ Exception: ${err.message}`)
    return false
  }
}

/**
 * Strategy 3: Stream upload from file (memory efficient)
 */
async function strategy3_streamFromFile(filepath: string, filename: string): Promise<boolean> {
  console.log('\n🧪 Strategy 3: Stream upload from file')

  try {
    const fileStream = fs.createReadStream(filepath)
    const stats = fs.statSync(filepath)

    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(`test/${filename}`, fileStream as any, {
        contentType: 'application/pdf',
        upsert: true
      })

    if (error) {
      console.log(`   ❌ Failed: ${error.message}`)
      return false
    }

    console.log(`   ✅ Success: ${data.path} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`)
    return true
  } catch (err: any) {
    console.log(`   ❌ Exception: ${err.message}`)
    return false
  }
}

/**
 * Strategy 4: Direct HTTP API with custom timeout
 */
async function strategy4_directHttpApi(buffer: Buffer, filename: string): Promise<boolean> {
  console.log('\n🧪 Strategy 4: Direct HTTP API with custom timeout')

  return new Promise((resolve) => {
    try {
      const url = new URL(`${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/factset-documents/test/${filename}`)

      const options = {
        hostname: url.hostname,
        path: url.pathname,
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${process.env.SUPABASE_SERVICE_ROLE_KEY}`,
          'Content-Type': 'application/pdf',
          'Content-Length': buffer.length,
          'x-upsert': 'true'
        },
        timeout: 600000 // 10 minutes
      }

      const req = https.request(options, (res) => {
        let responseData = ''

        res.on('data', (chunk) => {
          responseData += chunk
        })

        res.on('end', () => {
          if (res.statusCode === 200 || res.statusCode === 201) {
            console.log(`   ✅ Success: ${res.statusCode}`)
            resolve(true)
          } else {
            console.log(`   ❌ Failed: ${res.statusCode} - ${responseData}`)
            resolve(false)
          }
        })
      })

      req.on('error', (err) => {
        console.log(`   ❌ Request error: ${err.message}`)
        resolve(false)
      })

      req.on('timeout', () => {
        console.log(`   ❌ Request timeout`)
        req.destroy()
        resolve(false)
      })

      req.write(buffer)
      req.end()
    } catch (err: any) {
      console.log(`   ❌ Exception: ${err.message}`)
      resolve(false)
    }
  })
}

/**
 * Download PDF from FactSet
 */
async function downloadPDF(documentId: string): Promise<{ buffer: Buffer; filepath: string } | null> {
  try {
    console.log(`\n📥 Downloading PDF from FactSet (ID: ${documentId})...`)

    const url = `https://api.factset.com/content/documents-distributor/v1/single-document?id=${documentId}`

    // Download to buffer
    const response = await fetch(url, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf'
      }
    })

    if (!response.ok) {
      console.log(`   ❌ Download failed: ${response.status}`)
      return null
    }

    const buffer = Buffer.from(await response.arrayBuffer())
    console.log(`   ✅ Downloaded: ${(buffer.length / 1024 / 1024).toFixed(2)} MB`)

    // Also save to temp file for stream testing
    const tmpDir = path.join(__dirname, '../tmp')
    if (!fs.existsSync(tmpDir)) {
      fs.mkdirSync(tmpDir, { recursive: true })
    }

    const filepath = path.join(tmpDir, `test-${documentId}.pdf`)
    fs.writeFileSync(filepath, buffer)
    console.log(`   💾 Saved to: ${filepath}`)

    return { buffer, filepath }
  } catch (error: any) {
    console.log(`   ❌ Download error: ${error.message}`)
    return null
  }
}

/**
 * Search for large documents
 */
async function findLargeDocument(): Promise<string | null> {
  console.log('\n🔍 Searching for large technical documents...')

  const tickers = ['BHP-US', 'RIO-GB', 'VALE-US', 'FCX-US', 'SCCO-US']

  for (const ticker of tickers) {
    try {
      const params = new URLSearchParams({
        ids: ticker,
        sources: 'SDR',
        startDate: '20220101',
        endDate: '20250131',
        searchText: 'NI 43-101',
        _paginationLimit: '10'
      })

      const url = `https://api.factset.com/content/global-filings/v2/search?${params.toString()}`

      const response = await fetch(url, {
        headers: {
          'Authorization': authHeader,
          'Accept': 'application/json'
        }
      })

      if (response.ok) {
        const data = await response.json()

        if (data.data && Array.isArray(data.data)) {
          for (const result of data.data) {
            if (result.documents && Array.isArray(result.documents)) {
              for (const doc of result.documents) {
                // Look for large files (>10MB)
                const sizeMatch = doc.filingSize?.match(/(\d+\.?\d*)\s*MB/i)
                if (sizeMatch && parseFloat(sizeMatch[1]) > 10) {
                  console.log(`   ✅ Found large document: ${doc.headline}`)
                  console.log(`      Size: ${doc.filingSize}`)
                  console.log(`      ID: ${doc.documentId}`)
                  return doc.documentId
                }
              }
            }
          }
        }
      }
    } catch (error: any) {
      console.log(`   ⚠️  Error searching ${ticker}: ${error.message}`)
    }
  }

  console.log(`   ℹ️  No large documents found, using test data`)
  return null
}

async function main() {
  console.log('🧪 TESTING LARGE PDF UPLOAD STRATEGIES FOR SUPABASE')
  console.log('='.repeat(80))

  // Try to find a large document or use test data
  let documentId = await findLargeDocument()

  if (!documentId) {
    console.log('\n⚠️  No real large documents available. Creating test file...')

    // Create a ~20MB test file
    const tmpDir = path.join(__dirname, '../tmp')
    if (!fs.existsSync(tmpDir)) {
      fs.mkdirSync(tmpDir, { recursive: true })
    }

    const testFilepath = path.join(tmpDir, 'test-20mb.pdf')
    const buffer = Buffer.alloc(20 * 1024 * 1024) // 20 MB
    fs.writeFileSync(testFilepath, buffer)

    console.log(`   ✅ Created 20MB test file: ${testFilepath}`)

    const filename = `test-20mb-${Date.now()}.pdf`

    // Test all strategies with test file
    const results = {
      strategy1: await strategy1_bufferWithTimeout(buffer, `s1-${filename}`),
      strategy2: await strategy2_signedUrl(buffer, `s2-${filename}`),
      strategy3: await strategy3_streamFromFile(testFilepath, `s3-${filename}`),
      strategy4: await strategy4_directHttpApi(buffer, `s4-${filename}`)
    }

    console.log('\n' + '='.repeat(80))
    console.log('📊 RESULTS SUMMARY')
    console.log('='.repeat(80))
    console.log(`Strategy 1 (Buffer + Timeout):    ${results.strategy1 ? '✅ SUCCESS' : '❌ FAILED'}`)
    console.log(`Strategy 2 (Signed URL):          ${results.strategy2 ? '✅ SUCCESS' : '❌ FAILED'}`)
    console.log(`Strategy 3 (File Stream):         ${results.strategy3 ? '✅ SUCCESS' : '❌ FAILED'}`)
    console.log(`Strategy 4 (Direct HTTP API):     ${results.strategy4 ? '✅ SUCCESS' : '❌ FAILED'}`)
    console.log('='.repeat(80))

    const successfulStrategies = Object.entries(results)
      .filter(([_, success]) => success)
      .map(([strategy, _]) => strategy)

    if (successfulStrategies.length > 0) {
      console.log(`\n✅ Recommended strategy: ${successfulStrategies[0]}`)
    } else {
      console.log('\n❌ All strategies failed. Possible issues:')
      console.log('   1. Supabase storage bucket size limit reached')
      console.log('   2. Network timeout (check firewall/proxy settings)')
      console.log('   3. Supabase plan limitations on file size')
      console.log('   4. Service role key permissions')
    }

    // Clean up test file
    fs.unlinkSync(testFilepath)

    return
  }

  // Download the real PDF
  const download = await downloadPDF(documentId)

  if (!download) {
    console.log('❌ Failed to download PDF. Exiting.')
    return
  }

  const { buffer, filepath } = download
  const filename = `large-test-${crypto.randomUUID()}.pdf`

  // Test all strategies
  const results = {
    strategy1: await strategy1_bufferWithTimeout(buffer, `s1-${filename}`),
    strategy2: await strategy2_signedUrl(buffer, `s2-${filename}`),
    strategy3: await strategy3_streamFromFile(filepath, `s3-${filename}`),
    strategy4: await strategy4_directHttpApi(buffer, `s4-${filename}`)
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 RESULTS SUMMARY')
  console.log('='.repeat(80))
  console.log(`Strategy 1 (Buffer + Timeout):    ${results.strategy1 ? '✅ SUCCESS' : '❌ FAILED'}`)
  console.log(`Strategy 2 (Signed URL):          ${results.strategy2 ? '✅ SUCCESS' : '❌ FAILED'}`)
  console.log(`Strategy 3 (File Stream):         ${results.strategy3 ? '✅ SUCCESS' : '❌ FAILED'}`)
  console.log(`Strategy 4 (Direct HTTP API):     ${results.strategy4 ? '✅ SUCCESS' : '❌ FAILED'}`)
  console.log('='.repeat(80))

  const successfulStrategies = Object.entries(results)
    .filter(([_, success]) => success)
    .map(([strategy, _]) => strategy)

  if (successfulStrategies.length > 0) {
    console.log(`\n✅ Recommended strategy for 20-25MB PDFs: ${successfulStrategies[0]}`)
  } else {
    console.log('\n❌ All strategies failed. Checking Supabase storage configuration...')

    // Check storage bucket info
    const { data: buckets, error: bucketError } = await supabase.storage.listBuckets()

    if (bucketError) {
      console.log(`   ❌ Cannot access buckets: ${bucketError.message}`)
    } else {
      const factsetBucket = buckets?.find(b => b.name === 'factset-documents')
      if (factsetBucket) {
        console.log(`   ℹ️  Bucket info:`)
        console.log(`      Name: ${factsetBucket.name}`)
        console.log(`      Public: ${factsetBucket.public}`)
        console.log(`      File size limit: ${(factsetBucket.file_size_limit || 0) / 1024 / 1024} MB`)
      }
    }
  }

  // Clean up temp file
  if (fs.existsSync(filepath)) {
    fs.unlinkSync(filepath)
  }
}

main().catch(console.error)
