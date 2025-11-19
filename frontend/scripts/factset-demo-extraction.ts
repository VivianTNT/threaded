#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'
import crypto from 'crypto'
import fs from 'fs'
import { promisify } from 'util'
import { pipeline } from 'stream'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const authHeader = 'Basic ' + Buffer.from(`${process.env.FACTSET_USERNAME}:${process.env.FACTSET_API_KEY}`).toString('base64')
const streamPipeline = promisify(pipeline)

const APPROX_KB_PER_PAGE = 30
const MIN_DOCUMENT_SIZE_MB = 3 // ~100 pages

// Target specific companies with known technical reports
const TARGET_COMPANIES = [
  { name: "Ivanhoe Mines Ltd.", ticker: "IVN-CA" },
  { name: "Lundin Mining Corporation", ticker: "LUN-CA" },
  { name: "First Quantum Minerals Ltd.", ticker: "FM-CA" },
  { name: "Alamos Gold Inc.", ticker: "AGI-US" },
]

async function searchFactSetDocs(ticker: string, companyName: string) {
  const searchUrl = `https://api.factset.com/content/global-filings/v2/search?ids=${ticker}&sources=SDR&searchText=technical report ni 43-101&sort=-filingsDate&paginationLimit=10`

  const response = await fetch(searchUrl, {
    headers: { 'Authorization': authHeader }
  })

  const data = await response.json()
  const documents = data.data?.[0]?.documents || []

  // Filter for substantial technical reports only
  const filtered = documents.filter((doc: any) => {
    const headline = doc.headline.toLowerCase()
    const isMainReport = (headline.includes('technical report') || headline.includes('43-101')) &&
                         !headline.includes('consent') &&
                         !headline.includes('certificate')
    return isMainReport
  })

  return filtered
}

async function downloadAndUploadPDF(doc: any, companyName: string, ticker: string): Promise<{ projectName: string, storageUrl: string } | null> {
  try {
    console.log(`\n📄 ${doc.headline.substring(0, 100)}...`)
    console.log(`   Date: ${doc.filingsDate}`)

    // Download PDF to temp file first
    const tempPath = `/tmp/factset_${crypto.randomUUID()}.pdf`

    const pdfResponse = await fetch(doc.filingsLink, {
      headers: { 'Authorization': authHeader, 'Accept': 'application/pdf' }
    })

    if (!pdfResponse.ok) {
      console.log(`   ❌ Download failed: ${pdfResponse.status}`)
      return null
    }

    // Save to temp file
    const fileStream = fs.createWriteStream(tempPath)
    await streamPipeline(pdfResponse.body, fileStream)

    // Check file size
    const stats = fs.statSync(tempPath)
    const sizeMB = stats.size / 1024 / 1024
    const estimatedPages = Math.floor((stats.size / 1024) / APPROX_KB_PER_PAGE)

    console.log(`   📦 Downloaded: ${sizeMB.toFixed(2)} MB (~${estimatedPages} pages)`)

    if (sizeMB < MIN_DOCUMENT_SIZE_MB) {
      console.log(`   ⚠️  Too small, skipping`)
      fs.unlinkSync(tempPath)
      return null
    }

    // Upload to Supabase storage using file path
    const hash = crypto.createHash('md5').update(doc.documentId).digest('hex')
    const storagePath = `${ticker}/${hash}.pdf`

    console.log(`   📤 Uploading to Supabase storage...`)

    const fileBuffer = fs.readFileSync(tempPath)

    const { data: uploadData, error: uploadError } = await supabase.storage
      .from('factset-documents')
      .upload(storagePath, fileBuffer, {
        contentType: 'application/pdf',
        upsert: true
      })

    // Clean up temp file
    fs.unlinkSync(tempPath)

    if (uploadError) {
      console.log(`   ❌ Upload failed: ${uploadError.message}`)
      return null
    }

    // Get public URL
    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(storagePath)

    console.log(`   ✅ Uploaded successfully!`)
    console.log(`   🔗 Storage URL: ${urlData.publicUrl}`)

    // Extract project name from headline
    const projectName = doc.headline
      .replace(/technical report/gi, '')
      .replace(/ni 43-101/gi, '')
      .replace(/\(.*?\)/g, '')
      .replace(companyName, '')
      .replace(/-/g, '')
      .trim()
      .substring(0, 100) || `${companyName} Project`

    return {
      projectName: projectName,
      storageUrl: urlData.publicUrl
    }

  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`)
    return null
  }
}

async function main() {
  console.log('🎯 FACTSET MINING PROJECT EXTRACTION DEMO')
  console.log('=' .repeat(80))
  console.log('Extracting technical reports (100+ pages) from FactSet API')
  console.log('Uploading PDFs to Supabase Storage\n')

  const results: Array<{ company: string, projectName: string, storageUrl: string }> = []

  for (const company of TARGET_COMPANIES) {
    console.log(`\n${'='.repeat(80)}`)
    console.log(`📊 ${company.name} (${company.ticker})`)
    console.log('='.repeat(80))

    try {
      const docs = await searchFactSetDocs(company.ticker, company.name)
      console.log(`   Found ${docs.length} potential technical reports`)

      if (docs.length === 0) {
        console.log(`   ⚠️  No documents found`)
        continue
      }

      // Try first 3 documents
      let successCount = 0
      for (const doc of docs.slice(0, 3)) {
        if (successCount >= 2) break // Max 2 per company for demo

        const result = await downloadAndUploadPDF(doc, company.name, company.ticker)

        if (result) {
          results.push({
            company: company.name,
            projectName: result.projectName,
            storageUrl: result.storageUrl
          })
          successCount++
        }
      }

    } catch (error: any) {
      console.error(`   ❌ Error processing ${company.name}: ${error.message}`)
    }
  }

  // Summary
  console.log('\n\n' + '='.repeat(80))
  console.log('✅ EXTRACTION COMPLETE')
  console.log('='.repeat(80))
  console.log(`\nSuccessfully extracted ${results.length} projects:\n`)

  results.forEach((r, i) => {
    console.log(`${i + 1}. ${r.projectName}`)
    console.log(`   Company: ${r.company}`)
    console.log(`   📄 Supabase Storage URL: ${r.storageUrl}`)
    console.log('')
  })

  if (results.length === 0) {
    console.log('⚠️  No projects extracted. This may be due to:')
    console.log('   1. Supabase storage upload failures (connection/timeout)')
    console.log('   2. No documents matching size criteria (100+ pages)')
    console.log('   3. FactSet API rate limits')
  }
}

main().catch(console.error)
