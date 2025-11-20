#!/usr/bin/env node

/**
 * FactSet to Supabase - Complete Demo
 *
 * Shows three approaches to storing FactSet documents:
 * 1. Store full PDF in Supabase Storage
 * 2. Store only metadata + FactSet link (with caveats)
 * 3. Extract data, store structured info, optionally keep PDF
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)
const BASE_URL = 'https://api.factset.com/content/global-filings/v2'

interface Filing {
  headline: string
  filingsDateTime: string
  formTypes?: string[]
  documentId: string
  filingsLink: string
  filingSize?: string
}

function createAuthHeader(): string {
  return 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
}

async function searchFilings(ticker: string, limit: number = 5): Promise<Filing[]> {
  const params = new URLSearchParams({
    ids: ticker,
    sources: 'SDR,SDRP',
    startDate: '20240101',
    _paginationLimit: String(limit),
    timeZone: 'America/Toronto',
    _sort: '-filingsDateTime'
  })

  const response = await fetch(`${BASE_URL}/search?${params.toString()}`, {
    headers: {
      'Authorization': createAuthHeader(),
      'Accept': 'application/json'
    }
  })

  if (!response.ok) throw new Error(`Search failed: ${await response.text()}`)

  const data = await response.json()
  return data.data?.[0]?.documents || []
}

async function downloadDocument(filing: Filing): Promise<Buffer | null> {
  try {
    const response = await fetch(filing.filingsLink, {
      headers: {
        'Authorization': createAuthHeader(),
        'Accept': '*/*'
      }
    })

    if (!response.ok) return null

    return Buffer.from(await response.arrayBuffer())
  } catch (error) {
    console.error(`Download error:`, error)
    return null
  }
}

// ============================================================================
// APPROACH 1: Store Full PDF in Supabase Storage
// ============================================================================

async function approach1_UploadToStorage(filing: Filing, pdfBuffer: Buffer) {
  console.log(`\n📦 Approach 1: Upload PDF to Supabase Storage`)

  const filename = `factset/${filing.documentId}.pdf`

  // Upload to Supabase Storage
  const { data: uploadData, error: uploadError } = await supabase.storage
    .from('documents') // Create this bucket in Supabase dashboard
    .upload(filename, pdfBuffer, {
      contentType: 'application/pdf',
      upsert: true
    })

  if (uploadError) {
    console.error(`   ❌ Upload failed:`, uploadError.message)
    return null
  }

  // Get public URL
  const { data: urlData } = supabase.storage
    .from('documents')
    .getPublicUrl(filename)

  console.log(`   ✅ Uploaded: ${filename}`)
  console.log(`   🔗 URL: ${urlData.publicUrl}`)

  // Store metadata in database
  const { error: dbError } = await supabase
    .from('factset_documents')
    .upsert({
      document_id: filing.documentId,
      headline: filing.headline,
      filing_date: filing.filingsDateTime,
      form_type: filing.formTypes?.[0],
      storage_path: filename,
      public_url: urlData.publicUrl,
      file_size: pdfBuffer.length
    }, {
      onConflict: 'document_id'
    })

  if (dbError) {
    console.error(`   ⚠️  Database insert failed:`, dbError.message)
  }

  return urlData.publicUrl
}

// ============================================================================
// APPROACH 2: Store Only Metadata + FactSet Link
// ============================================================================

async function approach2_StoreMetadataOnly(filing: Filing) {
  console.log(`\n🔗 Approach 2: Store Metadata + FactSet Link`)

  const { error } = await supabase
    .from('factset_documents')
    .upsert({
      document_id: filing.documentId,
      headline: filing.headline,
      filing_date: filing.filingsDateTime,
      form_type: filing.formTypes?.[0],
      factset_link: filing.filingsLink, // Store the encrypted link
      storage_path: null,
      public_url: null
    }, {
      onConflict: 'document_id'
    })

  if (error) {
    console.error(`   ❌ Database insert failed:`, error.message)
    return false
  }

  console.log(`   ✅ Stored metadata only`)
  console.log(`   ⚠️  WARNING: FactSet link may expire!`)
  console.log(`   🔗 Link: ${filing.filingsLink.substring(0, 80)}...`)

  return true
}

// ============================================================================
// APPROACH 3: Extract Data, Store Structured Info
// ============================================================================

async function approach3_ExtractAndStore(filing: Filing, pdfBuffer: Buffer) {
  console.log(`\n🤖 Approach 3: Extract Data with OpenAI (Simulated)`)

  // In reality, you'd use OpenAI to parse the PDF
  // For demo purposes, we'll extract basic info from the headline
  const extractedData = {
    document_id: filing.documentId,
    headline: filing.headline,
    filing_date: filing.filingsDateTime,
    form_type: filing.formTypes?.[0],

    // Simulated extraction (would come from OpenAI in production)
    project_name: extractProjectName(filing.headline),
    document_type: filing.formTypes?.[0] || 'Unknown',
    has_technical_report: filing.headline.toLowerCase().includes('43-101'),
    has_financial_data: filing.headline.toLowerCase().includes('quarter') ||
                        filing.headline.toLowerCase().includes('annual'),

    // Metadata
    extracted_at: new Date().toISOString(),
    extraction_method: 'openai-gpt4',
    pdf_size: pdfBuffer.length
  }

  const { error } = await supabase
    .from('factset_extracted_data')
    .upsert(extractedData, {
      onConflict: 'document_id'
    })

  if (error) {
    console.error(`   ❌ Database insert failed:`, error.message)
    return null
  }

  console.log(`   ✅ Extracted and stored data`)
  console.log(`   📊 Project: ${extractedData.project_name || 'Not found'}`)
  console.log(`   📄 Type: ${extractedData.document_type}`)
  console.log(`   💰 Has financials: ${extractedData.has_financial_data ? 'Yes' : 'No'}`)

  return extractedData
}

function extractProjectName(headline: string): string | null {
  // Simple regex to extract project names (you'd use OpenAI in production)
  const patterns = [
    /([A-Z][a-z]+-[A-Z][a-z]+)\s+(Project|Mine|Property)/i,
    /(?:for|at)\s+(?:the\s+)?([A-Z][A-Za-z\s]+(?:Project|Mine|Property))/i
  ]

  for (const pattern of patterns) {
    const match = headline.match(pattern)
    if (match) return match[1]
  }

  return null
}

// ============================================================================
// Demo All Three Approaches
// ============================================================================

async function main() {
  console.log('='.repeat(70))
  console.log('FACTSET TO SUPABASE - STORAGE OPTIONS DEMO')
  console.log('='.repeat(70))

  const testCompany = { name: 'Ivanhoe Mines', ticker: 'IVN-CA' }

  console.log(`\n📊 Searching for: ${testCompany.name} (${testCompany.ticker})`)

  // Search for one filing
  const filings = await searchFilings(testCompany.ticker, 1)

  if (filings.length === 0) {
    console.log('❌ No filings found')
    return
  }

  const filing = filings[0]
  console.log(`\n✅ Found filing:`)
  console.log(`   Title: ${filing.headline}`)
  console.log(`   Date: ${filing.filingsDateTime}`)
  console.log(`   ID: ${filing.documentId}`)

  // Download the document
  console.log(`\n📥 Downloading document...`)
  const pdfBuffer = await downloadDocument(filing)

  if (!pdfBuffer) {
    console.log('❌ Download failed')
    return
  }

  console.log(`✅ Downloaded ${Math.round(pdfBuffer.length / 1024)} KB`)

  // Demonstrate all three approaches
  console.log('\n' + '='.repeat(70))
  console.log('DEMONSTRATING STORAGE APPROACHES')
  console.log('='.repeat(70))

  // Approach 1: Full storage
  try {
    await approach1_UploadToStorage(filing, pdfBuffer)
  } catch (error: any) {
    console.error(`   ❌ Approach 1 failed:`, error.message)
  }

  // Approach 2: Metadata only
  try {
    await approach2_StoreMetadataOnly(filing)
  } catch (error: any) {
    console.error(`   ❌ Approach 2 failed:`, error.message)
  }

  // Approach 3: Extract and store
  try {
    await approach3_ExtractAndStore(filing, pdfBuffer)
  } catch (error: any) {
    console.error(`   ❌ Approach 3 failed:`, error.message)
  }

  // Summary
  console.log('\n' + '='.repeat(70))
  console.log('SUMMARY & RECOMMENDATIONS')
  console.log('='.repeat(70))

  console.log(`
📊 COMPARISON:

Approach 1: Upload Full PDF to Supabase Storage
  ✅ Pros: Always accessible, no API dependency
  ❌ Cons: Storage costs (~$0.021/GB/month)
  💡 Use for: Important documents (NI 43-101 reports)

Approach 2: Store Only Metadata + Link
  ✅ Pros: No storage costs
  ❌ Cons: Links may expire, requires re-auth
  💡 Use for: Not recommended (links are temporary)

Approach 3: Extract Data, Optionally Keep PDF
  ✅ Pros: Structured data, fast queries
  ❌ Cons: Extraction takes time/money (OpenAI)
  💡 Use for: Your main workflow (recommended)

RECOMMENDED HYBRID APPROACH:
1. Download PDF from FactSet
2. Extract structured data with OpenAI → Store in 'projects' table
3. Upload PDF to Supabase Storage → Keep for reference
4. Store metadata linking the two

STORAGE COSTS (estimates):
- Average filing: 500 KB
- 1,000 filings: 500 MB = ~$0.01/month
- 10,000 filings: 5 GB = ~$0.10/month
→ Very affordable!
`)

  console.log('='.repeat(70))
}

main().catch(console.error)
