#!/usr/bin/env npx tsx

/**
 * EXHAUSTIVE FactSet Mining Project Population
 *
 * Processes all 123 mining companies from database:
 * 1. Search FactSet for NI 43-101, technical reports, feasibility studies (2018-2025)
 * 2. Download PDFs from FactSet Documents Distributor API
 * 3. Upload to Supabase Storage (factset-documents bucket)
 * 4. Create project records with name and document_storage_path
 *
 * Requirements:
 * - FACTSET_USERNAME and FACTSET_API_KEY environment variables
 * - Supabase credentials in .env.local
 *
 * Run with: npm run populate:projects
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import * as crypto from 'crypto'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

// ============================
// Configuration
// ============================

const FACTSET_USERNAME = process.env.FACTSET_USERNAME
const FACTSET_API_KEY = process.env.FACTSET_API_KEY
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

// FactSet API configuration
const FACTSET_BASE_URL = 'https://api.factset.com/content/global-filings/v2'
const DOCS_DISTRIBUTOR_URL = 'https://api.factset.com/content/documents-distributor/v1'
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

const DELAY_MS = 1500 // Rate limiting
const MIN_DOCUMENT_SIZE_KB = 50 // Only documents >= 50 KB
const MIN_PAGES_ESTIMATE = 100 // Only substantial technical reports (100+ pages)
const APPROX_KB_PER_PAGE = 30 // Conservative estimate: 30 KB per page for technical PDFs with images
const MIN_DOCUMENT_SIZE_MB = 3 // ~100 pages = ~3 MB minimum
const MAX_DOCUMENT_SIZE_MB = 50 // Supabase supports up to 50 MB
const MAX_DOCS_PER_COMPANY = 15 // Limit per company

// Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! })

// ============================
// Type Definitions
// ============================

interface Company {
  id: string
  name: string
  ticker: string | null
  exchange: string | null
  country: string | null
}

interface FactSetDocument {
  documentId: string
  headline: string
  filingDate: string
  formType: string
  filingSize: string
  filingsLink: string
  source: string
}

function delay(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// ============================
// Utility Functions
// ============================

/**
 * Convert ticker to FactSet format (ticker-COUNTRY)
 * Examples: BHP-US, TECK-CA, RIO-GB, BHP-AU
 */
function getFactSetTicker(ticker: string | null, exchange: string | null, country: string | null): string | null {
  if (!ticker) return null

  let suffix = 'US' // Default

  if (exchange) {
    const ex = exchange.toUpperCase()
    if (ex.includes('TSX') || ex.includes('TORONTO')) suffix = 'CA'
    else if (ex.includes('LSE') || ex.includes('LONDON')) suffix = 'GB'
    else if (ex.includes('ASX') || ex.includes('AUSTRALIAN')) suffix = 'AU'
    else if (ex.includes('JSE') || ex.includes('JOHANNESBURG')) suffix = 'ZA'
    else if (ex.includes('NYSE') || ex.includes('NASDAQ')) suffix = 'US'
  } else if (country) {
    const co = country.toUpperCase()
    if (co.includes('CANADA')) suffix = 'CA'
    else if (co.includes('UK') || co.includes('UNITED KINGDOM')) suffix = 'GB'
    else if (co.includes('AUSTRALIA')) suffix = 'AU'
    else if (co.includes('SOUTH AFRICA')) suffix = 'ZA'
  }

  return `${ticker}-${suffix}`
}


/**
 * Search FactSet for technical mining documents
 */
async function searchFactSetDocuments(ticker: string, companyName: string): Promise<FactSetDocument[]> {
  console.log(`\n  🔍 Searching FactSet for ${companyName} (${ticker})...`)

  const documents: FactSetDocument[] = []

  // Search SEDAR (Canadian NI 43-101), SEDAR+, and EDGAR (US)
  const sources = [
    { code: 'SDR', searchText: 'NI 43-101' },
    { code: 'SDRP', searchText: 'NI 43-101' },
    { code: 'EDG', searchText: 'mining technical feasibility' }
  ]

  for (const source of sources) {
    try {
      const params = new URLSearchParams({
        ids: ticker,
        sources: source.code,
        startDate: '20180101', // Last 7 years
        endDate: '20250201',
        searchText: source.searchText,
        _paginationLimit: '50'
      })

      const url = `${FACTSET_BASE_URL}/search?${params.toString()}`

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Authorization': authHeader,
          'Accept': 'application/json'
        }
      })

      if (!response.ok) {
        console.log(`     ⚠️  ${source.code} search failed (${response.status})`)
        await delay(DELAY_MS)
        continue
      }

      const data = await response.json()

      if (data.data && Array.isArray(data.data)) {
        for (const result of data.data) {
          if (result.documents && Array.isArray(result.documents)) {
            for (const doc of result.documents) {
              const headline = (doc.headline || '').toLowerCase()
              const formType = (doc.formTypes?.[0] || '').toLowerCase()

              // Filter for MAIN technical documents only (exclude certificates, consents)
              const isMainTechnicalReport =
                (headline.includes('technical report') || headline.includes('43-101')) &&
                !headline.includes('consent') &&
                !headline.includes('certificate') &&
                !headline.includes('amended consent') &&
                !headline.includes('amended certificate')

              const isFeasibilityStudy =
                (headline.includes('feasibility') || headline.includes('prefeasibility') || headline.includes('pre-feasibility')) &&
                !headline.includes('consent') &&
                !headline.includes('certificate')

              const isResourceUpdate =
                (headline.includes('mineral resource') || headline.includes('mineral reserve') || headline.includes('resource estimate')) &&
                headline.length > 100 && // Longer headlines = full reports
                !headline.includes('consent') &&
                !headline.includes('certificate')

              const isTechnical = isMainTechnicalReport || isFeasibilityStudy || isResourceUpdate

              if (isTechnical && doc.documentId) {
                documents.push({
                  documentId: doc.documentId,
                  headline: doc.headline || 'Untitled',
                  filingDate: doc.filingsDateTime || '',
                  formType: doc.formTypes?.[0] || '',
                  filingSize: doc.filingSize || '',
                  filingsLink: doc.filingsLink || '',
                  source: source.code
                })

                if (documents.length >= MAX_DOCS_PER_COMPANY) break
              }
            }
          }

          if (documents.length >= MAX_DOCS_PER_COMPANY) break
        }
      }

      await delay(DELAY_MS)
    } catch (error: any) {
      console.log(`     ❌ ${source.code} error: ${error.message}`)
    }
  }

  console.log(`     ✅ Found ${documents.length} technical documents`)
  return documents
}

/**
 * Download PDF from FactSet using filingsLink
 */
async function downloadPDF(filingsLink: string): Promise<Buffer | null> {
  try {
    const response = await fetch(filingsLink, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf'
      }
    })

    if (!response.ok) {
      console.log(`       ❌ Download failed (${response.status})`)
      return null
    }

    const buffer = Buffer.from(await response.arrayBuffer())
    const sizeKB = buffer.length / 1024
    const sizeMB = buffer.length / 1024 / 1024
    const estimatedPages = Math.floor(sizeKB / APPROX_KB_PER_PAGE)

    if (sizeMB < MIN_DOCUMENT_SIZE_MB) {
      console.log(`       ⚠️  Too small (${sizeMB.toFixed(1)} MB, ~${estimatedPages} pages), need 100+ pages (~${MIN_DOCUMENT_SIZE_MB}+ MB), skipping`)
      return null
    }

    if (sizeMB > MAX_DOCUMENT_SIZE_MB) {
      console.log(`       ⚠️  Too large (${sizeMB.toFixed(2)} MB), exceeds ${MAX_DOCUMENT_SIZE_MB}MB limit, skipping`)
      return null
    }

    console.log(`       ✅ Downloaded (${sizeMB.toFixed(2)} MB, ~${estimatedPages} pages)`)
    return buffer
  } catch (error: any) {
    console.log(`       ❌ Download error: ${error.message}`)
    return null
  }
}

/**
 * Upload PDF to Supabase Storage with retry logic and direct HTTP API for large files
 */
async function uploadToStorage(buffer: Buffer, ticker: string, documentId: string): Promise<string | null> {
  const maxRetries = 3
  let lastError: any = null

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const hash = crypto.createHash('md5').update(documentId).digest('hex')
      const storagePath = `${ticker}/${hash}.pdf`

      // Check if file already exists
      const { data: existingFile } = await supabase.storage
        .from('factset-documents')
        .list(ticker, {
          search: `${hash}.pdf`
        })

      if (existingFile && existingFile.length > 0) {
        console.log(`       ℹ️  File already exists, using existing`)
        const { data: urlData } = supabase.storage
          .from('factset-documents')
          .getPublicUrl(storagePath)
        return urlData.publicUrl
      }

      // For large files (>10MB), use direct HTTP API with longer timeout
      const sizeMB = buffer.length / 1024 / 1024
      if (sizeMB > 10) {
        console.log(`       📤 Uploading large file (${sizeMB.toFixed(1)} MB) via HTTP API...`)

        const uploadUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/factset-documents/${storagePath}`

        const uploadResponse = await fetch(uploadUrl, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.SUPABASE_SERVICE_ROLE_KEY}`,
            'Content-Type': 'application/pdf',
            'x-upsert': 'true'
          },
          body: buffer,
          // @ts-ignore - Node.js fetch supports signal
          signal: AbortSignal.timeout(300000) // 5 minute timeout
        })

        if (!uploadResponse.ok) {
          const errorText = await uploadResponse.text()
          throw new Error(`HTTP upload failed: ${uploadResponse.status} - ${errorText}`)
        }

        const { data: urlData } = supabase.storage
          .from('factset-documents')
          .getPublicUrl(storagePath)

        console.log(`       ✅ Uploaded (HTTP): ${storagePath}`)
        return urlData.publicUrl
      }

      // For smaller files, use standard SDK
      const { data, error } = await supabase.storage
        .from('factset-documents')
        .upload(storagePath, buffer, {
          contentType: 'application/pdf',
          cacheControl: '3600',
          upsert: true
        })

      if (error) {
        lastError = error
        console.log(`       ⚠️  Upload attempt ${attempt}/${maxRetries} failed: ${error.message}`)
        if (attempt < maxRetries) {
          await delay(3000) // Wait 3 seconds before retry
          continue
        }
        return null
      }

      const { data: urlData } = supabase.storage
        .from('factset-documents')
        .getPublicUrl(storagePath)

      console.log(`       ✅ Uploaded: ${storagePath}`)
      return urlData.publicUrl
    } catch (error: any) {
      lastError = error
      console.log(`       ⚠️  Upload attempt ${attempt}/${maxRetries} error: ${error.message}`)
      if (attempt < maxRetries) {
        await delay(3000)
        continue
      }
    }
  }

  console.log(`       ❌ Upload failed after ${maxRetries} attempts: ${lastError?.message || 'Unknown error'}`)
  return null
}

/**
 * Extract project name from document headline using OpenAI
 */
async function extractProjectNameWithAI(headline: string, companyName: string): Promise<string> {
  try {
    const prompt = `Extract the mining project name from this document headline. Return ONLY the project name, nothing else.

Headline: "${headline}"
Company: "${companyName}"

Rules:
- If a specific project/mine/property name is mentioned, return just that name
- Remove words like "Technical Report", "NI 43-101", "Feasibility", "Certificate", "Consent"
- If no specific project name, return: "${companyName} Mining Project"
- Keep it under 80 characters

Examples:
"Technical report (NI 43-101) - The Vareš Polymetallic Project" → "Vareš Polymetallic Project"
"Feasibility Study for the Quebrada Blanca Phase 2 Project" → "Quebrada Blanca Phase 2"
"Updated Mineral Resource Estimate for Detour Lake" → "Detour Lake"

Project name:`

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: 'You are a mining project name extraction expert. Return only the clean project name.' },
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 50
    })

    let name = response.choices[0].message.content?.trim() || ''

    // Fallback if AI returns empty or error
    if (name.length < 3) {
      name = headline
        .replace(/NI\s*43-101/gi, '')
        .replace(/technical\s+report/gi, '')
        .replace(/feasibility\s+study/gi, '')
        .replace(/pre-feasibility/gi, '')
        .replace(/mineral\s+resource/gi, '')
        .replace(/mineral\s+reserve/gi, '')
        .replace(/consent\s+of\s+qualified\s+person/gi, '')
        .replace(/certificate\s+of\s+qualified\s+person/gi, '')
        .trim()
    }

    if (name.length > 100) {
      name = name.substring(0, 97) + '...'
    }

    if (name.length < 3) {
      name = `${companyName} Mining Project`
    }

    return name
  } catch (error: any) {
    console.log(`       ⚠️  AI extraction failed: ${error.message}, using fallback`)
    return headline.substring(0, 80)
  }
}

// ============================
// Main Process
// ============================

/**
 * Process a single company
 */
async function processCompany(company: Company, index: number, total: number): Promise<number> {
  console.log(`\n${'='.repeat(80)}`)
  console.log(`[${index + 1}/${total}] 📊 ${company.name}`)
  console.log(`${'='.repeat(80)}`)

  const ticker = getFactSetTicker(company.ticker, company.exchange, company.country)
  if (!ticker) {
    console.log('  ⚠️  No ticker available, skipping')
    return 0
  }

  // Search for documents
  const documents = await searchFactSetDocuments(ticker, company.name)

  if (documents.length === 0) {
    console.log('  ℹ️  No technical documents found')
    return 0
  }

  let projectsCreated = 0

  // Process each document
  for (const doc of documents) {
    console.log(`\n    📄 ${doc.headline.substring(0, 70)}...`)
    console.log(`       Date: ${doc.filingDate.substring(0, 10)}, Source: ${doc.source}`)

    // Download PDF using filingsLink
    const pdfBuffer = await downloadPDF(doc.filingsLink)
    if (!pdfBuffer) {
      await delay(DELAY_MS)
      continue
    }

    await delay(DELAY_MS)

    // Upload to storage
    const publicUrl = await uploadToStorage(pdfBuffer, ticker, doc.documentId)
    if (!publicUrl) {
      await delay(DELAY_MS)
      continue
    }

    // Extract project name using OpenAI
    console.log(`       🤖 Extracting project name with AI...`)
    const projectName = await extractProjectNameWithAI(doc.headline, company.name)
    console.log(`       📝 Project name: ${projectName}`)
    const projectId = crypto.randomUUID()

    try {
      const { error } = await supabase.from('projects').insert({
        id: projectId,
        company_id: company.id,
        name: projectName,
        location: company.country || 'Unknown',
        stage: 'Resource Definition',
        commodities: [],
        status: 'Active',
        urls: [], // Don't store FactSet links, only Supabase storage path
        description: `Technical Report: ${doc.headline.substring(0, 200)}`,
        document_storage_path: publicUrl,
        watchlist: false
      })

      if (error) {
        console.log(`       ❌ Insert failed: ${error.message}`)
      } else {
        console.log(`       ✅ Project created: ${projectName}`)
        projectsCreated++
      }
    } catch (error: any) {
      console.log(`       ❌ Error: ${error.message}`)
    }

    await delay(DELAY_MS)
  }

  return projectsCreated
}

async function main() {
  console.log('🚀 FACTSET EXHAUSTIVE MINING PROJECT POPULATION')
  console.log('='.repeat(80))

  if (!FACTSET_USERNAME || !FACTSET_API_KEY) {
    console.error('\n❌ FactSet credentials not configured')
    console.error('Set FACTSET_USERNAME and FACTSET_API_KEY in .env.local')
    process.exit(1)
  }

  console.log('✅ FactSet credentials configured')
  console.log(`🔑 Username: ${FACTSET_USERNAME}\n`)

  console.log('📋 Fetching companies from database...\n')

  const { data: companies, error } = await supabase
    .from('companies')
    .select('id, name, ticker, exchange, country')
    .order('name')

  if (error || !companies) {
    console.error('❌ Failed to fetch companies:', error)
    process.exit(1)
  }

  console.log(`✅ Found ${companies.length} companies`)
  console.log('🔄 Starting EXHAUSTIVE processing of all companies...\n')

  // EXHAUSTIVE MODE: Process ALL 123 companies
  const TEST_MODE = false
  const companiesToProcess = TEST_MODE ? companies.slice(0, 10) : companies

  let totalProjects = 0
  const startTime = Date.now()

  for (let i = 0; i < companiesToProcess.length; i++) {
    const projectsCreated = await processCompany(companiesToProcess[i], i, companiesToProcess.length)
    totalProjects += projectsCreated

    console.log(`\n✅ Company ${i + 1}/${companiesToProcess.length} complete`)
    console.log(`📈 Running total: ${totalProjects} projects`)

    // Longer delay between companies
    await delay(DELAY_MS * 2)
  }

  const endTime = Date.now()
  const duration = ((endTime - startTime) / 1000 / 60).toFixed(1)

  console.log('\n' + '='.repeat(80))
  console.log('🎉 PIPELINE COMPLETE!')
  console.log('='.repeat(80))
  console.log(`✅ Processed: ${companiesToProcess.length} companies`)
  console.log(`✅ Created: ${totalProjects} projects`)
  console.log(`⏱️  Duration: ${duration} minutes`)
  console.log('='.repeat(80))

  if (TEST_MODE) {
    console.log('\n⚠️  TEST MODE: Only processed first 10 companies')
    console.log('   To process all 123 companies, set TEST_MODE = false in the script')
  } else {
    console.log('\n🎯 EXHAUSTIVE MODE: Processed all companies in database')
  }
}

main().catch(console.error)
