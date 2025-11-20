#!/usr/bin/env npx tsx

/**
 * Comprehensive FactSet Project Population & Extraction Pipeline
 *
 * This script:
 * 1. Loads ALL companies from Supabase
 * 2. Searches FactSet for technical documents (NI 43-101, SEC filings, 100+ pages)
 * 3. Downloads PDFs
 * 4. Uploads to Supabase Storage
 * 5. Creates project records
 * 6. Extracts financial data using OpenAI
 * 7. Updates all project columns
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import { getDocument } from 'unpdf'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Configuration
const MIN_DOCUMENT_SIZE_KB = 5000 // 5MB = ~100+ pages
const MAX_DOCUMENTS_PER_COMPANY = 3
const RATE_LIMIT_DELAY = 200

interface Company {
  id: string
  name: string
  ticker: string | null
  exchange: string | null
}

interface Filing {
  headline: string
  documentId: string
  filingsLink: string
  filingSize: string
  filingsDateTime: string
}

async function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms))
}

async function loadAllCompanies(): Promise<Company[]> {
  const { data, error } = await supabase
    .from('companies')
    .select('id, name, ticker, exchange')
    .not('ticker', 'is', null)
    .order('name')

  if (error) {
    console.error('Error loading companies:', error)
    return []
  }

  return data || []
}

function convertTickerToFactSetFormat(ticker: string, exchange: string | null): string {
  // Convert exchange ticker to FactSet format
  const cleanTicker = ticker.replace(/\.(TO|V|TSX|TSXV)$/, '')

  if (exchange === 'TSX' || exchange === 'Toronto Stock Exchange') {
    return `${cleanTicker}-CA`
  } else if (exchange === 'TSXV' || exchange === 'TSX Venture Exchange') {
    return `${cleanTicker}-CA`
  } else if (exchange === 'NYSE' || exchange === 'NASDAQ' || exchange === 'US') {
    return `${cleanTicker}-US`
  } else if (exchange === 'ASX') {
    return `${cleanTicker}-AU`
  } else if (exchange === 'LSE') {
    return `${cleanTicker}-GB`
  }

  return ticker
}

async function searchFactSetDocuments(company: Company): Promise<Filing[]> {
  if (!company.ticker) return []

  const factsetTicker = convertTickerToFactSetFormat(company.ticker, company.exchange)

  try {
    // Search SEDAR+ for NI 43-101 documents (Canadian)
    const params = new URLSearchParams({
      ids: factsetTicker,
      sources: 'SDRP,SDR,EDG', // SEDAR+, SEDAR, EDGAR
      startDate: '20200101',
      endDate: '20241231',
      searchText: 'NI 43-101 Technical Report OR Feasibility Study OR Preliminary Economic Assessment',
      _paginationLimit: '20'
    })

    const url = `https://api.factset.com/content/global-filings/v2/search?${params.toString()}`

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/json'
      }
    })

    if (!response.ok) {
      console.log(`   ❌ Search failed (${response.status})`)
      return []
    }

    const data = await response.json()
    const filings: Filing[] = []

    if (data.data && Array.isArray(data.data)) {
      for (const result of data.data) {
        if (result.documents && Array.isArray(result.documents)) {
          for (const doc of result.documents) {
            // Parse file size
            const sizeStr = doc.filingSize || '0'
            const sizeKB = parseInt(sizeStr.replace(/[^\d]/g, '')) || 0

            // Only include large technical documents (100+ pages)
            if (sizeKB >= MIN_DOCUMENT_SIZE_KB) {
              filings.push({
                headline: doc.headline,
                documentId: doc.documentId,
                filingsLink: doc.filingsLink,
                filingSize: doc.filingSize,
                filingsDateTime: doc.filingsDateTime
              })
            }
          }
        }
      }
    }

    // Limit to most recent large documents
    return filings.slice(0, MAX_DOCUMENTS_PER_COMPANY)

  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`)
    return []
  }
}

async function downloadPDF(filingLink: string): Promise<Buffer | null> {
  try {
    // Convert HTML link to PDF link
    const pdfUrl = filingLink.replace('report=story', 'report=pdf')

    const response = await fetch(pdfUrl, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf'
      }
    })

    if (!response.ok) return null

    const buffer = Buffer.from(await response.arrayBuffer())

    // Validate PDF header
    if (buffer.slice(0, 5).toString() === '%PDF-') {
      return buffer
    }

    return null
  } catch (error) {
    return null
  }
}

async function uploadToStorage(pdfBuffer: Buffer, documentId: string, companyName: string): Promise<string | null> {
  try {
    const sanitizedName = companyName.replace(/[^a-zA-Z0-9]/g, '_')
    const filename = `${sanitizedName}_${documentId}.pdf`
    const filepath = `project-documents/${filename}`

    const { error } = await supabase.storage
      .from('factset-documents')
      .upload(filepath, pdfBuffer, {
        contentType: 'application/pdf',
        upsert: true
      })

    if (error) {
      console.log(`      ⚠️  Storage upload error: ${error.message}`)
      return null
    }

    // Get public URL
    const { data } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(filepath)

    return data.publicUrl
  } catch (error: any) {
    console.log(`      ⚠️  Storage error: ${error.message}`)
    return null
  }
}

async function extractTextFromPDF(pdfBuffer: Buffer): Promise<string> {
  try {
    const pdf = await getDocument(pdfBuffer).promise
    let fullText = ''

    // Extract first 50 pages for analysis (enough for executive summary and financial data)
    const maxPages = Math.min(pdf.numPages, 50)

    for (let i = 1; i <= maxPages; i++) {
      const page = await pdf.getPage(i)
      const content = await page.getTextContent()
      const pageText = content.items.map((item: any) => item.str).join(' ')
      fullText += pageText + '\n'
    }

    return fullText
  } catch (error: any) {
    console.log(`      ⚠️  PDF extraction error: ${error.message}`)
    return ''
  }
}

async function extractFinancialData(text: string, projectName: string): Promise<any> {
  try {
    const prompt = `You are analyzing a mining project technical report. Extract the following financial and technical data:

Project: ${projectName}

Text excerpt (first 50 pages):
${text.substring(0, 50000)}

Extract and return JSON with these fields:
{
  "npv_usd_millions": number or null,
  "irr_percentage": number or null,
  "capex_usd_millions": number or null,
  "opex_usd_millions": number or null,
  "payback_period_years": number or null,
  "mine_life_years": number or null,
  "production_rate": string or null,
  "discount_rate_percentage": number or null,
  "location": string or null,
  "stage": string or null (e.g., "Feasibility", "PEA", "Operating"),
  "commodities": string[] or null,
  "resource_estimate": string or null,
  "reserve_estimate": string or null
}

Return ONLY valid JSON. Use null if data not found.`

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      response_format: { type: 'json_object' }
    })

    const result = completion.choices[0].message.content
    return result ? JSON.parse(result) : {}
  } catch (error: any) {
    console.log(`      ⚠️  OpenAI extraction error: ${error.message}`)
    return {}
  }
}

async function createProjectRecord(company: Company, filing: Filing, storageUrl: string, financialData: any): Promise<boolean> {
  try {
    const projectData: any = {
      company_id: company.id,
      name: filing.headline.substring(0, 200),
      location: financialData.location,
      stage: financialData.stage,
      commodities: financialData.commodities,
      resource_estimate: financialData.resource_estimate,
      reserve_estimate: financialData.reserve_estimate,
      npv_usd_millions: financialData.npv_usd_millions,
      irr_percentage: financialData.irr_percentage,
      capex_usd_millions: financialData.capex_usd_millions,
      opex_usd_millions: financialData.opex_usd_millions,
      payback_period_years: financialData.payback_period_years,
      mine_life_years: financialData.mine_life_years,
      production_rate: financialData.production_rate,
      discount_rate_percentage: financialData.discount_rate_percentage,
      document_urls: [storageUrl],
      document_storage_path: storageUrl,
      financial_metrics_updated_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    const { error } = await supabase
      .from('projects')
      .insert(projectData)

    if (error) {
      console.log(`      ⚠️  Insert error: ${error.message}`)
      return false
    }

    return true
  } catch (error: any) {
    console.log(`      ⚠️  Error creating project: ${error.message}`)
    return false
  }
}

async function processCompany(company: Company, index: number, total: number): Promise<number> {
  console.log(`\n[${index}/${total}] ${company.name} (${company.ticker})`)

  // Search for documents
  console.log('   🔍 Searching FactSet...')
  const filings = await searchFactSetDocuments(company)

  if (filings.length === 0) {
    console.log('   ℹ️  No large technical documents found')
    return 0
  }

  console.log(`   ✅ Found ${filings.length} large technical document(s)`)

  let projectsCreated = 0

  for (let i = 0; i < filings.length; i++) {
    const filing = filings[i]
    console.log(`\n   📄 [${i + 1}/${filings.length}] ${filing.headline.substring(0, 60)}...`)
    console.log(`      Size: ${filing.filingSize}`)

    // Download PDF
    console.log('      ⬇️  Downloading PDF...')
    const pdfBuffer = await downloadPDF(filing.filingsLink)

    if (!pdfBuffer) {
      console.log('      ❌ Download failed')
      continue
    }

    console.log(`      ✅ Downloaded ${(pdfBuffer.length / 1024 / 1024).toFixed(1)}MB`)

    // Upload to storage
    console.log('      ☁️  Uploading to Supabase Storage...')
    const storageUrl = await uploadToStorage(pdfBuffer, filing.documentId, company.name)

    if (!storageUrl) {
      console.log('      ❌ Storage upload failed')
      continue
    }

    console.log('      ✅ Uploaded to storage')

    // Extract text
    console.log('      📖 Extracting text from PDF...')
    const text = await extractTextFromPDF(pdfBuffer)

    if (!text || text.length < 1000) {
      console.log('      ⚠️  Insufficient text extracted')
      continue
    }

    console.log(`      ✅ Extracted ${(text.length / 1024).toFixed(0)}KB text`)

    // Extract financial data
    console.log('      🤖 Extracting financial data with OpenAI...')
    const financialData = await extractFinancialData(text, filing.headline)

    // Create project record
    console.log('      💾 Creating project record...')
    const success = await createProjectRecord(company, filing, storageUrl, financialData)

    if (success) {
      console.log('      ✅ Project created with financial data')
      projectsCreated++

      // Log extracted metrics
      if (financialData.npv_usd_millions) {
        console.log(`         💰 NPV: $${financialData.npv_usd_millions}M`)
      }
      if (financialData.irr_percentage) {
        console.log(`         📈 IRR: ${financialData.irr_percentage}%`)
      }
      if (financialData.capex_usd_millions) {
        console.log(`         🏗️  CAPEX: $${financialData.capex_usd_millions}M`)
      }
    } else {
      console.log('      ❌ Failed to create project')
    }

    // Rate limiting
    await sleep(RATE_LIMIT_DELAY)
  }

  return projectsCreated
}

async function main() {
  console.log('======================================================================')
  console.log('COMPREHENSIVE FACTSET PROJECT POPULATION & EXTRACTION PIPELINE')
  console.log('======================================================================')
  console.log()
  console.log('📋 Configuration:')
  console.log(`   Minimum document size: ${MIN_DOCUMENT_SIZE_KB}KB (~100+ pages)`)
  console.log(`   Max documents per company: ${MAX_DOCUMENTS_PER_COMPANY}`)
  console.log()

  // Load all companies
  console.log('📊 Loading companies from database...')
  const companies = await loadAllCompanies()
  console.log(`✅ Loaded ${companies.length} companies`)
  console.log()

  // Process each company
  let totalProjects = 0

  for (let i = 0; i < companies.length; i++) {
    const company = companies[i]
    const projectsCreated = await processCompany(company, i + 1, companies.length)
    totalProjects += projectsCreated

    // Rate limiting between companies
    await sleep(RATE_LIMIT_DELAY)
  }

  console.log()
  console.log('======================================================================')
  console.log('PIPELINE COMPLETE')
  console.log('======================================================================')
  console.log()
  console.log('📊 Summary:')
  console.log(`   Companies processed: ${companies.length}`)
  console.log(`   Total projects created: ${totalProjects}`)
  console.log()
  console.log('✅ All documents downloaded, uploaded to storage, and financial data extracted!')
  console.log('======================================================================')
}

main().catch(console.error)
