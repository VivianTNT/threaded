#!/usr/bin/env npx tsx
/**
 * FactSet Multi-Document Extraction Pipeline
 *
 * For each project/company:
 * 1. Find ALL related technical documents (Annual reports, NI 43-101, etc)
 * 2. Download all PDFs
 * 3. Extract data from EACH document
 * 4. Merge/aggregate the best values
 * 5. Create ONE project entry with comprehensive data
 */

import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'

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

const MINING_COMPANIES = [
  { ticker: 'ORV-CA', name: 'Orvana Minerals', commodity: 'Gold', country: 'Canada' },
  { ticker: 'FM-CA', name: 'First Quantum Minerals', commodity: 'Copper', country: 'Canada' },
  { ticker: 'LUN-CA', name: 'Lundin Mining', commodity: 'Copper', country: 'Canada' },
  { ticker: 'TKO-CA', name: 'Taseko Mines', commodity: 'Copper', country: 'Canada' },
  { ticker: 'ERO-CA', name: 'Ero Copper', commodity: 'Copper', country: 'Canada' }
]

interface ExtractedFinancials {
  npv?: number
  irr?: number
  capex?: number
  opex?: number
  location?: string
  commodities?: string[]
  stage?: string
  resources?: string
  reserves?: string
  mineLife?: number
}

async function searchAllDocuments(ticker: string): Promise<any[]> {
  console.log(`   🔍 Searching for all documents...`)

  const params = new URLSearchParams({
    ids: ticker,
    sources: 'SDR',  // SEDAR
    startDate: '20180101',
    endDate: '20241231',
    searchText: 'Technical Report OR Annual',
    _paginationLimit: '100' // Get many documents
  })

  const url = `https://api.factset.com/content/global-filings/v2/search?${params.toString()}`

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      'Authorization': authHeader,
      'Accept': 'application/json'
    }
  })

  if (!response.ok) return []

  const data = await response.json()
  const documents: any[] = []

  if (data.data && Array.isArray(data.data)) {
    for (const result of data.data) {
      if (result.documents && Array.isArray(result.documents)) {
        for (const doc of result.documents) {
          documents.push({
            headline: doc.headline,
            filingDate: doc.filingsDateTime,
            documentId: doc.documentId,
            pdfUrl: doc.filingsLink?.replace('report=story', 'report=pdf'),
            size: doc.filingSize || 0
          })
        }
      }
    }
  }

  console.log(`   📄 Found ${documents.length} documents`)
  return documents
}

async function downloadPDF(url: string): Promise<Buffer | null> {
  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf'
      }
    })

    if (!response.ok) return null

    const buffer = Buffer.from(await response.arrayBuffer())
    if (buffer.slice(0, 5).toString() === '%PDF-') {
      return buffer
    }

    return null
  } catch {
    return null
  }
}

async function extractFromPDFChunked(pdfText: string, headline: string): Promise<ExtractedFinancials> {
  // Take strategic chunks from the PDF (first 20k chars + search for key terms)
  const chunk1 = pdfText.substring(0, 20000)
  const chunk2 = pdfText.substring(Math.floor(pdfText.length * 0.3), Math.floor(pdfText.length * 0.3) + 20000)

  const combinedText = `Document: ${headline}\n\n${chunk1}\n\n[...]\n\n${chunk2}`

  const prompt = `Extract financial metrics from this mining technical report.

Look for:
- NPV (Net Present Value): After-tax NPV in millions USD/CAD
- IRR (Internal Rate of Return): After-tax IRR as percentage
- CAPEX (Capital Expenditure): Initial capital in millions USD/CAD
- OPEX (Operating Expenditure): Operating cost per tonne or per oz
- Mine Life: Years
- Location: Geographic location
- Stage: Development stage
- Commodities: Minerals being mined

Return as JSON:
{
  "npv": number or null,
  "irr": number or null,
  "capex": number or null,
  "opex": number or null,
  "mineLife": number or null,
  "location": "string or null",
  "stage": "string or null",
  "commodities": ["array"] or null,
  "resources": "brief description or null",
  "reserves": "brief description or null"
}

Text:
${combinedText}`

  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      response_format: { type: 'json_object' }
    })

    const extracted = JSON.parse(completion.choices[0].message.content || '{}')
    return extracted as ExtractedFinancials

  } catch (error: any) {
    console.log(`      ⚠️  AI extraction error: ${error.message}`)
    return {}
  }
}

async function extractTextFromPDF(pdfBuffer: Buffer): Promise<string> {
  try {
    // Use pdf-parse with CommonJS require
    const pdfParse = require('pdf-parse')
    const data = await pdfParse(pdfBuffer)
    return data.text
  } catch (error: any) {
    console.log(`      ⚠️  PDF parsing error: ${error.message}`)
    return ''
  }
}

function mergefinancials(allData: ExtractedFinancials[]): ExtractedFinancials {
  const merged: ExtractedFinancials = {}

  // Take the most recent/highest NPV
  const npvValues = allData.filter(d => d.npv).map(d => d.npv!)
  if (npvValues.length > 0) {
    merged.npv = Math.max(...npvValues)
  }

  // Take the most recent/highest IRR
  const irrValues = allData.filter(d => d.irr).map(d => d.irr!)
  if (irrValues.length > 0) {
    merged.irr = Math.max(...irrValues)
  }

  // Take the most recent/highest CAPEX
  const capexValues = allData.filter(d => d.capex).map(d => d.capex!)
  if (capexValues.length > 0) {
    merged.capex = Math.max(...capexValues)
  }

  // Take first available location
  merged.location = allData.find(d => d.location)?.location

  // Merge all commodities
  const allCommodities = allData.flatMap(d => d.commodities || [])
  if (allCommodities.length > 0) {
    merged.commodities = [...new Set(allCommodities)]
  }

  // Take first available stage
  merged.stage = allData.find(d => d.stage)?.stage

  // Take longest resources/reserves description
  const resourcesDesc = allData.filter(d => d.resources).sort((a, b) => (b.resources?.length || 0) - (a.resources?.length || 0))
  if (resourcesDesc.length > 0) {
    merged.resources = resourcesDesc[0].resources
  }

  const reservesDesc = allData.filter(d => d.reserves).sort((a, b) => (b.reserves?.length || 0) - (a.reserves?.length || 0))
  if (reservesDesc.length > 0) {
    merged.reserves = reservesDesc[0].reserves
  }

  // Take longest mine life
  const mineLifeValues = allData.filter(d => d.mineLife).map(d => d.mineLife!)
  if (mineLifeValues.length > 0) {
    merged.mineLife = Math.max(...mineLifeValues)
  }

  return merged
}

async function run() {
  console.log('🏭 FACTSET MULTI-DOCUMENT EXTRACTION PIPELINE')
  console.log('='.repeat(80))
  console.log('📋 For each company, this will:')
  console.log('   1. Find ALL technical documents (Reports, Annual filings, etc)')
  console.log('   2. Download and extract data from EACH document')
  console.log('   3. Merge/aggregate the best financial metrics')
  console.log('   4. Create ONE comprehensive project entry')
  console.log('='.repeat(80))

  // Clear old projects
  console.log('\n🗑️  Clearing old FactSet projects...')
  const { data: oldProjects } = await supabase
    .from('projects')
    .select('id')
    .like('description', '%FactSet SEDAR%')

  if (oldProjects && oldProjects.length > 0) {
    await supabase.from('projects').delete().in('id', oldProjects.map(p => p.id))
    console.log(`   ✅ Cleared ${oldProjects.length} old projects`)
  }

  let totalProjects = 0

  for (const company of MINING_COMPANIES) {
    console.log(`\n${'='.repeat(80)}`)
    console.log(`📊 ${company.name} (${company.ticker})`)
    console.log('='.repeat(80))

    // Ensure company exists
    let companyId: string
    const { data: existingCompany } = await supabase
      .from('companies')
      .select('id')
      .eq('ticker_symbol', company.ticker)
      .single()

    if (existingCompany) {
      companyId = existingCompany.id
    } else {
      companyId = crypto.randomUUID()
      await supabase.from('companies').insert({
        id: companyId,
        name: company.name,
        ticker_symbol: company.ticker,
        primary_commodity: company.commodity,
        headquarters_country: company.country
      })
      console.log(`   ✅ Created company record`)
    }

    // Find all documents
    const documents = await searchAllDocuments(company.ticker)

    if (documents.length === 0) {
      console.log(`   ⚠️  No documents found`)
      continue
    }

    // Download and extract from multiple documents
    const allExtractedData: ExtractedFinancials[] = []
    const allPdfUrls: string[] = []
    let processedDocs = 0

    for (const doc of documents.slice(0, 5)) { // Process up to 5 docs per company
      console.log(`\n   📄 ${doc.headline.substring(0, 60)}...`)

      // Download PDF
      const pdfBuffer = await downloadPDF(doc.pdfUrl)
      if (!pdfBuffer) {
        console.log(`      ⏭️  Download failed`)
        continue
      }

      const actualSizeMB = pdfBuffer.length / 1024 / 1024
      console.log(`      📦 ${actualSizeMB.toFixed(1)} MB`)

      // Skip very small files (consent forms, cover letters, etc)
      if (actualSizeMB < 0.5) {
        console.log(`      ⏭️  Too small`)
        continue
      }

      // Upload to Supabase
      const fileName = `${company.ticker}-${doc.documentId}.pdf`
      const filePath = `factset-documents/${fileName}`

      const { error: uploadError } = await supabase.storage
        .from('mining-documents')
        .upload(filePath, pdfBuffer, {
          contentType: 'application/pdf',
          upsert: true
        })

      if (!uploadError) {
        const { data: { publicUrl } } = supabase.storage
          .from('mining-documents')
          .getPublicUrl(filePath)

        allPdfUrls.push(publicUrl)
        console.log(`      ✅ Uploaded`)

        // Extract text and financial data
        console.log(`      🤖 Extracting financial data...`)
        const text = await extractTextFromPDF(pdfBuffer)

        if (text.length > 1000) {
          const extracted = await extractFromPDFChunked(text, doc.headline)

          if (extracted.npv || extracted.irr || extracted.capex) {
            console.log(`      ✅ Extracted:`, {
              npv: extracted.npv || 'N/A',
              irr: extracted.irr || 'N/A',
              capex: extracted.capex || 'N/A'
            })
            allExtractedData.push(extracted)
          }
        }

        processedDocs++
      }

      await new Promise(resolve => setTimeout(resolve, 2000))
    }

    // Merge all extracted data
    if (allExtractedData.length > 0) {
      const merged = mergefinancials(allExtractedData)

      console.log(`\n   💎 MERGED DATA FROM ${allExtractedData.length} DOCUMENTS:`)
      console.log(`      NPV: $${merged.npv || 'N/A'}M`)
      console.log(`      IRR: ${merged.irr || 'N/A'}%`)
      console.log(`      CAPEX: $${merged.capex || 'N/A'}M`)
      console.log(`      Location: ${merged.location || 'Unknown'}`)

      // Create comprehensive project
      const projectData = {
        id: crypto.randomUUID(),
        company_id: companyId,
        name: company.name,
        location: merged.location || company.country,
        stage: merged.stage || 'Unknown',
        commodities: merged.commodities || [company.commodity],
        status: 'Active',
        description: `${company.name} - Comprehensive data extracted from ${processedDocs} FactSet SEDAR documents\n\nSource: FactSet Global Filings API\n\nResources: ${merged.resources || 'See technical reports'}\n\nReserves: ${merged.reserves || 'See technical reports'}${merged.mineLife ? `\n\nMine Life: ${merged.mineLife} years` : ''}`,
        urls: allPdfUrls,
        npv: merged.npv || null,
        irr: merged.irr || null,
        capex: merged.capex || null,
        watchlist: false
      }

      const { error } = await supabase.from('projects').insert(projectData)

      if (!error) {
        console.log(`   ✅ Created comprehensive project with ${allPdfUrls.length} documents`)
        totalProjects++
      } else {
        console.log(`   ❌ DB error: ${error.message}`)
      }
    } else {
      console.log(`   ⚠️  No financial data extracted from any document`)
    }

    await new Promise(resolve => setTimeout(resolve, 3000))
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 SUMMARY')
  console.log('='.repeat(80))
  console.log(`✅ Projects created: ${totalProjects}`)
  console.log('\n✨ Multi-document extraction complete!')
}

run().catch(console.error)
