#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import pdfParse from 'pdf-parse'

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

// Canadian mining companies for SEDAR NI 43-101 reports
const MINING_COMPANIES = [
  { ticker: 'FM-CA', name: 'First Quantum Minerals', primary_commodity: 'Copper' },
  { ticker: 'ABX-CA', name: 'Barrick Gold', primary_commodity: 'Gold' },
  { ticker: 'K-CA', name: 'Kinross Gold', primary_commodity: 'Gold' },
  { ticker: 'AEM-CA', name: 'Agnico Eagle Mines', primary_commodity: 'Gold' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', primary_commodity: 'Copper' },
  { ticker: 'CS-CA', name: 'Capstone Copper', primary_commodity: 'Copper' },
  { ticker: 'TECK-CA', name: 'Teck Resources', primary_commodity: 'Coal' },
  { ticker: 'IMG-CA', name: 'IAMGOLD', primary_commodity: 'Gold' },
  { ticker: 'TKO-CA', name: 'Taseko Mines', primary_commodity: 'Copper' },
  { ticker: 'NGD-CA', name: 'New Gold', primary_commodity: 'Gold' }
]

interface Filing {
  companyName: string
  ticker: string
  headline: string
  filingDate: string
  documentId: string
  pdfUrl: string
  htmlUrl: string
  size: number
}

interface ExtractedData {
  projectName: string
  location: string
  stage: string
  commodities: string[]
  npv?: number
  irr?: number
  capex?: number
  opex?: number
  reserves?: string
  resources?: string
  mineLife?: number
  production?: string
}

async function searchSEDARFilings(ticker: string, companyName: string): Promise<Filing[]> {
  console.log(`\n🔍 Searching SEDAR for ${companyName} (${ticker})...`)

  try {
    // Search for NI 43-101 Technical Reports specifically
    const params = new URLSearchParams({
      ids: ticker,
      sources: 'SDR',  // SEDAR
      startDate: '20180101', // Go back further for more comprehensive reports
      endDate: '20241231',
      searchText: 'NI 43-101 Technical Report',
      _paginationLimit: '10' // Get more documents
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
          console.log(`   ✅ Found ${result.documents.length} filings`)

          for (const doc of result.documents) {
            const pdfUrl = doc.filingsLink?.replace('report=story', 'report=pdf')
            const sizeKB = doc.filingSize || 0

            // Include all NI 43-101 technical reports
            filings.push({
              companyName,
              ticker,
              headline: doc.headline,
              filingDate: doc.filingsDateTime,
              documentId: doc.documentId,
              htmlUrl: doc.filingsLink,
              pdfUrl: pdfUrl,
              size: sizeKB
            })

            console.log(`      📄 ${doc.headline?.substring(0, 60)}... (${sizeKB} KB)`)
          }
        }
      }
    }

    console.log(`   📊 Found ${filings.length} NI 43-101 technical reports`)
    return filings

  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`)
    return []
  }
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

    if (!response.ok) {
      return null
    }

    const buffer = Buffer.from(await response.arrayBuffer())

    // Validate PDF
    const pdfHeader = buffer.slice(0, 5).toString()
    if (pdfHeader === '%PDF-') {
      return buffer
    }

    return null
  } catch (error: any) {
    return null
  }
}

async function extractTextFromPDF(pdfBuffer: Buffer): Promise<string> {
  try {
    const data = await pdfParse(pdfBuffer)
    return data.text
  } catch (error: any) {
    console.log(`      ⚠️  PDF parsing error: ${error.message}`)
    return ''
  }
}

async function extractFinancialData(text: string, companyName: string, headline: string): Promise<ExtractedData | null> {
  console.log(`      🤖 Extracting financial data with AI...`)

  try {
    // Take first 30000 chars to stay within token limits, focus on executive summary and economics
    const textSample = text.substring(0, 30000)

    const prompt = `You are analyzing a mining NI 43-101 Technical Report for ${companyName}.

Document title: ${headline}

Extract the following information from this technical report. Be precise and only extract if the information is explicitly stated.

REQUIRED FIELDS:
- Project Name: The specific mine/project name
- Location: Country and region
- Development Stage: (Exploration/Feasibility/Pre-Production/Production/Expansion)
- Primary Commodity: Main mineral being mined
- Secondary Commodities: Other minerals (if any)

FINANCIAL METRICS (in USD millions unless otherwise stated):
- NPV (Net Present Value): After-tax NPV at specified discount rate
- IRR (Internal Rate of Return): After-tax IRR as percentage
- CAPEX (Capital Expenditure): Initial capital cost
- OPEX (Operating Expenditure): Operating cost per unit ($/tonne or $/oz)

TECHNICAL METRICS:
- Mineral Reserves: Proven and probable reserves with grades
- Mineral Resources: Measured, indicated, and inferred resources
- Mine Life: Expected operational life in years
- Production Rate: Annual production rate

Return as JSON with this exact structure:
{
  "projectName": "string",
  "location": "string",
  "stage": "string",
  "commodities": ["array of strings"],
  "npv": number or null,
  "irr": number or null,
  "capex": number or null,
  "opex": number or null,
  "reserves": "string description or null",
  "resources": "string description or null",
  "mineLife": number or null,
  "production": "string description or null"
}

IMPORTANT:
- For NPV/CAPEX/OPEX, extract the number in millions USD
- For IRR, extract as percentage (e.g., 15.5 for 15.5%)
- If a value is not found, use null
- Be conservative - only extract if you're confident

Here is the text to analyze:

${textSample}`

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.1,
      response_format: { type: 'json_object' }
    })

    const extracted = JSON.parse(completion.choices[0].message.content || '{}')
    console.log(`      ✅ Extracted data for: ${extracted.projectName || companyName}`)

    return extracted as ExtractedData

  } catch (error: any) {
    console.log(`      ❌ AI extraction error: ${error.message}`)
    return null
  }
}

async function processComprehensiveExtraction() {
  console.log('🏭 COMPREHENSIVE FACTSET EXTRACTION PIPELINE')
  console.log('='.repeat(80))
  console.log('📋 This will:')
  console.log('   1. Download ALL large technical reports (100+ pages)')
  console.log('   2. Extract text from PDFs')
  console.log('   3. Use AI to extract financial data BEFORE creating DB entries')
  console.log('   4. Store PDFs in Supabase Storage')
  console.log('='.repeat(80))

  // Clear existing projects
  console.log('\n🗑️  Clearing existing projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')

  let totalProjects = 0
  let totalDocuments = 0

  for (const company of MINING_COMPANIES) {
    console.log(`\n${'='.repeat(80)}`)
    console.log(`📊 Processing ${company.name}`)
    console.log('='.repeat(80))

    const filings = await searchSEDARFilings(company.ticker, company.name)

    if (filings.length === 0) {
      console.log(`   ⚠️  No large technical reports found, skipping...`)
      continue
    }

    // Process each filing (up to 3 per company to get multiple projects)
    const filingsToProcess = filings.slice(0, 3)

    for (const filing of filingsToProcess) {
      console.log(`\n   📄 Processing: ${filing.headline.substring(0, 70)}...`)
      console.log(`      📅 Date: ${filing.filingDate?.substring(0, 10)}`)
      console.log(`      📦 Size: ${(filing.size / 1024).toFixed(2)} MB`)

      // Download PDF
      console.log(`      📥 Downloading PDF...`)
      const pdfBuffer = await downloadPDF(filing.pdfUrl)

      if (!pdfBuffer) {
        console.log(`      ❌ Failed to download PDF, skipping...`)
        continue
      }

      const fileSizeMB = pdfBuffer.length / 1024 / 1024
      console.log(`      ✅ Downloaded (${fileSizeMB.toFixed(2)} MB)`)

      // Skip if file is too small (less than 1MB, unlikely to be comprehensive technical report)
      if (fileSizeMB < 1) {
        console.log(`      ⏭️  Skipping - file too small for comprehensive report`)
        continue
      }

      // Extract text from PDF
      console.log(`      📖 Extracting text from PDF...`)
      const text = await extractTextFromPDF(pdfBuffer)

      if (!text || text.length < 1000) {
        console.log(`      ⚠️  Insufficient text extracted (${text.length} chars), skipping...`)
        continue
      }

      console.log(`      ✅ Extracted ${(text.length / 1000).toFixed(1)}K characters`)

      // Extract financial data using AI BEFORE creating database entry
      const extractedData = await extractFinancialData(text, company.name, filing.headline)

      if (!extractedData) {
        console.log(`      ❌ Failed to extract data, skipping...`)
        continue
      }

      // Upload PDF to Supabase Storage
      const projectId = crypto.randomUUID()
      const fileName = `${projectId}.pdf`
      const filePath = `factset-documents/${fileName}`

      console.log(`      ☁️  Uploading to Supabase Storage...`)

      const { error: uploadError } = await supabase.storage
        .from('mining-documents')
        .upload(filePath, pdfBuffer, {
          contentType: 'application/pdf',
          upsert: true
        })

      if (uploadError) {
        console.log(`      ❌ Upload failed: ${uploadError.message}`)
        continue
      }

      // Get public URL
      const { data: { publicUrl } } = supabase.storage
        .from('mining-documents')
        .getPublicUrl(filePath)

      console.log(`      ✅ Uploaded: ${publicUrl}`)

      // Create project with EXTRACTED data
      const projectData = {
        id: projectId,
        name: extractedData.projectName || company.name,
        location: extractedData.location || 'Canada',
        stage: extractedData.stage || 'Unknown',
        commodities: extractedData.commodities || [company.primary_commodity],
        status: 'Active',
        description: `${filing.headline}. Source: FactSet SEDAR NI 43-101 Technical Report dated ${filing.filingDate?.substring(0, 10)}.\n\nReserves: ${extractedData.reserves || 'N/A'}\nResources: ${extractedData.resources || 'N/A'}\nProduction: ${extractedData.production || 'N/A'}${extractedData.mineLife ? `\nMine Life: ${extractedData.mineLife} years` : ''}`,
        urls: [publicUrl],
        npv: extractedData.npv || null,
        irr: extractedData.irr || null,
        capex: extractedData.capex || null,
        watchlist: false
      }

      const { error: insertError } = await supabase.from('projects').insert(projectData)

      if (!insertError) {
        console.log(`      ✅ Created project: ${projectData.name}`)
        console.log(`         💰 NPV: $${extractedData.npv || 'N/A'}M | IRR: ${extractedData.irr || 'N/A'}% | CAPEX: $${extractedData.capex || 'N/A'}M`)
        totalProjects++
        totalDocuments++
      } else {
        console.log(`      ❌ Failed to create project: ${insertError.message}`)
      }

      // Rate limiting to avoid API throttling
      await new Promise(resolve => setTimeout(resolve, 3000))
    }
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 SUMMARY')
  console.log('='.repeat(80))
  console.log(`✅ Projects created: ${totalProjects}`)
  console.log(`📄 Documents processed: ${totalDocuments}`)
  console.log('\n✨ All data extracted and stored successfully!')
}

processComprehensiveExtraction().catch(console.error)
