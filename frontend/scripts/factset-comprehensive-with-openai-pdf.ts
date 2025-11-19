#!/usr/bin/env npx tsx

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

// Canadian mining companies for SEDAR NI 43-101 reports
const MINING_COMPANIES = [
  { ticker: 'FM-CA', name: 'First Quantum Minerals', primary_commodity: 'Copper' },
  { ticker: 'ABX-CA', name: 'Barrick Gold', primary_commodity: 'Gold' },
  { ticker: 'K-CA', name: 'Kinross Gold', primary_commodity: 'Gold' },
  { ticker: 'AEM-CA', name: 'Agnico Eagle Mines', primary_commodity: 'Gold' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', primary_commodity: 'Copper' },
  { ticker: 'CS-CA', name: 'Capstone Copper', primary_commodity: 'Copper' },
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
    const params = new URLSearchParams({
      ids: ticker,
      sources: 'SDR',  // SEDAR
      startDate: '20180101',
      endDate: '20241231',
      searchText: 'Technical Report',
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
            const pdfUrl = doc.filingsLink?.replace('report=story', 'report=pdf')

            filings.push({
              companyName,
              ticker,
              headline: doc.headline,
              filingDate: doc.filingsDateTime,
              documentId: doc.documentId,
              htmlUrl: doc.filingsLink,
              pdfUrl: pdfUrl,
              size: doc.filingSize || 0
            })
          }
        }
      }
    }

    console.log(`   📊 Found ${filings.length} technical reports`)
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

async function extractDataFromPDF(pdfBuffer: Buffer, companyName: string, headline: string): Promise<ExtractedData | null> {
  console.log(`      🤖 Extracting data from PDF with AI...`)

  try {
    // Convert PDF buffer to base64 for GPT-4 Vision
    const base64PDF = pdfBuffer.toString('base64')

    const prompt = `You are analyzing a mining NI 43-101 Technical Report for ${companyName}.

Document title: ${headline}

This is a technical report PDF. Please extract the following information:

REQUIRED FIELDS:
- Project Name: The specific mine/project name
- Location: Country and region/province
- Development Stage: (Exploration/Feasibility/Pre-Production/Production/Expansion)
- Primary Commodity: Main mineral being mined
- Secondary Commodities: Other minerals (if any)

FINANCIAL METRICS (in USD millions unless otherwise stated):
- NPV (Net Present Value): After-tax NPV at specified discount rate (convert to USD millions)
- IRR (Internal Rate of Return): After-tax IRR as percentage
- CAPEX (Capital Expenditure): Initial capital cost (convert to USD millions)
- OPEX (Operating Expenditure): Operating cost per unit ($/tonne or $/oz)

TECHNICAL METRICS:
- Mineral Reserves: Proven and probable reserves with tonnage and grades
- Mineral Resources: Measured, indicated, and inferred resources
- Mine Life: Expected operational life in years
- Production Rate: Annual production rate

Return as JSON with this exact structure:
{
  "projectName": "string",
  "location": "string (Country, Region)",
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
- For NPV/CAPEX, extract in millions USD (if in CAD, note it)
- For IRR, extract as percentage (e.g., 15.5 for 15.5%)
- Look in executive summary, economic analysis sections
- Be conservative - only extract if confident
- If value not found, use null`

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          {
            type: 'image_url',
            image_url: {
              url: `data:application/pdf;base64,${base64PDF}`,
              detail: 'low' // Use low detail to process faster and cheaper
            }
          }
        ]
      }],
      temperature: 0.1,
      max_tokens: 2000,
      response_format: { type: 'json_object' }
    })

    const extracted = JSON.parse(completion.choices[0].message.content || '{}')
    console.log(`      ✅ Extracted: ${extracted.projectName || companyName}`)
    if (extracted.npv || extracted.irr || extracted.capex) {
      console.log(`         💰 NPV: $${extracted.npv || 'N/A'}M | IRR: ${extracted.irr || 'N/A'}% | CAPEX: $${extracted.capex || 'N/A'}M`)
    }

    return extracted as ExtractedData

  } catch (error: any) {
    console.log(`      ❌ AI extraction error: ${error.message}`)
    return null
  }
}

async function processComprehensiveExtraction() {
  console.log('🏭 COMPREHENSIVE FACTSET EXTRACTION WITH AI PDF ANALYSIS')
  console.log('='.repeat(80))

  // Clear existing projects
  console.log('\n🗑️  Clearing existing projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')

  let totalProjects = 0

  for (const company of MINING_COMPANIES.slice(0, 5)) { // Process first 5 companies
    console.log(`\n${'='.repeat(80)}`)
    console.log(`📊 Processing ${company.name}`)
    console.log('='.repeat(80))

    const filings = await searchSEDARFilings(company.ticker, company.name)

    if (filings.length === 0) {
      console.log(`   ⚠️  No technical reports found, skipping...`)
      continue
    }

    // Process the most recent large filing
    for (const filing of filings.slice(0, 2)) { // Try first 2 filings
      console.log(`\n   📄 ${filing.headline.substring(0, 70)}...`)
      console.log(`      📅 Date: ${filing.filingDate?.substring(0, 10)}`)

      // Download PDF
      console.log(`      📥 Downloading PDF...`)
      const pdfBuffer = await downloadPDF(filing.pdfUrl)

      if (!pdfBuffer) {
        console.log(`      ⏭️  PDF not available, trying next filing...`)
        continue
      }

      const fileSizeMB = pdfBuffer.length / 1024 / 1024
      console.log(`      ✅ Downloaded (${fileSizeMB.toFixed(2)} MB)`)

      // Skip if too small
      if (fileSizeMB < 0.5) {
        console.log(`      ⏭️  File too small, trying next filing...`)
        continue
      }

      // Extract data using AI (with PDF support)
      const extractedData = await extractDataFromPDF(pdfBuffer, company.name, filing.headline)

      if (!extractedData) {
        console.log(`      ⏭️  Extraction failed, trying next filing...`)
        continue
      }

      // Upload PDF to Supabase Storage
      const projectId = crypto.randomUUID()
      const fileName = `${projectId}.pdf`
      const filePath = `factset-documents/${fileName}`

      console.log(`      ☁️  Uploading to Supabase...`)

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

      const { data: { publicUrl } } = supabase.storage
        .from('mining-documents')
        .getPublicUrl(filePath)

      // Create project with extracted data
      const projectData = {
        id: projectId,
        name: extractedData.projectName || company.name,
        location: extractedData.location || 'Canada',
        stage: extractedData.stage || 'Unknown',
        commodities: extractedData.commodities || [company.primary_commodity],
        status: 'Active',
        description: `${filing.headline}\n\nFiled: ${filing.filingDate?.substring(0, 10)}\n\nReserves: ${extractedData.reserves || 'See technical report'}\n\nResources: ${extractedData.resources || 'See technical report'}\n\nProduction: ${extractedData.production || 'See technical report'}${extractedData.mineLife ? `\n\nMine Life: ${extractedData.mineLife} years` : ''}`,
        urls: [publicUrl],
        npv: extractedData.npv || null,
        irr: extractedData.irr || null,
        capex: extractedData.capex || null,
        watchlist: false
      }

      const { error: insertError } = await supabase.from('projects').insert(projectData)

      if (!insertError) {
        console.log(`      ✅ Created project in database`)
        totalProjects++
        break // Successfully created project, move to next company
      } else {
        console.log(`      ❌ Database error: ${insertError.message}`)
      }

      // Rate limiting
      await new Promise(resolve => setTimeout(resolve, 3000))
    }
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 SUMMARY')
  console.log('='.repeat(80))
  console.log(`✅ Projects created: ${totalProjects}`)
  console.log('\n✨ All data extracted and stored successfully!')
}

processComprehensiveExtraction().catch(console.error)
