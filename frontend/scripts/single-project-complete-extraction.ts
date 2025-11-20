#!/usr/bin/env npx tsx

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import path from 'path'
import { getDocument } from 'unpdf'

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

interface FactSetFiling {
  filingId: string
  headline: string
  filingLink: string
  filingSize: number
  filingDate: string
}

interface ExtractedFinancials {
  projectName: string
  npv: number | null
  irr: number | null
  capex: number | null
  opex: number | null
  mineLife: number | null
  annualProduction: number | null
  resource: string | null
  reserve: string | null
  stage: string | null
  description: string | null
}

async function searchFactSetFilings(ticker: string): Promise<FactSetFiling[]> {
  const url = 'https://api.factset.com/global-filings/v1/search'

  const body = {
    meta: {
      schema: 1,
      pagination: {
        isLastPage: false,
        limit: 50
      }
    },
    data: {
      searchCriteria: {
        searchText: ticker,
        sources: ['SEDAR', 'SEDAR_PLUS', 'SEC'],
        categories: ['Reports', 'Financial Statements']
      }
    }
  }

  console.log(`🔍 Searching FactSet for ticker: ${ticker}`)

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': authHeader,
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    body: JSON.stringify(body)
  })

  if (!response.ok) {
    throw new Error(`FactSet API error: ${response.status} ${response.statusText}`)
  }

  const data = await response.json()

  if (!data.data?.searchResults) {
    console.log('   No results found')
    return []
  }

  const filings: FactSetFiling[] = data.data.searchResults.map((result: any) => ({
    filingId: result.filingId,
    headline: result.headline,
    filingLink: result.filingLink,
    filingSize: result.filingSize || 0,
    filingDate: result.filingDate
  }))

  // Filter for large documents (100+ pages = ~500KB+)
  const largeDocs = filings.filter(f => f.filingSize >= 500000)

  console.log(`   Found ${filings.length} total filings, ${largeDocs.length} large documents (>500KB)`)

  return largeDocs.slice(0, 3) // Take top 3 largest documents
}

async function downloadDocument(filingLink: string): Promise<Buffer | null> {
  try {
    console.log('   ⬇️  Downloading document...')

    // Try to get PDF version
    const pdfUrl = filingLink.includes('report=')
      ? filingLink.replace(/report=\w+/, 'report=pdf')
      : filingLink + '&report=pdf'

    const response = await fetch(pdfUrl, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf'
      }
    })

    if (!response.ok) {
      console.log(`   ⚠️  PDF download failed (${response.status}), trying HTML...`)

      // Try HTML version
      const htmlResponse = await fetch(filingLink, {
        method: 'GET',
        headers: {
          'Authorization': authHeader,
          'Accept': 'text/html'
        }
      })

      if (!htmlResponse.ok) return null

      const html = await htmlResponse.text()
      return Buffer.from(html, 'utf-8')
    }

    const buffer = Buffer.from(await response.arrayBuffer())

    // Verify it's a PDF
    if (buffer.slice(0, 5).toString() === '%PDF-') {
      console.log(`   ✅ Downloaded PDF (${(buffer.length / 1024).toFixed(2)} KB)`)
      return buffer
    }

    console.log('   ⚠️  Response not a valid PDF')
    return null
  } catch (error) {
    console.log(`   ❌ Download error: ${error}`)
    return null
  }
}

async function uploadToSupabase(buffer: Buffer, filename: string): Promise<string | null> {
  try {
    console.log('   ☁️  Uploading to Supabase Storage...')

    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(filename, buffer, {
        contentType: filename.endsWith('.pdf') ? 'application/pdf' : 'text/html',
        upsert: true
      })

    if (error) {
      console.log(`   ❌ Upload error: ${error.message}`)
      return null
    }

    // Get public URL
    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(filename)

    console.log(`   ✅ Uploaded to: ${urlData.publicUrl}`)
    return urlData.publicUrl
  } catch (error) {
    console.log(`   ❌ Upload failed: ${error}`)
    return null
  }
}

async function extractTextFromPDF(buffer: Buffer): Promise<string> {
  try {
    console.log('   📖 Extracting text from PDF...')

    const pdf = await getDocument(buffer).promise
    let fullText = ''

    // Extract first 50 pages (enough for financial data)
    const maxPages = Math.min(pdf.numPages, 50)

    for (let i = 1; i <= maxPages; i++) {
      const page = await pdf.getPage(i)
      const textContent = await page.getTextContent()
      const pageText = textContent.items.map((item: any) => item.str).join(' ')
      fullText += pageText + '\n'
    }

    console.log(`   ✅ Extracted ${fullText.length} characters from ${maxPages} pages`)
    return fullText
  } catch (error) {
    console.log(`   ⚠️  PDF extraction failed: ${error}`)
    return ''
  }
}

async function extractFinancialsWithAI(text: string, headline: string, companyName: string): Promise<ExtractedFinancials> {
  try {
    console.log('   🤖 Extracting financial data with OpenAI...')

    const prompt = `You are a mining project financial analyst. Extract the following information from this technical report.

Company: ${companyName}
Document: ${headline}

Extract:
1. PROJECT NAME: The specific mine/project name (not just company name)
2. NPV (Net Present Value) in USD millions
3. IRR (Internal Rate of Return) as percentage
4. CAPEX (Capital Expenditure) in USD millions
5. OPEX (Operating Expenditure) in USD per tonne
6. Mine Life in years
7. Annual Production in tonnes
8. Total Resource estimate (e.g., "50M tonnes @ 1.2% grade")
9. Total Reserve estimate (e.g., "30M tonnes @ 1.1% grade")
10. Project Stage (Exploration, Pre-Feasibility, Feasibility, Development, Production)
11. Brief project description (2-3 sentences)

Document text (first 15000 characters):
${text.substring(0, 15000)}

Return ONLY valid JSON with this exact structure:
{
  "projectName": "string",
  "npv": number or null,
  "irr": number or null,
  "capex": number or null,
  "opex": number or null,
  "mineLife": number or null,
  "annualProduction": number or null,
  "resource": "string or null",
  "reserve": "string or null",
  "stage": "string or null",
  "description": "string or null"
}

For NPV and CAPEX: Extract the POST-TAX values in USD millions.
For IRR: Extract as percentage (e.g., 15.5 for 15.5%)
For OPEX: Extract operating cost per tonne in USD
For mine life: Extract in years
For production: Extract annual production in tonnes

If a value is not found, use null. Do not make up values.`

    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      response_format: { type: 'json_object' }
    })

    const result = JSON.parse(completion.choices[0].message.content!)
    console.log('   ✅ Extracted financial data')
    console.log('      - Project:', result.projectName)
    console.log('      - NPV: $' + (result.npv ? `${result.npv}M` : 'N/A'))
    console.log('      - IRR:', result.irr ? `${result.irr}%` : 'N/A')
    console.log('      - CAPEX: $' + (result.capex ? `${result.capex}M` : 'N/A'))
    console.log('      - OPEX: $' + (result.opex ? `${result.opex}/t` : 'N/A'))

    return result
  } catch (error) {
    console.log(`   ❌ AI extraction failed: ${error}`)
    // Return default values
    return {
      projectName: `${companyName} - ${headline.substring(0, 50)}`,
      npv: null,
      irr: null,
      capex: null,
      opex: null,
      mineLife: null,
      annualProduction: null,
      resource: null,
      reserve: null,
      stage: null,
      description: null
    }
  }
}

async function createProject(
  companyId: number,
  companyName: string,
  storageUrl: string,
  financials: ExtractedFinancials
): Promise<boolean> {
  try {
    console.log('   💾 Creating project in database...')

    const { data, error } = await supabase
      .from('projects')
      .insert({
        company_id: companyId,
        name: financials.projectName,
        company_name: companyName,
        document_storage_path: storageUrl,
        npv: financials.npv,
        irr: financials.irr,
        capex: financials.capex,
        resource: financials.resource,
        reserve: financials.reserve,
        stage: financials.stage || 'Feasibility',
        description: financials.description,
        urls: [storageUrl],
        status: 'Active',
        financial_metrics_updated_at: new Date().toISOString()
      })
      .select()

    if (error) {
      console.log(`   ❌ Database error: ${error.message}`)
      return false
    }

    console.log('   ✅ Project created successfully!')
    return true
  } catch (error) {
    console.log(`   ❌ Failed to create project: ${error}`)
    return false
  }
}

async function processOneCompany() {
  console.log('🚀 Starting single project extraction (DEMO)\n')

  // Get first Canadian mining company
  const { data: companies, error: companyError } = await supabase
    .from('companies')
    .select('id, name, ticker, country')
    .eq('country', 'Canada')
    .not('ticker', 'is', null)
    .limit(1)

  if (companyError || !companies || companies.length === 0) {
    console.error('❌ No companies found')
    return
  }

  const company = companies[0]
  console.log(`\n📊 Company: ${company.name} (${company.ticker})`)
  console.log('─'.repeat(80))

  // Search FactSet
  const filings = await searchFactSetFilings(company.ticker)

  if (filings.length === 0) {
    console.log('❌ No large technical documents found')
    return
  }

  // Process first filing
  const filing = filings[0]
  console.log(`\n📄 Processing: ${filing.headline}`)
  console.log(`   Date: ${filing.filingDate}`)
  console.log(`   Size: ${(filing.filingSize / 1024).toFixed(2)} KB`)

  // Download document
  const buffer = await downloadDocument(filing.filingLink)
  if (!buffer) {
    console.log('❌ Failed to download document')
    return
  }

  // Upload to Supabase
  const filename = `${company.ticker}_${filing.filingId}.pdf`
  const storageUrl = await uploadToSupabase(buffer, filename)
  if (!storageUrl) {
    console.log('❌ Failed to upload to storage')
    return
  }

  // Extract text
  const text = await extractTextFromPDF(buffer)
  if (!text) {
    console.log('❌ Failed to extract text')
    return
  }

  // Extract financials with AI
  const financials = await extractFinancialsWithAI(text, filing.headline, company.name)

  // Create project
  const success = await createProject(company.id, company.name, storageUrl, financials)

  if (success) {
    console.log('\n✅ COMPLETE! Successfully processed one project.')
    console.log('\n📋 SUMMARY:')
    console.log('   Company:', company.name)
    console.log('   Project:', financials.projectName)
    console.log('   Document URL:', storageUrl)
    console.log('   NPV: $' + (financials.npv ? `${financials.npv}M` : 'N/A'))
    console.log('   IRR:', financials.irr ? `${financials.irr}%` : 'N/A')
    console.log('   CAPEX: $' + (financials.capex ? `${financials.capex}M` : 'N/A'))
  } else {
    console.log('\n❌ Failed to create project')
  }
}

processOneCompany()
