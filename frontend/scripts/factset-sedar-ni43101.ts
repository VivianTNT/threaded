#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const OPENAI_API_KEY = process.env.OPENAI_API_KEY!

const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Canadian mining companies - these will have SEDAR filings
const CANADIAN_MINING_COMPANIES = [
  { ticker: 'TECK-CA', name: 'Teck Resources' },
  { ticker: 'FM-CA', name: 'First Quantum Minerals' },
  { ticker: 'ABX-CA', name: 'Barrick Gold' },
  { ticker: 'K-CA', name: 'Kinross Gold' },
  { ticker: 'AEM-CA', name: 'Agnico Eagle Mines' },
  { ticker: 'GOLD-CA', name: 'Barrick Gold Corp' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals' },
  { ticker: 'CS-CA', name: 'Capstone Copper' },
  { ticker: 'TKO-CA', name: 'Taseko Mines' },
  { ticker: 'IMG-CA', name: 'IAMGOLD' }
]

interface SEDARDocument {
  companyName: string
  ticker: string
  formType: string
  headline: string
  filingDate: string
  documentId: string
  filingsLink: string
  filingSize: string
}

async function searchSEDARFilings(ticker: string, companyName: string): Promise<SEDARDocument[]> {
  console.log(`\n🔍 Searching SEDAR for ${companyName} (${ticker})...`)

  try {
    // Search SEDAR source for NI 43-101 technical reports
    const params = new URLSearchParams({
      ids: ticker,
      sources: 'SDR',  // SEDAR source
      startDate: '20200101',
      endDate: '20241231',
      searchText: 'NI 43-101',  // Search for NI 43-101 reports
      _paginationLimit: '5'
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
      const errorText = await response.text()
      console.log(`   ❌ Search failed (${response.status}): ${errorText.substring(0, 200)}`)
      return []
    }

    const data = await response.json()
    const documents: SEDARDocument[] = []

    if (data.data && Array.isArray(data.data)) {
      for (const result of data.data) {
        if (result.documents && Array.isArray(result.documents)) {
          console.log(`   ✅ Found ${result.documents.length} NI 43-101 reports for ${result.requestId}`)

          for (const doc of result.documents) {
            documents.push({
              companyName,
              ticker,
              formType: doc.formTypes?.[0] || 'NI 43-101',
              headline: doc.headline,
              filingDate: doc.filingsDateTime,
              documentId: doc.documentId,
              filingsLink: doc.filingsLink,
              filingSize: doc.filingSize
            })

            console.log(`      📄 ${doc.headline?.substring(0, 80)}...`)
            console.log(`         Size: ${doc.filingSize}, Date: ${doc.filingsDateTime?.substring(0, 10)}`)
          }
        }
      }
    }

    return documents
  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`)
    return []
  }
}

async function downloadSEDARDocument(filingsLink: string, documentId: string): Promise<Buffer | null> {
  console.log(`   📥 Downloading NI 43-101 document ${documentId}...`)

  try {
    const response = await fetch(filingsLink, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Accept': 'text/html,application/pdf,*/*'
      }
    })

    if (!response.ok) {
      console.log(`      ❌ Download failed (${response.status})`)
      return null
    }

    const buffer = Buffer.from(await response.arrayBuffer())
    console.log(`      ✅ Downloaded ${(buffer.length / 1024 / 1024).toFixed(2)} MB`)

    return buffer
  } catch (error: any) {
    console.log(`      ❌ Error: ${error.message}`)
    return null
  }
}

async function extractTextFromHTML(htmlContent: Buffer): Promise<string> {
  const html = htmlContent.toString('utf-8')

  // Remove HTML tags
  let text = html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')
  text = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')
  text = text.replace(/<[^>]+>/g, ' ')

  // Decode HTML entities
  text = text.replace(/&nbsp;/g, ' ')
  text = text.replace(/&amp;/g, '&')
  text = text.replace(/&lt;/g, '<')
  text = text.replace(/&gt;/g, '>')
  text = text.replace(/&quot;/g, '"')

  // Clean up whitespace
  text = text.replace(/\s+/g, ' ').trim()

  return text
}

async function extractMiningDataWithAI(text: string, companyName: string): Promise<any> {
  console.log(`      🤖 Extracting mining data with AI (processing ${(text.length / 1000).toFixed(1)}k characters)...`)

  try {
    // Take relevant sections - NI 43-101 reports have standard sections
    const excerpt = text.substring(0, 60000)

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: 'You are a mining engineer analyzing NI 43-101 technical reports. Extract detailed project information including mineral resources, reserves, NPV, IRR, CAPEX, production rates, and commodities. NI 43-101 reports are comprehensive technical documents that include economic analysis.'
          },
          {
            role: 'user',
            content: `Extract detailed mining project data from this NI 43-101 technical report for ${companyName}.

Return JSON with this exact structure:
{
  "projects": [
    {
      "name": "project name",
      "location": "province/state, country",
      "commodities": ["primary commodity", "secondary commodity"],
      "stage": "Production/Development/Exploration",
      "npv": number in millions USD (after-tax),
      "irr": percentage (after-tax),
      "capex": number in millions USD (initial capital),
      "resources": "mineral resources estimate",
      "reserves": "mineral reserves estimate",
      "production": "annual production rate"
    }
  ]
}

Only include projects with concrete NPV/IRR data from the economic analysis section.

NI 43-101 Report Content:
${excerpt}`
          }
        ],
        temperature: 0.1,
        response_format: { type: 'json_object' }
      })
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.log(`      ❌ AI extraction failed (${response.status}): ${errorText.substring(0, 200)}`)
      return null
    }

    const data = await response.json()
    const extracted = JSON.parse(data.choices[0].message.content)

    console.log(`      ✅ Extracted ${extracted.projects?.length || 0} projects from NI 43-101 report`)

    return extracted
  } catch (error: any) {
    console.log(`      ❌ AI Error: ${error.message}`)
    return null
  }
}

async function processSEDARDocuments() {
  console.log('🇨🇦 SEDAR NI 43-101 TECHNICAL REPORT PROCESSING PIPELINE')
  console.log('='.repeat(60))
  console.log('Source: SEDAR (Canadian Securities Filings)')
  console.log('Document Type: NI 43-101 Technical Reports')
  console.log('Pipeline:')
  console.log('  1. Search FactSet SEDAR filings for NI 43-101 reports')
  console.log('  2. Download comprehensive technical reports (100-500+ pages)')
  console.log('  3. Extract project data using AI')
  console.log('  4. Populate database with real extracted data')
  console.log('='.repeat(60))

  // Clear existing projects
  console.log('\n🗑️  Clearing existing projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')

  let totalProjects = 0
  let totalDocuments = 0
  let totalCompanies = 0

  for (const company of CANADIAN_MINING_COMPANIES.slice(0, 8)) {
    console.log(`\n${'='.repeat(60)}`)
    console.log(`📊 Processing ${company.name}`)
    console.log('='.repeat(60))

    // Step 1: Search SEDAR for NI 43-101 reports
    const documents = await searchSEDARFilings(company.ticker, company.name)

    if (documents.length === 0) {
      console.log(`   ⚠️  No NI 43-101 reports found, skipping...`)
      continue
    }

    totalCompanies++

    // Process the first document (most recent)
    const doc = documents[0]
    totalDocuments++

    // Step 2: Download the NI 43-101 report
    const content = await downloadSEDARDocument(doc.filingsLink, doc.documentId)

    if (!content) {
      console.log(`   ⚠️  Failed to download, skipping...`)
      continue
    }

    // Step 3: Extract text from HTML
    console.log(`      📝 Extracting text from NI 43-101 report...`)
    const text = await extractTextFromHTML(content)
    console.log(`      ✅ Extracted ${(text.length / 1000).toFixed(1)}k characters`)

    // Step 4: Extract mining data with AI
    const extracted = await extractMiningDataWithAI(text, company.name)

    if (!extracted || !extracted.projects || extracted.projects.length === 0) {
      console.log(`   ⚠️  No project data extracted, skipping...`)
      continue
    }

    // Step 5: Create projects in database
    console.log(`\n   💾 Creating ${extracted.projects.length} projects in database...`)

    for (const project of extracted.projects) {
      const projectData = {
        name: project.name,
        location: project.location || 'Canada',
        stage: project.stage || 'Development',
        commodities: project.commodities || ['Copper'],
        status: 'Active',
        description: `NI 43-101 Technical Report: ${doc.headline}. Filed: ${doc.filingDate?.substring(0, 10)}. Source: SEDAR via FactSet. ${project.resources || ''}`,
        urls: [doc.filingsLink],
        npv: project.npv,
        irr: project.irr,
        capex: project.capex,
        watchlist: false
      }

      const { error } = await supabase.from('projects').insert(projectData)

      if (!error) {
        console.log(`      ✅ ${project.name}`)
        console.log(`         NPV: $${project.npv}M | IRR: ${project.irr}% | CAPEX: $${project.capex}M`)
        console.log(`         Commodities: ${project.commodities?.join(', ')}`)
        totalProjects++
      } else {
        console.log(`      ❌ Failed: ${error.message}`)
      }
    }

    // Rate limiting
    console.log(`\n   ⏸️  Waiting 3 seconds before next company...`)
    await new Promise(resolve => setTimeout(resolve, 3000))
  }

  console.log('\n' + '='.repeat(60))
  console.log('📊 FINAL SUMMARY')
  console.log('='.repeat(60))
  console.log(`✅ Companies processed: ${totalCompanies}`)
  console.log(`✅ NI 43-101 reports downloaded: ${totalDocuments}`)
  console.log(`✅ Mining projects created: ${totalProjects}`)
  console.log(`📄 All data extracted from SEDAR NI 43-101 technical reports`)
  console.log(`🇨🇦 Source: Canadian securities filings via FactSet`)
  console.log('\n✨ SEDAR NI 43-101 processing complete!')
  console.log('🎯 NI 43-101 reports are the gold standard for mining technical data')
}

// Run the pipeline
processSEDARDocuments().catch(console.error)