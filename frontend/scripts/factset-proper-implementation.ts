#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import * as fs from 'fs'
import * as path from 'path'
import PDFParser from 'pdf-parse'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const OPENAI_API_KEY = process.env.OPENAI_API_KEY!

const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Mining companies to search
const MINING_COMPANIES = [
  { ticker: 'BHP-AU', name: 'BHP Group' },
  { ticker: 'RIO-GB', name: 'Rio Tinto' },
  { ticker: 'VALE', name: 'Vale' },
  { ticker: 'FCX', name: 'Freeport-McMoRan' },
  { ticker: 'NEM', name: 'Newmont' },
  { ticker: 'GOLD', name: 'Barrick Gold' },
  { ticker: 'SCCO', name: 'Southern Copper' },
  { ticker: 'TECK-CA', name: 'Teck Resources' },
  { ticker: 'FM-CA', name: 'First Quantum Minerals' },
  { ticker: 'ALB', name: 'Albemarle' }
]

interface FactSetDocument {
  companyName: string
  ticker: string
  formType: string
  headline: string
  filingDate: string
  documentId: string
  filingsLink: string
  filingSize: string
}

async function searchFactSetFilings(ticker: string, companyName: string): Promise<FactSetDocument[]> {
  console.log(`\n🔍 Searching FactSet filings for ${companyName} (${ticker})...`)

  try {
    const params = new URLSearchParams({
      ids: ticker,
      sources: 'EDG',  // EDGAR source
      startDate: '20230101',
      endDate: '20241231',
      formTypes: '10-K,20-F',  // Annual reports only
      _paginationLimit: '3'
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
    const documents: FactSetDocument[] = []

    if (data.data && Array.isArray(data.data)) {
      for (const result of data.data) {
        if (result.documents && Array.isArray(result.documents)) {
          console.log(`   ✅ Found ${result.documents.length} filings for ${result.requestId}`)

          for (const doc of result.documents) {
            documents.push({
              companyName,
              ticker,
              formType: doc.formTypes?.[0] || 'UNKNOWN',
              headline: doc.headline,
              filingDate: doc.filingsDateTime,
              documentId: doc.documentId,
              filingsLink: doc.filingsLink,
              filingSize: doc.filingSize
            })

            console.log(`      📄 ${doc.formTypes?.[0]} - ${doc.filingSize} - ${doc.filingsDateTime?.substring(0, 10)}`)
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

async function downloadFactSetDocument(filingsLink: string, documentId: string): Promise<Buffer | null> {
  console.log(`   📥 Downloading document ${documentId}...`)

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
    console.log(`      ✅ Downloaded ${buffer.length} bytes`)

    return buffer
  } catch (error: any) {
    console.log(`      ❌ Error: ${error.message}`)
    return null
  }
}

async function extractTextFromHTML(htmlContent: Buffer): Promise<string> {
  // Convert HTML to text (basic extraction)
  const html = htmlContent.toString('utf-8')

  // Remove HTML tags
  let text = html.replace(/<[^>]+>/g, ' ')

  // Decode HTML entities
  text = text.replace(/&nbsp;/g, ' ')
  text = text.replace(/&amp;/g, '&')
  text = text.replace(/&lt;/g, '<')
  text = text.replace(/&gt;/g, '>')

  // Clean up whitespace
  text = text.replace(/\s+/g, ' ').trim()

  return text
}

async function extractDataWithAI(text: string, companyName: string): Promise<any> {
  console.log(`      🤖 Extracting data with AI (processing ${text.length} characters)...`)

  try {
    // Take first 50000 characters to stay within API limits
    const excerpt = text.substring(0, 50000)

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
            content: 'You are a mining industry analyst extracting financial and operational data from technical reports. Extract NPV, IRR, CAPEX, project locations, and commodities.'
          },
          {
            role: 'user',
            content: `Extract mining project data from this ${companyName} SEC filing. Return JSON with: {"projects": [{"name": "project name", "location": "location", "commodities": ["commodity"], "npv": number in millions, "irr": percentage, "capex": number in millions}]}. Only include projects with concrete financial data.\n\n${excerpt}`
          }
        ],
        temperature: 0.1,
        response_format: { type: 'json_object' }
      })
    })

    if (!response.ok) {
      console.log(`      ❌ AI extraction failed (${response.status})`)
      return null
    }

    const data = await response.json()
    const extracted = JSON.parse(data.choices[0].message.content)

    console.log(`      ✅ Extracted ${extracted.projects?.length || 0} projects`)

    return extracted
  } catch (error: any) {
    console.log(`      ❌ AI Error: ${error.message}`)
    return null
  }
}

async function processFactSetDocuments() {
  console.log('🏭 FACTSET DOCUMENT PROCESSING PIPELINE')
  console.log('='.repeat(60))
  console.log('1. Search FactSet Global Filings API')
  console.log('2. Download technical documents (10-K, 20-F)')
  console.log('3. Extract project data using AI')
  console.log('4. Populate database with real extracted data')
  console.log('='.repeat(60))

  // First, clear existing projects
  console.log('\n🗑️  Clearing existing projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')

  let totalProjects = 0
  let totalDocuments = 0

  for (const company of MINING_COMPANIES.slice(0, 5)) {  // Process 5 companies
    console.log(`\n${'='.repeat(60)}`)
    console.log(`📊 Processing ${company.name}`)
    console.log('='.repeat(60))

    // Step 1: Search FactSet for filings
    const documents = await searchFactSetFilings(company.ticker, company.name)

    if (documents.length === 0) {
      console.log(`   ⚠️  No documents found, skipping...`)
      continue
    }

    // Take the most recent filing
    const doc = documents[0]
    totalDocuments++

    // Step 2: Download the document
    const content = await downloadFactSetDocument(doc.filingsLink, doc.documentId)

    if (!content) {
      console.log(`   ⚠️  Failed to download, skipping...`)
      continue
    }

    // Step 3: Extract text from HTML
    console.log(`      📝 Extracting text from document...`)
    const text = await extractTextFromHTML(content)
    console.log(`      ✅ Extracted ${text.length} characters of text`)

    // Step 4: Extract data with AI
    const extracted = await extractDataWithAI(text, company.name)

    if (!extracted || !extracted.projects || extracted.projects.length === 0) {
      console.log(`   ⚠️  No project data extracted, skipping...`)
      continue
    }

    // Step 5: Create projects in database
    console.log(`\n   💾 Creating ${extracted.projects.length} projects in database...`)

    for (const project of extracted.projects) {
      const projectData = {
        name: project.name,
        location: project.location || 'Global',
        stage: 'Production',
        commodities: project.commodities || ['Copper'],
        status: 'Active',
        description: `Extracted from ${company.name} ${doc.formType} filing dated ${doc.filingDate?.substring(0, 10)}. Source: FactSet Global Filings API.`,
        urls: [doc.filingsLink],
        npv: project.npv,
        irr: project.irr,
        capex: project.capex,
        watchlist: false
      }

      const { error } = await supabase.from('projects').insert(projectData)

      if (!error) {
        console.log(`      ✅ Created: ${project.name} (NPV: $${project.npv}M, IRR: ${project.irr}%)`)
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
  console.log(`✅ Documents processed: ${totalDocuments}`)
  console.log(`✅ Projects created: ${totalProjects}`)
  console.log(`📄 All data extracted from FactSet technical reports`)
  console.log(`🔒 Documents sourced via FactSet Global Filings API`)
  console.log('\n✨ FactSet document processing complete!')
}

// Run the pipeline
processFactSetDocuments().catch(console.error)