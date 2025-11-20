#!/usr/bin/env npx tsx
/**
 * FactSet SEDAR NI 43-101 Technical Report Extraction Pipeline
 *
 * This script:
 * 1. Searches FactSet Global Filings API for NI 43-101 technical reports
 * 2. Downloads large PDF documents (>2MB = comprehensive reports)
 * 3. Uploads PDFs to Supabase Storage
 * 4. Creates company records in companies table
 * 5. Creates project records with PDF URLs
 * 6. Automatically extracts financial data via API
 * 7. Updates projects with extracted NPV, IRR, CAPEX
 *
 * Usage:
 *   npx tsx scripts/factset-sedar-extraction-pipeline.ts
 *
 * Environment variables required:
 *   - NEXT_PUBLIC_SUPABASE_URL
 *   - SUPABASE_SERVICE_ROLE_KEY
 *   - FACTSET_USERNAME
 *   - FACTSET_API_KEY
 */

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Canadian mining companies with SEDAR filings
const MINING_COMPANIES = [
  { ticker: 'FM-CA', name: 'First Quantum Minerals', commodity: 'Copper', country: 'Canada' },
  { ticker: 'ABX-CA', name: 'Barrick Gold', commodity: 'Gold', country: 'Canada' },
  { ticker: 'K-CA', name: 'Kinross Gold', commodity: 'Gold', country: 'Canada' },
  { ticker: 'AEM-CA', name: 'Agnico Eagle Mines', commodity: 'Gold', country: 'Canada' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', commodity: 'Copper', country: 'Canada' },
  { ticker: 'CS-CA', name: 'Capstone Copper', commodity: 'Copper', country: 'Canada' },
  { ticker: 'IMG-CA', name: 'IAMGOLD', commodity: 'Gold', country: 'Canada' },
  { ticker: 'TKO-CA', name: 'Taseko Mines', commodity: 'Copper', country: 'Canada' },
  { ticker: 'NGD-CA', name: 'New Gold', commodity: 'Gold', country: 'Canada' },
  { ticker: 'LUN-CA', name: 'Lundin Mining', commodity: 'Copper', country: 'Canada' },
  { ticker: 'ELD-CA', name: 'Eldorado Gold', commodity: 'Gold', country: 'Canada' },
  { ticker: 'ERO-CA', name: 'Ero Copper', commodity: 'Copper', country: 'Canada' },
  { ticker: 'SMT-CA', name: 'Sierra Metals', commodity: 'Copper', country: 'Canada' }
]

interface Filing {
  companyName: string
  ticker: string
  headline: string
  filingDate: string
  documentId: string
  pdfUrl: string
  size: number
}

async function ensureCompanyExists(company: typeof MINING_COMPANIES[0]): Promise<string> {
  // Check if company exists
  const { data: existing } = await supabase
    .from('companies')
    .select('id')
    .eq('ticker_symbol', company.ticker)
    .single()

  if (existing) {
    return existing.id
  }

  // Create new company
  const companyData = {
    id: crypto.randomUUID(),
    name: company.name,
    ticker_symbol: company.ticker,
    primary_commodity: company.commodity,
    headquarters_country: company.country
  }

  const { data, error } = await supabase
    .from('companies')
    .insert(companyData)
    .select('id')
    .single()

  if (error) {
    console.log(`      ⚠️  Company creation failed: ${error.message}`)
    return companyData.id
  }

  console.log(`      ✅ Created company record`)
  return data.id
}

async function searchSEDARFilings(ticker: string, companyName: string): Promise<Filing[]> {
  try {
    const params = new URLSearchParams({
      ids: ticker,
      sources: 'SDR',  // SEDAR
      startDate: '20150101',
      endDate: '20241231',
      searchText: 'Technical Report',
      _paginationLimit: '50'
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
      return []
    }

    const data = await response.json()
    const filings: Filing[] = []

    if (data.data && Array.isArray(data.data)) {
      for (const result of data.data) {
        if (result.documents && Array.isArray(result.documents)) {
          for (const doc of result.documents) {
            filings.push({
              companyName,
              ticker,
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

    return filings
  } catch (error: any) {
    console.log(`      ❌ Search error: ${error.message}`)
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

    if (!response.ok) return null

    const buffer = Buffer.from(await response.arrayBuffer())

    // Validate PDF header
    if (buffer.slice(0, 5).toString() === '%PDF-') {
      return buffer
    }

    return null
  } catch {
    return null
  }
}

async function extractDataViaAPI(pdfUrl: string, projectId: string): Promise<boolean> {
  try {
    console.log(`      🤖 Extracting financial data...`)

    const apiUrl = `http://localhost:3000/api/pdf/extract-highlights?pdfUrl=${encodeURIComponent(pdfUrl)}`

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      // Long timeout for large PDFs
      signal: AbortSignal.timeout(120000)
    })

    if (!response.ok) {
      console.log(`      ⚠️  Extraction API returned ${response.status}`)
      return false
    }

    const result = await response.json()

    if (result.updated && result.updated.length > 0) {
      const updated = result.updated[0]
      console.log(`      ✅ Financial data extracted:`)
      if (updated.npv) console.log(`         💰 NPV: $${updated.npv}M`)
      if (updated.irr) console.log(`         📊 IRR: ${updated.irr}%`)
      if (updated.capex) console.log(`         💵 CAPEX: $${updated.capex}M`)
      return true
    }

    console.log(`      ℹ️  No financial metrics found`)
    return false

  } catch (error: any) {
    console.log(`      ⚠️  Extraction timeout/error: ${error.message}`)
    return false
  }
}

async function run() {
  console.log('🏭 FACTSET SEDAR NI 43-101 EXTRACTION PIPELINE')
  console.log('='.repeat(80))
  console.log('📋 Configuration:')
  console.log(`   Companies: ${MINING_COMPANIES.length}`)
  console.log(`   Min PDF size: 2MB (ensures comprehensive reports)`)
  console.log(`   Max docs per company: 3`)
  console.log(`   Source: FactSet SEDAR`)
  console.log(`   Years: 2015-2024`)
  console.log('='.repeat(80))

  let totalProjects = 0
  let totalWithData = 0

  for (const company of MINING_COMPANIES) {
    console.log(`\n${'='.repeat(80)}`)
    console.log(`📊 ${company.name} (${company.ticker})`)
    console.log('='.repeat(80))

    // Ensure company exists in database
    const companyId = await ensureCompanyExists(company)

    // Search for filings
    console.log(`   🔍 Searching for technical reports...`)
    const filings = await searchSEDARFilings(company.ticker, company.name)
    console.log(`   📄 Found ${filings.length} documents`)

    if (filings.length === 0) continue

    // Process up to 3 large reports per company
    let processed = 0

    for (const filing of filings) {
      if (processed >= 3) break

      // Download PDF
      const pdfBuffer = await downloadPDF(filing.pdfUrl)
      if (!pdfBuffer) continue

      const sizeMB = pdfBuffer.length / 1024 / 1024

      // Only process large comprehensive reports
      if (sizeMB < 2) continue

      console.log(`\n   📄 ${filing.headline.substring(0, 60)}...`)
      console.log(`      📅 ${filing.filingDate?.substring(0, 10)}`)
      console.log(`      📦 ${sizeMB.toFixed(1)} MB`)

      // Upload to Supabase
      const projectId = crypto.randomUUID()
      const filePath = `factset-documents/${projectId}.pdf`

      const { error: uploadError } = await supabase.storage
        .from('mining-documents')
        .upload(filePath, pdfBuffer, {
          contentType: 'application/pdf',
          upsert: true
        })

      if (uploadError) {
        console.log(`      ❌ Upload failed`)
        continue
      }

      const { data: { publicUrl } } = supabase.storage
        .from('mining-documents')
        .getPublicUrl(filePath)

      // Create project
      const projectData = {
        id: projectId,
        company_id: companyId,
        name: company.name,
        location: 'Canada',
        stage: 'Unknown',
        commodities: [company.commodity],
        status: 'Active',
        description: `${filing.headline}\n\nFiled: ${filing.filingDate?.substring(0, 10)}\nSource: FactSet SEDAR NI 43-101 Technical Report\nDocument Size: ${sizeMB.toFixed(1)} MB`,
        urls: [publicUrl],
        npv: null,
        irr: null,
        capex: null,
        watchlist: false
      }

      const { error: insertError } = await supabase
        .from('projects')
        .insert(projectData)

      if (insertError) {
        console.log(`      ❌ DB insert failed`)
        continue
      }

      console.log(`      ✅ Project created`)
      totalProjects++

      // Extract financial data
      await new Promise(resolve => setTimeout(resolve, 2000))

      const extracted = await extractDataViaAPI(publicUrl, projectId)
      if (extracted) {
        totalWithData++
      }

      processed++

      // Rate limiting
      await new Promise(resolve => setTimeout(resolve, 3000))
    }

    // Pause between companies
    await new Promise(resolve => setTimeout(resolve, 2000))
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 PIPELINE SUMMARY')
  console.log('='.repeat(80))
  console.log(`✅ Total projects created: ${totalProjects}`)
  console.log(`✅ Projects with financial data: ${totalWithData}`)
  console.log(`📈 Success rate: ${totalProjects > 0 ? ((totalWithData / totalProjects) * 100).toFixed(1) : 0}%`)
  console.log('\n✨ Pipeline complete!')
}

run().catch(console.error)
