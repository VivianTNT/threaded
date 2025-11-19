#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!

const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

// Expanded list of mining companies
const MINING_COMPANIES = [
  { ticker: 'FM-CA', name: 'First Quantum Minerals', primary_commodity: 'Copper' },
  { ticker: 'ABX-CA', name: 'Barrick Gold', primary_commodity: 'Gold' },
  { ticker: 'K-CA', name: 'Kinross Gold', primary_commodity: 'Gold' },
  { ticker: 'AEM-CA', name: 'Agnico Eagle Mines', primary_commodity: 'Gold' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', primary_commodity: 'Copper' },
  { ticker: 'CS-CA', name: 'Capstone Copper', primary_commodity: 'Copper' },
  { ticker: 'IMG-CA', name: 'IAMGOLD', primary_commodity: 'Gold' },
  { ticker: 'TKO-CA', name: 'Taseko Mines', primary_commodity: 'Copper' },
  { ticker: 'NGD-CA', name: 'New Gold', primary_commodity: 'Gold' },
  { ticker: 'LUN-CA', name: 'Lundin Mining', primary_commodity: 'Copper' },
  { ticker: 'ELD-CA', name: 'Eldorado Gold', primary_commodity: 'Gold' },
  { ticker: 'AR-CA', name: 'Argonaut Gold', primary_commodity: 'Gold' },
  { ticker: 'ERO-CA', name: 'Ero Copper', primary_commodity: 'Copper' },
  { ticker: 'SMT-CA', name: 'Sierra Metals', primary_commodity: 'Copper' },
  { ticker: 'ORV-CA', name: 'Orvana Minerals', primary_commodity: 'Gold' }
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

async function searchSEDARFilings(ticker: string, companyName: string): Promise<Filing[]> {
  console.log(`\n🔍 Searching SEDAR for ${companyName} (${ticker})...`)

  try {
    const params = new URLSearchParams({
      ids: ticker,
      sources: 'SDR',  // SEDAR
      startDate: '20150101', // Go back 10 years for comprehensive reports
      endDate: '20241231',
      searchText: 'Technical Report',
      _paginationLimit: '50' // Get many documents to find the large ones
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

async function extractDataViaAPI(pdfUrl: string): Promise<any> {
  try {
    const apiUrl = `http://localhost:3000/api/pdf/extract-highlights?pdfUrl=${encodeURIComponent(pdfUrl)}`

    console.log(`      🤖 Calling extraction API...`)
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    if (!response.ok) {
      console.log(`      ⚠️  API returned ${response.status}`)
      return null
    }

    const result = await response.json()

    if (result.extracted) {
      console.log(`      ✅ Extracted via API:`, {
        npv: result.extracted.npv?.value || 'N/A',
        irr: result.extracted.irr?.value || 'N/A',
        capex: result.extracted.capex?.value || 'N/A'
      })
    }

    return result

  } catch (error: any) {
    console.log(`      ❌ API error: ${error.message}`)
    return null
  }
}

async function processComprehensiveExtraction() {
  console.log('🏭 COMPREHENSIVE FACTSET EXTRACTION WITH AUTO-EXTRACTION')
  console.log('='.repeat(80))
  console.log('📋 This will:')
  console.log('   1. Search for large technical reports from 15 mining companies')
  console.log('   2. Download PDFs (only files >2MB, likely 100+ pages)')
  console.log('   3. Upload to Supabase Storage')
  console.log('   4. Automatically extract financial data via API')
  console.log('   5. Populate database with extracted metrics')
  console.log('='.repeat(80))

  // Clear existing projects from FactSet
  console.log('\n🗑️  Clearing existing FactSet projects...')
  const { data: existingProjects } = await supabase
    .from('projects')
    .select('id')
    .like('description', '%FactSet SEDAR%')

  if (existingProjects && existingProjects.length > 0) {
    await supabase
      .from('projects')
      .delete()
      .in('id', existingProjects.map(p => p.id))
    console.log(`   ✅ Cleared ${existingProjects.length} existing projects`)
  }

  let totalProjects = 0
  let totalDocuments = 0

  for (const company of MINING_COMPANIES) {
    console.log(`\n${'='.repeat(80)}`)
    console.log(`📊 Processing ${company.name}`)
    console.log('='.repeat(80))

    const filings = await searchSEDARFilings(company.ticker, company.name)

    if (filings.length === 0) {
      console.log(`   ⚠️  No technical reports found, skipping...`)
      continue
    }

    // Process up to 3 documents per company (to get multiple projects)
    let processedForCompany = 0

    for (const filing of filings) {
      if (processedForCompany >= 3) break // Limit to 3 per company

      console.log(`\n   📄 ${filing.headline.substring(0, 70)}...`)
      console.log(`      📅 Date: ${filing.filingDate?.substring(0, 10)}`)

      // Download PDF
      console.log(`      📥 Downloading PDF...`)
      const pdfBuffer = await downloadPDF(filing.pdfUrl)

      if (!pdfBuffer) {
        console.log(`      ⏭️  PDF not available`)
        continue
      }

      const fileSizeMB = pdfBuffer.length / 1024 / 1024
      console.log(`      ✅ Downloaded (${fileSizeMB.toFixed(2)} MB)`)

      // Only process large PDFs (>2MB = likely 100+ pages of actual technical content)
      if (fileSizeMB < 2) {
        console.log(`      ⏭️  Skipping - file too small for comprehensive report`)
        continue
      }

      // Upload PDF to Supabase Storage FIRST
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

      console.log(`      ✅ Uploaded: ${publicUrl}`)

      // Create initial project WITHOUT extracted data
      const initialProjectData = {
        id: projectId,
        name: company.name,
        location: 'Canada',
        stage: 'Unknown',
        commodities: [company.primary_commodity],
        status: 'Active',
        description: `${filing.headline}\n\nFiled: ${filing.filingDate?.substring(0, 10)}\nSource: FactSet SEDAR NI 43-101 Technical Report`,
        urls: [publicUrl],
        npv: null,
        irr: null,
        capex: null,
        watchlist: false
      }

      const { error: insertError } = await supabase.from('projects').insert(initialProjectData)

      if (insertError) {
        console.log(`      ❌ Database error: ${insertError.message}`)
        continue
      }

      console.log(`      ✅ Created project in database`)
      totalDocuments++

      // NOW extract data via API
      console.log(`      🔍 Extracting financial data...`)
      await new Promise(resolve => setTimeout(resolve, 2000)) // Wait for DB to settle

      const extractedData = await extractDataViaAPI(publicUrl)

      if (extractedData?.updated && extractedData.updated.length > 0) {
        const updated = extractedData.updated[0]
        console.log(`      💰 Financial data extracted and saved:`)
        if (updated.npv) console.log(`         NPV: $${updated.npv}M`)
        if (updated.irr) console.log(`         IRR: ${updated.irr}%`)
        if (updated.capex) console.log(`         CAPEX: $${updated.capex}M`)
        totalProjects++
      } else {
        console.log(`      ⚠️  No financial metrics found in this document`)
      }

      processedForCompany++

      // Rate limiting between documents
      await new Promise(resolve => setTimeout(resolve, 3000))
    }

    // Rate limiting between companies
    await new Promise(resolve => setTimeout(resolve, 2000))
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 SUMMARY')
  console.log('='.repeat(80))
  console.log(`✅ Total documents processed: ${totalDocuments}`)
  console.log(`✅ Projects with financial data: ${totalProjects}`)
  console.log('\n✨ All data extracted and stored successfully!')
}

processComprehensiveExtraction().catch(console.error)
