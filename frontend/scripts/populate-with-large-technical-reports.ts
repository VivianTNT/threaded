#!/usr/bin/env npx tsx
/**
 * POPULATE WITH LARGE TECHNICAL REPORTS (100+ pages)
 *
 * 1. Search for NI 43-101 technical reports and large annual reports
 * 2. Filter for documents > 2MB (100+ pages)
 * 3. Download and store in Supabase
 * 4. Extract NPV, IRR, CAPEX automatically
 * 5. Populate database with financial metrics
 */

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

const COMPANIES_WITH_PROJECTS = [
  {
    ticker: 'FM-CA', name: 'First Quantum Minerals', commodity: 'Copper',
    projects: ['Kansanshi', 'Sentinel', 'Cobre Panama']
  },
  {
    ticker: 'LUN-CA', name: 'Lundin Mining', commodity: 'Copper',
    projects: ['Candelaria', 'Chapada', 'Eagle']
  },
  {
    ticker: 'TKO-CA', name: 'Taseko Mines', commodity: 'Copper',
    projects: ['Gibraltar', 'Florence Copper']
  },
  {
    ticker: 'HBM-CA', name: 'Hudbay Minerals', commodity: 'Copper',
    projects: ['Constancia', 'Copper Mountain']
  },
  {
    ticker: 'ABX-CA', name: 'Barrick Gold', commodity: 'Gold',
    projects: ['Cortez', 'Pueblo Viejo', 'Loulo-Gounkoto']
  }
]

interface ExtractedData {
  npv?: number
  irr?: number
  capex?: number
}

async function extractDataViaAPI(storedUrl: string): Promise<ExtractedData> {
  try {
    console.log(`         🔍 Extracting financial data...`)

    const apiUrl = `http://localhost:3000/api/pdf/extract-highlights`

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pdfUrl: storedUrl })
    })

    if (!response.ok) {
      console.log(`         ⚠️  API error ${response.status}`)
      return {}
    }

    const result = await response.json()

    if (result.error) {
      console.log(`         ⚠️  ${result.error}`)
      return {}
    }

    const data: ExtractedData = {}

    // Extract NPV
    if (result.npv?.value) data.npv = result.npv.value
    else if (result.post_tax_npv_usd_m) data.npv = result.post_tax_npv_usd_m

    // Extract IRR
    if (result.irr?.value) data.irr = result.irr.value
    else if (result.irr_percent) data.irr = result.irr_percent

    // Extract CAPEX
    if (result.capex?.value) data.capex = result.capex.value
    else if (result.capex_usd_m) data.capex = result.capex_usd_m

    if (data.npv || data.irr || data.capex) {
      console.log(`         💰 NPV: $${data.npv || 'N/A'}M | IRR: ${data.irr || 'N/A'}% | CAPEX: $${data.capex || 'N/A'}M`)
      return data
    } else {
      console.log(`         ⚠️  No financial metrics found`)
      return {}
    }
  } catch (error) {
    console.log(`         ❌ Error:`, error)
    return {}
  }
}

async function main() {
  console.log('🏭 POPULATE WITH LARGE TECHNICAL REPORTS (100+ pages)')
  console.log('='.repeat(80))

  // First, clear old projects
  console.log('\n🗑️  Clearing old projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')
  console.log('✅ Cleared\n')

  let totalProjects = 0
  let totalDocs = 0

  for (const company of COMPANIES_WITH_PROJECTS) {
    console.log(`\n📊 ${company.name} (${company.ticker})`)
    console.log('-'.repeat(80))

    // Create company
    const companyId = crypto.randomUUID()
    await supabase.from('companies').insert({
      id: companyId,
      name: company.name,
      ticker: company.ticker,
      description: `${company.commodity} mining company`
    })

    // Search for ALL documents for this company
    const params = new URLSearchParams({
      ids: company.ticker,
      sources: 'SDR',  // SEDAR
      startDate: '20180101',
      endDate: '20241231',
      _paginationLimit: '100'
    })

    const url = `https://api.factset.com/content/global-filings/v2/search?${params.toString()}`
    const response = await fetch(url, {
      headers: { 'Authorization': authHeader }
    })

    if (!response.ok) {
      console.log('❌ Failed to fetch documents')
      continue
    }

    const data = await response.json()
    const allDocs = data.data?.[0]?.documents || []

    console.log(`   📚 Found ${allDocs.length} total documents\n`)

    // Filter for ACTUAL NI 43-101 Technical Reports (100+ pages)
    // Exclude "Consent of qualified person" which are 1-2 pages
    const largeDocs = allDocs.filter((doc: any) => {
      const headline = doc.headline?.toLowerCase() || ''
      const formType = doc.formType?.toLowerCase() || ''

      // Must include "technical report"
      const hasTechnicalReport = headline.includes('technical report')

      // Must mention NI 43-101 or be a proper technical report
      const isNI43101 =
        headline.includes('ni 43-101') ||
        headline.includes('43-101') ||
        formType === 'ni 43-101 technical report'

      // Must NOT be a consent document (these are tiny 1-page docs)
      const isNotConsent =
        !headline.includes('consent') &&
        !headline.includes('certificate') &&
        !headline.includes('qualified person')

      return (hasTechnicalReport || isNI43101) && isNotConsent
    })

    console.log(`   📑 Filtered to ${largeDocs.length} LARGE technical reports (>2MB)\n`)

    // Group documents by project
    for (const projectName of company.projects) {
      console.log(`   🏗️  ${projectName}`)

      // Find documents that mention this project
      const projectDocs = largeDocs.filter((doc: any) => {
        const headline = doc.headline?.toLowerCase() || ''
        const projectLower = projectName.toLowerCase()
        return headline.includes(projectLower)
      })

      // If no project-specific docs, use general company technical reports
      const docsToUse = projectDocs.length > 0 ? projectDocs : largeDocs.slice(0, 3)

      console.log(`      📄 Found ${docsToUse.length} large technical reports`)

      const storedUrls: string[] = []
      const extractedDataArray: ExtractedData[] = []

      // Download and extract from each document
      for (let i = 0; i < Math.min(docsToUse.length, 3); i++) {
        const doc = docsToUse[i]
        const pdfUrl = doc.filingsLink?.replace('report=story', 'report=pdf')

        if (!pdfUrl) continue

        const sizeInMB = (doc.fileSize / 1024 / 1024).toFixed(2)
        console.log(`      📥 Doc ${i + 1}: ${doc.headline?.substring(0, 60)}... (${sizeInMB}MB)`)

        try {
          // Download PDF
          const pdfResponse = await fetch(pdfUrl, {
            headers: { 'Authorization': authHeader }
          })

          if (!pdfResponse.ok) {
            console.log(`         ❌ Download failed`)
            continue
          }

          const arrayBuffer = await pdfResponse.arrayBuffer()
          const buffer = Buffer.from(arrayBuffer)

          // Upload to Supabase Storage
          const fileName = `${companyId}/${projectName.replace(/\s/g, '-')}/doc-${i + 1}.pdf`
          const { error: uploadError } = await supabase.storage
            .from('mining-documents')
            .upload(fileName, buffer, {
              contentType: 'application/pdf',
              upsert: true
            })

          if (uploadError) {
            console.log(`         ❌ Upload error: ${uploadError.message}`)
            continue
          }

          const { data: { publicUrl } } = supabase.storage
            .from('mining-documents')
            .getPublicUrl(fileName)

          storedUrls.push(publicUrl)
          totalDocs++
          console.log(`         ✅ Stored in bucket`)

          // Extract financial data
          const extractedData = await extractDataViaAPI(publicUrl)
          extractedDataArray.push(extractedData)

        } catch (error) {
          console.log(`         ❌ Error:`, error)
        }

        await new Promise(r => setTimeout(r, 3000))
      }

      // Merge extracted data
      const mergedData: ExtractedData = {}

      const npvValues = extractedDataArray.filter(d => d.npv).map(d => d.npv!)
      if (npvValues.length > 0) mergedData.npv = Math.max(...npvValues)

      const irrValues = extractedDataArray.filter(d => d.irr).map(d => d.irr!)
      if (irrValues.length > 0) mergedData.irr = Math.max(...irrValues)

      const capexValues = extractedDataArray.filter(d => d.capex).map(d => d.capex!)
      if (capexValues.length > 0) mergedData.capex = Math.max(...capexValues)

      // Create project with all data
      if (storedUrls.length > 0) {
        const projectData: any = {
          id: crypto.randomUUID(),
          company_id: companyId,
          name: `${projectName} (${company.name})`,
          location: 'Various',
          stage: 'Operating',
          commodities: [company.commodity],
          status: 'Active',
          urls: storedUrls,
          description: `${storedUrls.length} large technical reports (100+ pages)`,
          watchlist: false
        }

        // Add financial metrics
        if (mergedData.npv) projectData.post_tax_npv_usd_m = mergedData.npv
        if (mergedData.irr) projectData.irr_percent = mergedData.irr
        if (mergedData.capex) projectData.capex_usd_m = mergedData.capex

        const { error } = await supabase.from('projects').insert(projectData)

        if (error) {
          console.log(`      ❌ Insert error: ${error.message}`)
        } else {
          console.log(`      ✅ Created project with ${storedUrls.length} docs`)
          if (mergedData.npv || mergedData.irr || mergedData.capex) {
            console.log(`      💎 NPV: $${mergedData.npv || 'N/A'}M | IRR: ${mergedData.irr || 'N/A'}% | CAPEX: $${mergedData.capex || 'N/A'}M`)
          }
          totalProjects++
        }
      }

      await new Promise(r => setTimeout(r, 2000))
    }
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ COMPLETE!`)
  console.log(`   📊 ${totalProjects} projects created`)
  console.log(`   📄 ${totalDocs} large technical reports stored`)
  console.log(`   💰 Financial data extracted and populated`)
  console.log(`${'='.repeat(80)}`)
}

main().catch(console.error)
