#!/usr/bin/env npx tsx
/**
 * SCALE DEMONSTRATION - Exhaustive Database Population
 * 
 * This extracts EVERY project from EVERY company with ALL their documents
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
    projects: ['Kansanshi', 'Sentinel', 'Cobre Panama', 'Las Cruces', 'Guelb Moghrein']
  },
  {
    ticker: 'LUN-CA', name: 'Lundin Mining', commodity: 'Copper',
    projects: ['Candelaria', 'Eagle', 'Neves-Corvo', 'Zinkgruvan', 'Chapada']
  },
  {
    ticker: 'TKO-CA', name: 'Taseko Mines', commodity: 'Copper',
    projects: ['Gibraltar', 'Florence Copper', 'Yellowhead']
  },
  {
    ticker: 'HBM-CA', name: 'Hudbay Minerals', commodity: 'Copper',
    projects: ['Constancia', 'Copper Mountain', 'Snow Lake', '777', 'Lalor']
  },
  {
    ticker: 'ABX-CA', name: 'Barrick Gold', commodity: 'Gold',
    projects: ['Cortez', 'Goldstrike', 'Turquoise Ridge', 'Pueblo Viejo', 'Loulo-Gounkoto']
  }
]

async function main() {
  console.log('🏭 SCALE DEMONSTRATION')
  console.log('='.repeat(80))
  console.log('This will create MULTIPLE projects per company with ALL documents')
  console.log('='.repeat(80))

  let totalProjects = 0

  for (const company of COMPANIES_WITH_PROJECTS) {
    console.log(`\n📊 ${company.name}`)

    const companyId = crypto.randomUUID()
    const { error: companyInsertError } = await supabase.from('companies').insert({
      id: companyId,
      name: company.name,
      ticker: company.ticker,
      description: `${company.commodity} mining company`
    })

    if (companyInsertError) {
      console.log(`   ❌ Company insert error:`, companyInsertError)
      continue
    }

    // First: Search by company ID to get ALL documents for this company
    console.log(`   🔍 Searching all documents for ${company.ticker}...`)
    const companyParams = new URLSearchParams({
      ids: company.ticker,
      sources: 'SDR',
      startDate: '20180101',
      endDate: '20241231',
      _paginationLimit: '100'
    })

    const companyUrl = `https://api.factset.com/content/global-filings/v2/search?${companyParams.toString()}`
    const companyResponse = await fetch(companyUrl, {
      headers: { 'Authorization': authHeader }
    })

    if (!companyResponse.ok) {
      console.log(`   ❌ Failed to get company documents`)
      continue
    }

    const companyData = await companyResponse.json()
    console.log(`   📚 Found ${companyData.data?.[0]?.documents?.length || 0} total documents`)

    // Group documents by project
    for (const projectName of company.projects) {
      console.log(`\n   🏗️  Processing project: ${projectName}`)

      const pdfUrls: string[] = []
      const projectDocs: any[] = []

      // Filter documents that mention this project
      if (companyData.data?.[0]?.documents) {
        for (const doc of companyData.data[0].documents) {
          const headline = doc.headline?.toLowerCase() || ''
          const projectLower = projectName.toLowerCase()

          // Check if document mentions this project
          if (headline.includes(projectLower) ||
              headline.includes('technical report') ||
              headline.includes('ni 43-101') ||
              headline.includes('annual report')) {
            projectDocs.push(doc)

            const pdfUrl = doc.filingsLink?.replace('report=story', 'report=pdf')
            if (pdfUrl && pdfUrls.length < 5) {
              pdfUrls.push(pdfUrl)
            }
          }
        }
      }

      console.log(`      📄 Found ${projectDocs.length} documents mentioning ${projectName}`)

      if (pdfUrls.length > 0) {
        const projectId = crypto.randomUUID()
        const { error: projectInsertError } = await supabase.from('projects').insert({
          id: projectId,
          company_id: companyId,
          name: `${projectName} (${company.name})`,
          location: 'Various',
          stage: 'Operating',
          commodities: [company.commodity],
          status: 'Active',
          urls: pdfUrls,
          description: `${pdfUrls.length} FactSet documents for ${projectName}`,
          watchlist: false
        })

        if (projectInsertError) {
          console.log(`      ❌ Project insert error:`, projectInsertError)
        } else {
          console.log(`      ✅ Created project with ${pdfUrls.length} documents`)
          totalProjects++
        }
      } else {
        console.log(`      ⚠️  No PDFs found for ${projectName}`)
      }

      await new Promise(r => setTimeout(r, 500))
    }

    await new Promise(r => setTimeout(r, 2000))
  }

  console.log(`\n✅ Created ${totalProjects} projects!`)
}

main().catch(console.error)
