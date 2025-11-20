#!/usr/bin/env npx tsx
/**
 * COMPREHENSIVE MINING DATABASE POPULATION
 *
 * 1. Exhaustive list of major mining companies
 * 2. Search for LARGE NI 43-101 technical reports (100+ pages)
 * 3. Download and store in Supabase
 * 4. Extract NPV, IRR, CAPEX using OpenAI
 * 5. Populate database with financial metrics
 */

import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import pdf from 'pdf-parse'

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

// EXHAUSTIVE list of major mining companies with their key projects
const MINING_COMPANIES = [
  // Copper
  { ticker: 'FM-CA', name: 'First Quantum Minerals', commodity: 'Copper', projects: ['Kansanshi', 'Sentinel', 'Cobre Panama', 'Las Cruces'] },
  { ticker: 'TECK-CA', name: 'Teck Resources', commodity: 'Copper', projects: ['Highland Valley', 'Antamina', 'Quebrada Blanca', 'Carmen de Andacollo'] },
  { ticker: 'LUN-CA', name: 'Lundin Mining', commodity: 'Copper', projects: ['Candelaria', 'Ojos del Salado', 'Eagle', 'Neves-Corvo', 'Zinkgruvan', 'Chapada'] },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', commodity: 'Copper', projects: ['Constancia', 'Copper Mountain', 'Snow Lake', '777', 'Lalor'] },
  { ticker: 'CMMC-CA', name: 'Copper Mountain Mining', commodity: 'Copper', projects: ['Copper Mountain', 'Eva Copper'] },
  { ticker: 'CS-CA', name: 'Capstone Copper', commodity: 'Copper', projects: ['Pinto Valley', 'Cozamin', 'Mantos Blancos', 'Santo Domingo'] },
  { ticker: 'TKO-CA', name: 'Taseko Mines', commodity: 'Copper', projects: ['Gibraltar', 'Florence Copper', 'Yellowhead', 'Aley'] },
  { ticker: 'SCCO-US', name: 'Southern Copper', commodity: 'Copper', projects: ['Buenavista', 'La Caridad', 'Cuajone', 'Toquepala'] },

  // Gold
  { ticker: 'ABX-CA', name: 'Barrick Gold', commodity: 'Gold', projects: ['Cortez', 'Goldstrike', 'Turquoise Ridge', 'Pueblo Viejo', 'Loulo-Gounkoto', 'Kibali'] },
  { ticker: 'NEM-US', name: 'Newmont', commodity: 'Gold', projects: ['Carlin', 'Boddington', 'Tanami', 'Ahafo', 'Yanacocha'] },
  { ticker: 'AEM-CA', name: 'Agnico Eagle', commodity: 'Gold', projects: ['LaRonde', 'Canadian Malartic', 'Meadowbank', 'Meliadine', 'Detour Lake'] },
  { ticker: 'KL-CA', name: 'Kirkland Lake Gold', commodity: 'Gold', projects: ['Macassa', 'Detour Lake', 'Fosterville'] },
  { ticker: 'EVN-AU', name: 'Evolution Mining', commodity: 'Gold', projects: ['Cowal', 'Mt Rawdon', 'Mungari', 'Red Lake'] },
  { ticker: 'B2G-CA', name: 'B2Gold', commodity: 'Gold', projects: ['Fekola', 'Masbate', 'Otjikoto'] },
  { ticker: 'ELD-CA', name: 'Eldorado Gold', commodity: 'Gold', projects: ['Kisladag', 'Efemcukuru', 'Olympias', 'Lamaque'] },
  { ticker: 'YRI-CA', name: 'Yamana Gold', commodity: 'Gold', projects: ['Jacobina', 'Canadian Malartic', 'Cerro Moro'] },
  { ticker: 'KGC-CA', name: 'Kinross Gold', commodity: 'Gold', projects: ['Tasiast', 'Paracatu', 'Fort Knox', 'Kupol'] },

  // Nickel
  { ticker: 'VALE-US', name: 'Vale', commodity: 'Nickel', projects: ['Sudbury', 'Voiseys Bay', 'Thompson', 'Onca Puma'] },
  { ticker: 'SBSW-US', name: 'Sibanye Stillwater', commodity: 'Nickel', projects: ['Stillwater', 'East Boulder', 'Marathon'] },
  { ticker: 'IGO-AU', name: 'IGO Limited', commodity: 'Nickel', projects: ['Nova', 'Forrestania', 'Cosmos'] },

  // Lithium
  { ticker: 'ALB-US', name: 'Albemarle', commodity: 'Lithium', projects: ['Atacama', 'Greenbushes', 'Wodgina'] },
  { ticker: 'SQM-US', name: 'SQM', commodity: 'Lithium', projects: ['Atacama', 'Mt Holland'] },
  { ticker: 'LAC-CA', name: 'Lithium Americas', commodity: 'Lithium', projects: ['Thacker Pass', 'Caucharí-Olaroz'] },
  { ticker: 'PLS-AU', name: 'Pilbara Minerals', commodity: 'Lithium', projects: ['Pilgangoora'] },
  { ticker: 'MIN-AU', name: 'Mineral Resources', commodity: 'Lithium', projects: ['Mt Marion', 'Wodgina'] },

  // Zinc/Lead
  { ticker: 'TCK-US', name: 'Teck Resources', commodity: 'Zinc', projects: ['Red Dog', 'Trail'] },
  { ticker: 'BHP-AU', name: 'BHP', commodity: 'Zinc', projects: ['Antamina', 'Olympic Dam', 'Cannington'] },

  // Iron Ore
  { ticker: 'RIO-AU', name: 'Rio Tinto', commodity: 'Iron Ore', projects: ['Pilbara', 'IOC', 'Oyu Tolgoi'] },
  { ticker: 'FMG-AU', name: 'Fortescue', commodity: 'Iron Ore', projects: ['Chichester', 'Solomon', 'Eliwana'] }
]

interface ExtractedData {
  npv?: number
  irr?: number
  capex?: number
}

async function extractTextFromPDF(buffer: Buffer): Promise<string> {
  try {
    const data = await pdf(buffer)
    return data.text
  } catch (error) {
    console.log(`         ⚠️  PDF parse error`)
    return ''
  }
}

async function extractFinancialDataWithAI(text: string, projectName: string): Promise<ExtractedData> {
  try {
    // Focus on first 15000 chars (executive summary + economics section)
    const relevantText = text.substring(0, 15000)

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `You are a mining financial analyst. Extract NPV (Net Present Value in USD millions), IRR (Internal Rate of Return as percentage), and CAPEX (Capital Expenditure in USD millions) from this NI 43-101 technical report for the ${projectName} project. Return ONLY valid JSON with numeric values. Look for terms like "Post-Tax NPV", "After-Tax NPV", "Internal Rate of Return", "IRR", "Initial CAPEX", "Capital Cost". Return empty object if not found.`
        },
        {
          role: 'user',
          content: `Extract financial metrics from this mining technical report:\n\n${relevantText}`
        }
      ],
      response_format: { type: 'json_object' },
      temperature: 0.1
    })

    const result = JSON.parse(response.choices[0].message.content || '{}')

    const data: ExtractedData = {}
    if (result.npv || result.NPV) data.npv = parseFloat(result.npv || result.NPV)
    if (result.irr || result.IRR) data.irr = parseFloat(result.irr || result.IRR)
    if (result.capex || result.CAPEX) data.capex = parseFloat(result.capex || result.CAPEX)

    return data
  } catch (error) {
    console.log(`         ⚠️  AI extraction error`)
    return {}
  }
}

async function searchForNI43101Reports(companyTicker: string, projectName: string): Promise<any[]> {
  try {
    const params = new URLSearchParams({
      ids: companyTicker,
      sources: 'SDR',
      searchText: `${projectName} NI 43-101`,
      startDate: '20180101',
      endDate: '20241231',
      _paginationLimit: '30'
    })

    const url = `https://api.factset.com/content/global-filings/v2/search?${params.toString()}`
    const response = await fetch(url, {
      headers: { 'Authorization': authHeader }
    })

    if (!response.ok) return []

    const data = await response.json()
    const docs = data.data?.[0]?.documents || []

    // Filter for ACTUAL technical reports
    return docs.filter((doc: any) => {
      const headline = doc.headline?.toLowerCase() || ''
      return (
        headline.includes('technical report') &&
        (headline.includes('ni 43-101') || headline.includes('43-101')) &&
        !headline.includes('consent') &&
        !headline.includes('certificate')
      )
    })
  } catch (error) {
    return []
  }
}

async function main() {
  console.log('🌍 COMPREHENSIVE MINING DATABASE POPULATION')
  console.log('='.repeat(80))
  console.log(`Processing ${MINING_COMPANIES.length} mining companies`)
  console.log('Extracting NPV, IRR, CAPEX from NI 43-101 technical reports')
  console.log('='.repeat(80))

  // Clear old data
  console.log('\n🗑️  Clearing old projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')
  console.log('✅ Cleared\n')

  let totalProjects = 0
  let totalDocs = 0
  let totalExtracted = 0

  for (const company of MINING_COMPANIES) {
    console.log(`\n📊 ${company.name} (${company.ticker})`)
    console.log('-'.repeat(80))

    const companyId = crypto.randomUUID()
    await supabase.from('companies').insert({
      id: companyId,
      name: company.name,
      ticker: company.ticker,
      description: `${company.commodity} mining company`
    })

    for (const projectName of company.projects) {
      console.log(`\n   🏗️  ${projectName}`)

      const technicalReports = await searchForNI43101Reports(company.ticker, projectName)
      console.log(`      📚 Found ${technicalReports.length} NI 43-101 reports`)

      if (technicalReports.length === 0) {
        console.log(`      ⚠️  Skipping - no technical reports`)
        continue
      }

      const storedUrls: string[] = []
      const extractedDataArray: ExtractedData[] = []

      // Process up to 2 technical reports per project
      for (let i = 0; i < Math.min(technicalReports.length, 2); i++) {
        const doc = technicalReports[i]
        const pdfUrl = doc.filingsLink?.replace('report=story', 'report=pdf')
        if (!pdfUrl) continue

        console.log(`      📥 Doc ${i + 1}: ${doc.headline?.substring(0, 60)}...`)

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
          const sizeInMB = (buffer.length / 1024 / 1024).toFixed(2)
          console.log(`         📦 ${sizeInMB}MB`)

          // Only process large documents (>2MB = 100+ pages)
          if (buffer.length < 2000000) {
            console.log(`         ⚠️  Too small, skipping`)
            continue
          }

          // Upload to Supabase
          const fileName = `${companyId}/${projectName.replace(/\s/g, '-')}/ni43101-${i + 1}.pdf`
          const { error: uploadError } = await supabase.storage
            .from('mining-documents')
            .upload(fileName, buffer, {
              contentType: 'application/pdf',
              upsert: true
            })

          if (uploadError) {
            console.log(`         ❌ Upload failed: ${uploadError.message}`)
            continue
          }

          const { data: { publicUrl } } = supabase.storage
            .from('mining-documents')
            .getPublicUrl(fileName)

          storedUrls.push(publicUrl)
          totalDocs++
          console.log(`         ✅ Stored in bucket`)

          // Extract text from PDF
          console.log(`         📖 Extracting text...`)
          const text = await extractTextFromPDF(buffer)

          if (text.length > 1000) {
            console.log(`         🔍 Analyzing with AI...`)
            const extractedData = await extractFinancialDataWithAI(text, projectName)
            extractedDataArray.push(extractedData)

            if (extractedData.npv || extractedData.irr || extractedData.capex) {
              console.log(`         💰 NPV: $${extractedData.npv || 'N/A'}M | IRR: ${extractedData.irr || 'N/A'}% | CAPEX: $${extractedData.capex || 'N/A'}M`)
            }
          }

        } catch (error) {
          console.log(`         ❌ Error: ${error}`)
        }

        await new Promise(r => setTimeout(r, 2000))
      }

      // Merge extracted data
      const mergedData: ExtractedData = {}
      const npvValues = extractedDataArray.filter(d => d.npv && !isNaN(d.npv)).map(d => d.npv!)
      const irrValues = extractedDataArray.filter(d => d.irr && !isNaN(d.irr)).map(d => d.irr!)
      const capexValues = extractedDataArray.filter(d => d.capex && !isNaN(d.capex)).map(d => d.capex!)

      if (npvValues.length > 0) mergedData.npv = Math.max(...npvValues)
      if (irrValues.length > 0) mergedData.irr = Math.max(...irrValues)
      if (capexValues.length > 0) mergedData.capex = Math.max(...capexValues)

      // Create project
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
          description: `${storedUrls.length} NI 43-101 technical reports`,
          watchlist: false
        }

        if (mergedData.npv) projectData.post_tax_npv_usd_m = mergedData.npv
        if (mergedData.irr) projectData.irr_percent = mergedData.irr
        if (mergedData.capex) projectData.capex_usd_m = mergedData.capex

        const { error } = await supabase.from('projects').insert(projectData)

        if (!error) {
          console.log(`      ✅ Created project with ${storedUrls.length} docs`)
          if (mergedData.npv || mergedData.irr || mergedData.capex) {
            console.log(`      💎 NPV: $${mergedData.npv || 'N/A'}M | IRR: ${mergedData.irr || 'N/A'}% | CAPEX: $${mergedData.capex || 'N/A'}M`)
            totalExtracted++
          }
          totalProjects++
        }
      }

      await new Promise(r => setTimeout(r, 1000))
    }

    await new Promise(r => setTimeout(r, 2000))
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ COMPLETE!`)
  console.log(`   📊 ${totalProjects} projects created`)
  console.log(`   📄 ${totalDocs} NI 43-101 reports stored`)
  console.log(`   💰 ${totalExtracted} projects with extracted financial data`)
  console.log(`${'='.repeat(80)}`)
}

main().catch(console.error)
