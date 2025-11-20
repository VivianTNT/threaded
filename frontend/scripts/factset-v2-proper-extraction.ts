#!/usr/bin/env npx tsx

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import axios from 'axios'
import * as fs from 'fs/promises'
import * as path from 'path'
import pdf from 'pdf-parse'
import { v4 as uuidv4 } from 'uuid'

config({ path: path.join(__dirname, '..', '.env.local') })

// Updated FactSet credentials from user
const FACTSET_CONFIG = {
  username: 'LITHOS-2220379',
  apiKey: '2BP7hqJOtH73DHvao1RJCQW0m1c4tVyOF81QVHxu',
  baseUrl: 'https://api.factset.com/content/global-filings/v2'
}

// Supabase clients
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Mining-specific search terms for NI 43-101 and technical reports
const SEARCH_QUERIES = [
  'NI 43-101 technical report',
  'feasibility study mining',
  'preliminary economic assessment',
  'mineral resource estimate',
  'proven and probable reserves'
]

// Known mining companies with active projects
const TARGET_COMPANIES = [
  { ticker: 'NEM', name: 'Newmont Corporation' },
  { ticker: 'ABX.TO', name: 'Barrick Gold' },
  { ticker: 'AEM', name: 'Agnico Eagle' },
  { ticker: 'KGC', name: 'Kinross Gold' },
  { ticker: 'FNV', name: 'Franco-Nevada' },
  { ticker: 'WPM', name: 'Wheaton Precious Metals' },
  { ticker: 'PAAS', name: 'Pan American Silver' },
  { ticker: 'AG', name: 'First Majestic Silver' },
  { ticker: 'CDE', name: 'Coeur Mining' },
  { ticker: 'HL', name: 'Hecla Mining' }
]

async function searchForFilings(ticker: string, companyName: string) {
  console.log(`🔍 Searching for ${companyName} (${ticker}) filings...`)

  const allFilings: any[] = []

  for (const query of SEARCH_QUERIES) {
    try {
      const searchPayload = {
        sources: ['SDR', 'SDRP', 'EDG'], // SEDAR, SEDAR+, EDGAR
        search: `${ticker} ${query}`,
        limit: 5,
        sort: ['-filingDate'],
        formTypes: ['SEDAR:NI43-101', 'EDGAR:8-K', 'EDGAR:6-K', 'SEDAR:TECHNICAL']
      }

      console.log(`   → Searching: "${query}"`)

      const response = await axios.post(
        `${FACTSET_CONFIG.baseUrl}/search`,
        searchPayload,
        {
          auth: {
            username: FACTSET_CONFIG.username,
            password: FACTSET_CONFIG.apiKey
          },
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          timeout: 30000
        }
      )

      if (response.data?.data && Array.isArray(response.data.data)) {
        const filings = response.data.data.filter((f: any) => {
          const title = (f.title || '').toLowerCase()
          return title.includes('technical') ||
                 title.includes('43-101') ||
                 title.includes('feasibility') ||
                 title.includes('resource') ||
                 title.includes('economic assessment')
        })

        if (filings.length > 0) {
          console.log(`   ✅ Found ${filings.length} relevant filings`)
          allFilings.push(...filings)
        }
      }

      // Rate limiting
      await new Promise(r => setTimeout(r, 1000))

    } catch (error: any) {
      if (error.response?.status === 404) {
        console.log(`   ⚠️ No filings found for this query`)
      } else if (error.response?.status === 429) {
        console.log(`   ⚠️ Rate limited, waiting...`)
        await new Promise(r => setTimeout(r, 5000))
      } else {
        console.log(`   ❌ Search error: ${error.response?.data?.error?.message || error.message}`)
      }
    }
  }

  // Deduplicate by document URL
  const uniqueFilings = Array.from(
    new Map(allFilings.map(f => [f.documentUrl || f.accessionNo, f])).values()
  )

  return uniqueFilings.slice(0, 3) // Return top 3 most relevant
}

async function downloadDocument(url: string, fileName: string): Promise<Buffer | null> {
  try {
    console.log(`   📥 Downloading: ${fileName}`)

    // CRITICAL: Must authenticate when downloading as per email
    const response = await axios.get(url, {
      auth: {
        username: FACTSET_CONFIG.username,
        password: FACTSET_CONFIG.apiKey
      },
      responseType: 'arraybuffer',
      timeout: 120000, // 2 minutes for large PDFs
      maxContentLength: 200 * 1024 * 1024, // 200MB max
      headers: {
        'Accept': 'application/pdf,application/octet-stream,*/*'
      }
    })

    const buffer = Buffer.from(response.data)
    console.log(`   ✅ Downloaded ${(buffer.length / 1024 / 1024).toFixed(2)}MB`)
    return buffer

  } catch (error: any) {
    console.log(`   ❌ Download failed: ${error.response?.status || error.message}`)
    return null
  }
}

async function extractFinancialMetrics(buffer: Buffer): Promise<any> {
  try {
    const data = await pdf(buffer)
    const text = data.text
    const pageCount = data.numpages

    console.log(`   📄 Parsing ${pageCount} pages...`)

    const metrics: any = {
      npv: null,
      irr: null,
      capex: null,
      resource: null,
      reserve: null,
      mineLife: null,
      ownership: null
    }

    // NPV patterns
    const npvPatterns = [
      /NPV[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i,
      /Net Present Value[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i,
      /After-tax NPV[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i
    ]

    for (const pattern of npvPatterns) {
      const match = text.match(pattern)
      if (match) {
        metrics.npv = parseFloat(match[1].replace(/,/g, ''))
        break
      }
    }

    // IRR patterns
    const irrPatterns = [
      /IRR[^\d]*([\d]+(?:\.\d+)?)\s*%/i,
      /Internal Rate of Return[^\d]*([\d]+(?:\.\d+)?)\s*%/i,
      /After-tax IRR[^\d]*([\d]+(?:\.\d+)?)\s*%/i
    ]

    for (const pattern of irrPatterns) {
      const match = text.match(pattern)
      if (match) {
        metrics.irr = parseFloat(match[1])
        break
      }
    }

    // CAPEX patterns
    const capexPatterns = [
      /(?:Initial )?Capital Cost[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i,
      /CAPEX[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i,
      /Pre-production Capital[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i
    ]

    for (const pattern of capexPatterns) {
      const match = text.match(pattern)
      if (match) {
        metrics.capex = parseFloat(match[1].replace(/,/g, ''))
        break
      }
    }

    // Resource estimation
    const resourceMatch = text.match(
      /(?:Total |Measured \+ Indicated |M\+I )?Resource[^\d]*([\d,]+(?:\.\d+)?)\s*(?:Mt|million tonnes|Moz|million ounces)/i
    )
    if (resourceMatch) {
      metrics.resource = resourceMatch[0].substring(0, 100)
    }

    // Reserve estimation
    const reserveMatch = text.match(
      /(?:Total |Proven \+ Probable |P\+P )?Reserve[^\d]*([\d,]+(?:\.\d+)?)\s*(?:Mt|million tonnes|Moz|million ounces)/i
    )
    if (reserveMatch) {
      metrics.reserve = reserveMatch[0].substring(0, 100)
    }

    // Mine life
    const mineLifeMatch = text.match(/Mine Life[^\d]*([\d]+(?:\.\d+)?)\s*years/i)
    if (mineLifeMatch) {
      metrics.mineLife = parseFloat(mineLifeMatch[1])
    }

    // Ownership
    const ownershipMatch = text.match(/(?:owns?|ownership|interest)[^\d]*([\d]+(?:\.\d+)?)\s*%/i)
    if (ownershipMatch) {
      metrics.ownership = parseFloat(ownershipMatch[1])
    }

    // Count valid metrics
    const validMetrics = Object.values(metrics).filter(v => v !== null).length
    const isValid = validMetrics >= 3 // At least 3 financial metrics

    return { metrics, isValid, pageCount, validMetrics }

  } catch (error: any) {
    console.log(`   ❌ PDF parsing error: ${error.message}`)
    return { metrics: {}, isValid: false, pageCount: 0, validMetrics: 0 }
  }
}

async function processCompany(company: any) {
  console.log(`\n${'='.repeat(60)}`)
  console.log(`🏢 Processing ${company.name} (${company.ticker})`)
  console.log('='.repeat(60))

  try {
    // Search for filings
    const filings = await searchForFilings(company.ticker, company.name)

    if (filings.length === 0) {
      console.log('   ⚠️ No technical reports found')
      return { processed: 0, extracted: 0 }
    }

    console.log(`   📚 Found ${filings.length} technical documents to process`)

    let processedCount = 0
    let extractedCount = 0

    for (const filing of filings) {
      if (!filing.documentUrl) {
        console.log(`   ⚠️ No document URL for ${filing.title}`)
        continue
      }

      const fileName = `${company.ticker}_${filing.filingDate}_${filing.accessionNo || uuidv4()}.pdf`

      // Download the document
      const pdfBuffer = await downloadDocument(filing.documentUrl, fileName)
      if (!pdfBuffer) continue

      processedCount++

      // Extract financial metrics
      const { metrics, isValid, pageCount, validMetrics } = await extractFinancialMetrics(pdfBuffer)

      console.log(`   📊 Extracted ${validMetrics} metrics from ${pageCount} pages`)

      if (!isValid) {
        console.log(`   ⚠️ Insufficient financial metrics (need at least 3)`)
        continue
      }

      extractedCount++

      // Upload to Supabase storage
      const storagePath = `factset-documents/${fileName}`
      const { error: uploadError } = await supabase.storage
        .from('factset-documents')
        .upload(storagePath, pdfBuffer, {
          contentType: 'application/pdf',
          upsert: true
        })

      if (uploadError) {
        console.log(`   ❌ Storage upload failed: ${uploadError.message}`)
        continue
      }

      console.log(`   ☁️ Uploaded to storage: ${storagePath}`)

      // Create project record
      const projectName = filing.title
        ?.replace(/NI 43-101|Technical Report|Feasibility Study/gi, '')
        ?.trim() || `${company.name} Project`

      const projectId = uuidv4()

      const projectData = {
        id: projectId,
        name: projectName,
        company_id: company.id,
        location: filing.description?.match(/(?:located in|property in)\s+([^,\.]+)/i)?.[1] || null,
        stage: metrics.capex ? 'Development' : 'Feasibility',
        commodities: ['Gold'], // Default, could be extracted
        status: 'Active',
        npv: metrics.npv,
        irr: metrics.irr,
        capex: metrics.capex,
        resource: metrics.resource,
        reserve: metrics.reserve,
        document_storage_path: storagePath,
        urls: [filing.documentUrl],
        description: `Technical report from ${filing.source} dated ${filing.filingDate}. ${pageCount} pages.`,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }

      const { error: projectError } = await supabase
        .from('projects')
        .upsert(projectData)

      if (projectError) {
        console.log(`   ❌ Failed to save project: ${projectError.message}`)
      } else {
        console.log(`   ✅ Created project: ${projectName}`)
        console.log(`      NPV: $${metrics.npv}M | IRR: ${metrics.irr}%`)
        console.log(`      CAPEX: $${metrics.capex}M`)

        // Store highlights for each metric
        const highlights = []
        if (metrics.npv) {
          highlights.push({
            id: uuidv4(),
            project_id: projectId,
            data_type: 'npv',
            value: metrics.npv.toString(),
            quote: `NPV: $${metrics.npv} million`,
            page: 1
          })
        }

        if (metrics.irr) {
          highlights.push({
            id: uuidv4(),
            project_id: projectId,
            data_type: 'irr',
            value: metrics.irr.toString(),
            quote: `IRR: ${metrics.irr}%`,
            page: 1
          })
        }

        if (metrics.capex) {
          highlights.push({
            id: uuidv4(),
            project_id: projectId,
            data_type: 'capex',
            value: metrics.capex.toString(),
            quote: `CAPEX: $${metrics.capex} million`,
            page: 1
          })
        }

        if (highlights.length > 0) {
          await supabase.from('pdf_highlights').insert(highlights)
          console.log(`      📌 Saved ${highlights.length} highlights`)
        }
      }

      // Rate limiting between documents
      await new Promise(r => setTimeout(r, 2000))
    }

    return { processed: processedCount, extracted: extractedCount }

  } catch (error: any) {
    console.log(`   ❌ Company processing error: ${error.message}`)
    return { processed: 0, extracted: 0 }
  }
}

async function main() {
  console.log('🚀 FactSet Global Filings V2 - PDF Extraction Pipeline')
  console.log('📋 Using proper dual authentication pattern')
  console.log('🔐 Credentials: ' + FACTSET_CONFIG.username)
  console.log('')

  // Ensure storage bucket exists
  const { data: buckets } = await supabase.storage.listBuckets()
  if (!buckets?.find(b => b.name === 'factset-documents')) {
    const { error } = await supabase.storage.createBucket('factset-documents', {
      public: false,
      fileSizeLimit: 200 * 1024 * 1024 // 200MB
    })
    if (!error) {
      console.log('✅ Created factset-documents storage bucket\n')
    }
  }

  // Get companies from database
  const { data: dbCompanies, error: companiesError } = await supabase
    .from('companies')
    .select('*')
    .in('ticker', TARGET_COMPANIES.map(c => c.ticker))

  let companies = dbCompanies || []

  // If companies don't exist in DB, add them
  if (companies.length === 0) {
    console.log('📝 Adding target companies to database...')

    for (const company of TARGET_COMPANIES) {
      const { data } = await supabase
        .from('companies')
        .upsert({
          id: uuidv4(),
          name: company.name,
          ticker: company.ticker,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        .select()
        .single()

      if (data) {
        companies.push(data)
      }
    }
    console.log(`✅ Added ${companies.length} companies\n`)
  }

  console.log(`📊 Processing ${companies.length} mining companies...`)
  console.log('')

  let totalProcessed = 0
  let totalExtracted = 0

  for (const company of companies) {
    const { processed, extracted } = await processCompany(company)
    totalProcessed += processed
    totalExtracted += extracted

    // Rate limiting between companies
    await new Promise(r => setTimeout(r, 3000))
  }

  // Final summary
  console.log('\n' + '='.repeat(60))
  console.log('📊 EXTRACTION COMPLETE')
  console.log('='.repeat(60))
  console.log(`✅ Documents processed: ${totalProcessed}`)
  console.log(`✅ Projects extracted: ${totalExtracted}`)

  // Show final counts
  const { count: projectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .not('document_storage_path', 'is', null)

  const { count: highlightCount } = await supabase
    .from('pdf_highlights')
    .select('*', { count: 'exact', head: true })

  console.log(`📁 Total projects with documents: ${projectCount || 0}`)
  console.log(`📌 Total highlights stored: ${highlightCount || 0}`)
  console.log('='.repeat(60))
}

main().catch(console.error)