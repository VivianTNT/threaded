#!/usr/bin/env npx tsx

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import axios from 'axios'
import * as path from 'path'
import { v4 as uuidv4 } from 'uuid'

config({ path: path.join(__dirname, '..', '.env.local') })

// FactSet credentials - updated from user
const FACTSET_AUTH = {
  username: 'LITHOS-2220379',
  password: '2BP7hqJOtH73DHvao1RJCQW0m1c4tVyOF81QVHxu'
}

// Supabase
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Based on the notebook - using GET request for search, not POST
async function searchFilings(ticker: string) {
  try {
    console.log(`   🔍 Searching for ${ticker} filings...`)

    // Build search query based on notebook example
    const params = new URLSearchParams({
      sources: 'SDR,SDRP,EDG', // SEDAR, SEDAR+, EDGAR
      search: `${ticker} AND (technical report OR feasibility OR 43-101)`,
      limit: '10',
      sort: '-filingDate'
    })

    const url = `https://api.factset.com/content/global-filings/v2/search?${params}`

    const response = await axios.get(url, {
      auth: {
        username: FACTSET_AUTH.username,
        password: FACTSET_AUTH.password
      },
      headers: {
        'Accept': 'application/json'
      }
    })

    if (response.data?.data) {
      const technicalReports = response.data.data.filter((doc: any) => {
        const title = (doc.title || '').toLowerCase()
        return title.includes('technical') ||
               title.includes('43-101') ||
               title.includes('feasibility')
      })

      console.log(`   ✅ Found ${technicalReports.length} technical reports`)
      return technicalReports
    }

    return []
  } catch (error: any) {
    if (error.response?.status === 401) {
      console.log(`   ❌ Authentication failed - check credentials`)
    } else if (error.response?.status === 400) {
      console.log(`   ❌ Bad request - ${error.response?.data?.error?.message || 'check parameters'}`)
    } else {
      console.log(`   ❌ Error: ${error.response?.status} - ${error.message}`)
    }
    return []
  }
}

// Download document with auth in request
async function downloadPDF(url: string): Promise<Buffer | null> {
  try {
    console.log(`   📥 Downloading PDF...`)

    const response = await axios.get(url, {
      auth: {
        username: FACTSET_AUTH.username,
        password: FACTSET_AUTH.password
      },
      responseType: 'arraybuffer',
      timeout: 60000,
      maxContentLength: 100 * 1024 * 1024
    })

    const buffer = Buffer.from(response.data)
    console.log(`   ✅ Downloaded ${(buffer.length / 1024 / 1024).toFixed(2)}MB`)
    return buffer

  } catch (error: any) {
    console.log(`   ❌ Download failed: ${error.response?.status || error.message}`)
    return null
  }
}

// Simple extraction without pdf-parse (since we may have HTML content)
function extractMetricsFromContent(content: string): any {
  const metrics: any = {
    npv: null,
    irr: null,
    capex: null,
    resource: null,
    reserve: null
  }

  // NPV
  const npvMatch = content.match(/NPV[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i)
  if (npvMatch) {
    metrics.npv = parseFloat(npvMatch[1].replace(/,/g, ''))
  }

  // IRR
  const irrMatch = content.match(/IRR[^\d]*([\d]+(?:\.\d+)?)\s*%/i)
  if (irrMatch) {
    metrics.irr = parseFloat(irrMatch[1])
  }

  // CAPEX
  const capexMatch = content.match(/(?:capital cost|capex)[^\d]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)/i)
  if (capexMatch) {
    metrics.capex = parseFloat(capexMatch[1].replace(/,/g, ''))
  }

  // Resources
  const resourceMatch = content.match(/(?:resource|M\+I)[^\d]*([\d,]+(?:\.\d+)?)\s*(?:Mt|Moz)/i)
  if (resourceMatch) {
    metrics.resource = resourceMatch[0].substring(0, 50)
  }

  // Reserves
  const reserveMatch = content.match(/(?:reserve|P\+P)[^\d]*([\d,]+(?:\.\d+)?)\s*(?:Mt|Moz)/i)
  if (reserveMatch) {
    metrics.reserve = reserveMatch[0].substring(0, 50)
  }

  const validCount = Object.values(metrics).filter(v => v !== null).length
  return { metrics, isValid: validCount >= 2, validCount }
}

async function processCompany(ticker: string, name: string) {
  console.log(`\n🏢 Processing ${name} (${ticker})`)
  console.log('='.repeat(50))

  // Search for filings
  const filings = await searchFilings(ticker)

  if (filings.length === 0) {
    console.log('   ⚠️ No filings found')
    return
  }

  // Process first filing
  const filing = filings[0]
  console.log(`   📄 Processing: ${filing.title}`)

  if (!filing.documentUrl) {
    console.log('   ⚠️ No document URL')
    return
  }

  // Download document
  const pdfBuffer = await downloadPDF(filing.documentUrl)
  if (!pdfBuffer) return

  // Try to extract metrics (may be HTML or PDF)
  const content = pdfBuffer.toString('utf-8').substring(0, 500000) // First 500KB
  const { metrics, isValid, validCount } = extractMetricsFromContent(content)

  console.log(`   📊 Found ${validCount} metrics`)

  if (!isValid) {
    console.log('   ⚠️ Insufficient metrics')
    return
  }

  // Upload to storage
  const fileName = `${ticker}_${filing.filingDate || 'undated'}_${filing.accessionNo || uuidv4()}.pdf`
  const storagePath = `factset-documents/${fileName}`

  const { error: uploadError } = await supabase.storage
    .from('factset-documents')
    .upload(storagePath, pdfBuffer, {
      contentType: 'application/pdf',
      upsert: true
    })

  if (uploadError) {
    console.log(`   ❌ Upload failed: ${uploadError.message}`)
    return
  }

  console.log(`   ☁️ Uploaded to: ${storagePath}`)

  // Get company ID
  const { data: companies } = await supabase
    .from('companies')
    .select('id')
    .ilike('ticker', ticker)
    .limit(1)

  const companyId = companies?.[0]?.id || uuidv4()

  // Create project
  const projectName = filing.title?.replace(/technical report|ni 43-101/gi, '').trim() || `${name} Project`

  const { error: projectError } = await supabase
    .from('projects')
    .upsert({
      id: uuidv4(),
      company_id: companyId,
      name: projectName,
      npv: metrics.npv,
      irr: metrics.irr,
      capex: metrics.capex,
      resource: metrics.resource,
      reserve: metrics.reserve,
      document_storage_path: storagePath,
      urls: [filing.documentUrl],
      stage: 'Feasibility',
      status: 'Active',
      commodities: ['Gold'],
      description: `From ${filing.source} dated ${filing.filingDate}`,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    })

  if (projectError) {
    console.log(`   ❌ Project save failed: ${projectError.message}`)
  } else {
    console.log(`   ✅ Project created: ${projectName}`)
    if (metrics.npv) console.log(`      NPV: $${metrics.npv}M`)
    if (metrics.irr) console.log(`      IRR: ${metrics.irr}%`)
    if (metrics.capex) console.log(`      CAPEX: $${metrics.capex}M`)
  }
}

async function main() {
  console.log('🚀 FactSet Global Filings Extraction (Notebook-based approach)')
  console.log('📋 Using GET requests as per notebook examples')
  console.log('')

  // Ensure bucket exists
  const { data: buckets } = await supabase.storage.listBuckets()
  if (!buckets?.find(b => b.name === 'factset-documents')) {
    await supabase.storage.createBucket('factset-documents', { public: false })
    console.log('✅ Created storage bucket\n')
  }

  // Mining companies to process
  const companies = [
    { ticker: 'NEM', name: 'Newmont' },
    { ticker: 'ABX', name: 'Barrick Gold' },
    { ticker: 'AEM', name: 'Agnico Eagle' },
    { ticker: 'KGC', name: 'Kinross Gold' },
    { ticker: 'PAAS', name: 'Pan American Silver' }
  ]

  for (const company of companies) {
    await processCompany(company.ticker, company.name)
    await new Promise(r => setTimeout(r, 2000)) // Rate limiting
  }

  // Summary
  const { count } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .not('document_storage_path', 'is', null)

  console.log('\n' + '='.repeat(50))
  console.log(`✅ Complete! Projects with documents: ${count || 0}`)
  console.log('='.repeat(50))
}

main().catch(console.error)