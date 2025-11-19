#!/usr/bin/env npx tsx

import axios from 'axios'
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'
import fs from 'fs'
import crypto from 'crypto'
import * as pdfParse from 'pdf-parse'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Correct FactSet configuration
const FACTSET_CONFIG = {
  username: 'LITHOS-2220379',
  apiKey: '2BP7hqJOtH73DHvao1RJCQW0m1c4tVyOF81QVHxu',
  baseUrl: 'https://api.factset.com/content/global-filings/v2'
}

// Major mining companies with correct tickers
const MINING_COMPANIES = [
  { ticker: 'NEM-US', name: 'Newmont Corporation', source: 'EDG' },
  { ticker: 'FCX-US', name: 'Freeport-McMoRan', source: 'EDG' },
  { ticker: 'SCCO-US', name: 'Southern Copper', source: 'EDG' },
  { ticker: 'GOLD-US', name: 'Barrick Gold', source: 'EDG' },
  { ticker: 'AEM-US', name: 'Agnico Eagle Mines', source: 'EDG' },
  { ticker: 'KGC-US', name: 'Kinross Gold', source: 'EDG' }
]

async function searchForAnnualReports(ticker: string, source: string) {
  try {
    console.log(`\n🔍 Searching for ${ticker} annual reports...`)

    const url = `${FACTSET_CONFIG.baseUrl}/search`
    const params = {
      ids: ticker,
      sources: source,
      startDate: '20200101',
      endDate: '20241231',
      formTypes: '10-K'
    }

    const response = await axios.get(url, {
      params,
      auth: {
        username: FACTSET_CONFIG.username,
        password: FACTSET_CONFIG.apiKey
      },
      headers: {
        'Accept': 'application/json'
      }
    })

    if (response.data?.data?.[0]?.documents) {
      const documents = response.data.data[0].documents
      console.log(`  📄 Found ${documents.length} annual report(s)`)
      return documents
    }

    return []
  } catch (error: any) {
    console.error(`  ❌ Search error:`, error.response?.data || error.message)
    return []
  }
}

async function downloadDocument(url: string): Promise<Buffer | null> {
  try {
    console.log(`  📥 Downloading document...`)

    // Download with Basic Auth
    const response = await axios.get(url, {
      auth: {
        username: FACTSET_CONFIG.username,
        password: FACTSET_CONFIG.apiKey
      },
      responseType: 'arraybuffer',
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 120000 // 2 minutes timeout
    })

    const buffer = Buffer.from(response.data)
    const sizeMB = (buffer.length / 1024 / 1024).toFixed(2)
    console.log(`  ✅ Downloaded ${sizeMB} MB`)
    return buffer
  } catch (error: any) {
    console.error(`  ❌ Download failed:`, error.message)
    return null
  }
}

async function extractProjectData(buffer: Buffer, companyName: string) {
  try {
    console.log(`  🔬 Extracting project data...`)

    // For HTML documents, convert to text
    let text = buffer.toString('utf-8')

    // Remove HTML tags if present
    if (text.includes('<html') || text.includes('<HTML')) {
      text = text.replace(/<[^>]*>/g, ' ')
    }

    const projectData: any[] = []

    // Look for mining project mentions
    const projectPatterns = [
      /([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:mine|project|operation|deposit)/gi,
      /(?:our|the)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:mine|project|operation)/gi
    ]

    const projects = new Set<string>()

    for (const pattern of projectPatterns) {
      const matches = text.matchAll(pattern)
      for (const match of matches) {
        const projectName = match[1]?.trim()
        if (projectName && projectName.length > 3 && projectName.length < 50) {
          projects.add(projectName)
        }
      }
    }

    // Extract financial metrics for each project
    for (const projectName of Array.from(projects).slice(0, 5)) { // Limit to 5 projects
      console.log(`    📍 Found project: ${projectName}`)

      const projectText = text.substring(
        Math.max(0, text.indexOf(projectName) - 2000),
        Math.min(text.length, text.indexOf(projectName) + 2000)
      )

      const projectInfo: any = {
        name: projectName,
        company: companyName
      }

      // Extract metrics
      const npvMatch = projectText.match(/npv[^\$]*\$?([\d,]+(?:\.\d+)?)\s*(?:million|billion)/i)
      if (npvMatch) {
        const value = parseFloat(npvMatch[1].replace(/,/g, ''))
        const multiplier = projectText.includes('billion') ? 1000 : 1
        projectInfo.npv = value * multiplier
      }

      const productionMatch = projectText.match(/([\d,]+(?:\.\d+)?)\s*(?:thousand|million)?\s*(?:ounces|oz|tonnes|tons)/i)
      if (productionMatch) {
        projectInfo.production = productionMatch[0]
      }

      const locationMatch = projectText.match(/(?:located in|situated in|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)/i)
      if (locationMatch) {
        projectInfo.location = locationMatch[1]
      }

      if (Object.keys(projectInfo).length > 2) {
        projectData.push(projectInfo)
      }
    }

    console.log(`  ✅ Extracted data for ${projectData.length} projects`)
    return projectData
  } catch (error: any) {
    console.error(`  ❌ Extraction error:`, error.message)
    return []
  }
}

async function uploadToSupabase(buffer: Buffer, filename: string) {
  try {
    console.log(`  ☁️ Uploading to Supabase...`)

    const bucketName = 'factset-documents'
    const filePath = `10k/${filename}`

    // Ensure bucket exists
    const { data: buckets } = await supabase.storage.listBuckets()
    if (!buckets?.find(b => b.name === bucketName)) {
      await supabase.storage.createBucket(bucketName, { public: false })
    }

    const { data, error } = await supabase.storage
      .from(bucketName)
      .upload(filePath, buffer, {
        contentType: 'text/html',
        upsert: true
      })

    if (error) throw error

    console.log(`  ✅ Uploaded: ${filePath}`)
    return filePath
  } catch (error: any) {
    console.error(`  ❌ Upload error:`, error.message)
    return null
  }
}

async function main() {
  console.log('=' .repeat(80))
  console.log('🚀 FACTSET 10-K EXTRACTION PIPELINE')
  console.log('=' .repeat(80))

  let totalDocuments = 0
  let totalProjects = 0

  for (const company of MINING_COMPANIES) {
    console.log('\n' + '='.repeat(60))
    console.log(`Processing: ${company.name} (${company.ticker})`)
    console.log('='.repeat(60))

    // Search for 10-K reports
    const reports = await searchForAnnualReports(company.ticker, company.source)

    if (reports.length === 0) {
      console.log('  ⚠️ No annual reports found')
      continue
    }

    // Process the most recent report
    const report = reports[0]
    totalDocuments++

    if (!report.filingsLink) {
      console.log('  ⚠️ No download link')
      continue
    }

    // Download the document
    const buffer = await downloadDocument(report.filingsLink)
    if (!buffer) continue

    // Extract project data
    const projects = await extractProjectData(buffer, company.name)

    // Upload to Supabase
    const filename = `${company.ticker}_${report.documentId || crypto.randomUUID()}.html`
    const storagePath = await uploadToSupabase(buffer, filename)

    // Create database entries for projects
    if (projects.length > 0) {
      // Find or create company
      const { data: companies } = await supabase
        .from('companies')
        .select('id')
        .eq('ticker', company.ticker)
        .limit(1)

      let companyId = companies?.[0]?.id

      if (!companyId) {
        const { data: newCompany } = await supabase
          .from('companies')
          .insert({
            name: company.name,
            ticker: company.ticker,
            exchange: company.source
          })
          .select('id')
          .single()

        companyId = newCompany?.id
      }

      // Create projects
      for (const project of projects) {
        if (companyId) {
          totalProjects++

          const { error } = await supabase
            .from('projects')
            .insert({
              id: crypto.randomUUID(),
              company_id: companyId,
              name: project.name,
              location: project.location || null,
              npv: project.npv || null,
              description: `Mining project from ${company.name} 10-K filing. ${project.production ? `Production: ${project.production}` : ''}`,
              document_storage_path: storagePath,
              urls: [report.filingsLink],
              stage: 'Production',
              status: 'Active'
            })

          if (error) {
            console.error(`  ❌ Database error:`, error.message)
          } else {
            console.log(`  ✅ Created project: ${project.name}`)
          }
        }
      }
    }
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 EXTRACTION COMPLETE')
  console.log('='.repeat(80))
  console.log(`Documents processed: ${totalDocuments}`)
  console.log(`Projects created: ${totalProjects}`)

  // Show database status
  const { count: projectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  const { count: companyCount } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true })

  console.log(`\n📈 Database Status:`)
  console.log(`  Companies: ${companyCount}`)
  console.log(`  Projects: ${projectCount}`)
}

main().catch(console.error)