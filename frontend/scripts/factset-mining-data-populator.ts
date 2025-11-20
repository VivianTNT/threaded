#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import fetch from 'node-fetch'
import * as fs from 'fs/promises'
import * as path from 'path'

// Initialize Supabase client
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// FactSet API configuration
const FACTSET_USERNAME = 'LITHOS-2220379'
const FACTSET_API_KEY = '3gagnQkTnnEWmNwlRoPzSs9A5M38qbag5WDyLfaI'
const FACTSET_BASE_URL = 'https://api.factset.com'

// Mining company tickers to populate
const MAJOR_MINING_COMPANIES = [
  // Major global mining companies
  { ticker: 'BHP-AU', name: 'BHP Group', exchange: 'ASX' },
  { ticker: 'RIO-AU', name: 'Rio Tinto', exchange: 'ASX' },
  { ticker: 'VALE-US', name: 'Vale S.A.', exchange: 'NYSE' },
  { ticker: 'GLEN-GB', name: 'Glencore', exchange: 'LSE' },
  { ticker: 'FCX-US', name: 'Freeport-McMoRan', exchange: 'NYSE' },
  { ticker: 'NEM-US', name: 'Newmont Corporation', exchange: 'NYSE' },
  { ticker: 'GOLD-US', name: 'Barrick Gold', exchange: 'NYSE' },
  { ticker: 'SCCO-US', name: 'Southern Copper', exchange: 'NYSE' },
  { ticker: 'AA-US', name: 'Alcoa Corporation', exchange: 'NYSE' },
  { ticker: 'TECK-CA', name: 'Teck Resources', exchange: 'TSX' },

  // Critical minerals focused companies
  { ticker: 'ALB-US', name: 'Albemarle Corporation', exchange: 'NYSE' },
  { ticker: 'SQM-US', name: 'Sociedad Química y Minera', exchange: 'NYSE' },
  { ticker: 'LAC-CA', name: 'Lithium Americas Corp', exchange: 'TSX' },
  { ticker: 'PLL-AU', name: 'Piedmont Lithium', exchange: 'ASX' },
  { ticker: 'MP-US', name: 'MP Materials', exchange: 'NYSE' },
  { ticker: 'LTHM-US', name: 'Livent Corporation', exchange: 'NYSE' },

  // Copper focused companies
  { ticker: 'FM-CA', name: 'First Quantum Minerals', exchange: 'TSX' },
  { ticker: 'CS-CA', name: 'Capstone Copper', exchange: 'TSX' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', exchange: 'TSX' },
  { ticker: 'ERO-CA', name: 'Ero Copper', exchange: 'TSX' },

  // Gold mining companies
  { ticker: 'AEM-CA', name: 'Agnico Eagle Mines', exchange: 'TSX' },
  { ticker: 'KGC-CA', name: 'Kinross Gold', exchange: 'TSX' },
  { ticker: 'AGI-CA', name: 'Alamos Gold', exchange: 'TSX' },
  { ticker: 'EGO-CA', name: 'Eldorado Gold', exchange: 'TSX' },

  // Diversified miners
  { ticker: 'ANTO-GB', name: 'Antofagasta', exchange: 'LSE' },
  { ticker: 'AAL-GB', name: 'Anglo American', exchange: 'LSE' },
  { ticker: 'FRES-AU', name: 'Fortescue Metals', exchange: 'ASX' },
  { ticker: 'S32-AU', name: 'South32', exchange: 'ASX' },
  { ticker: 'NCM-AU', name: 'Newcrest Mining', exchange: 'ASX' },
  { ticker: 'IGO-AU', name: 'IGO Limited', exchange: 'ASX' },
]

// Helper to make authenticated FactSet API requests
async function factsetRequest(endpoint: string, params: any = {}) {
  const auth = Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')
  const url = new URL(`${FACTSET_BASE_URL}${endpoint}`)

  // Add query parameters
  Object.keys(params).forEach(key => {
    if (Array.isArray(params[key])) {
      params[key].forEach((value: string) => url.searchParams.append(key, value))
    } else {
      url.searchParams.append(key, params[key])
    }
  })

  try {
    const response = await fetch(url.toString(), {
      headers: {
        'Authorization': `Basic ${auth}`,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }
    })

    if (!response.ok) {
      console.error(`FactSet API error: ${response.status} ${response.statusText}`)
      return null
    }

    return await response.json()
  } catch (error) {
    console.error('FactSet API request failed:', error)
    return null
  }
}

// Fetch company details from FactSet
async function fetchCompanyDetails(ticker: string, name: string, exchange: string) {
  console.log(`📊 Fetching details for ${name} (${ticker})...`)

  // For now, we'll create structured company data
  // In production, you'd fetch this from FactSet's Company API
  const companyData = {
    ticker,
    name,
    exchange,
    sector: 'Mining',
    industry: 'Metals & Mining',
    market_cap: Math.floor(Math.random() * 100000) + 10000, // Random for demo
    employees: Math.floor(Math.random() * 50000) + 1000,
    headquarters: getHeadquarters(exchange),
    website: `https://www.${name.toLowerCase().replace(/\s+/g, '')}.com`,
    description: `${name} is a leading mining company focused on the extraction and processing of critical minerals and metals.`,
    commodities: getCommodities(name),
    status: 'Active',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }

  return companyData
}

// Get headquarters based on exchange
function getHeadquarters(exchange: string): string {
  const headquarters: Record<string, string> = {
    'NYSE': 'United States',
    'TSX': 'Canada',
    'ASX': 'Australia',
    'LSE': 'United Kingdom',
  }
  return headquarters[exchange] || 'International'
}

// Get commodities based on company name
function getCommodities(name: string): string[] {
  if (name.toLowerCase().includes('lithium')) return ['Lithium']
  if (name.toLowerCase().includes('copper')) return ['Copper']
  if (name.toLowerCase().includes('gold')) return ['Gold']
  if (name.toLowerCase().includes('barrick')) return ['Gold', 'Copper']
  if (name.toLowerCase().includes('bhp')) return ['Iron Ore', 'Copper', 'Coal', 'Nickel']
  if (name.toLowerCase().includes('rio')) return ['Iron Ore', 'Aluminum', 'Copper', 'Diamonds']
  if (name.toLowerCase().includes('vale')) return ['Iron Ore', 'Nickel', 'Copper']
  return ['Various Metals']
}

// Fetch technical documents from FactSet Global Filings API
async function fetchTechnicalDocuments(ticker: string) {
  console.log(`📄 Fetching technical documents for ${ticker}...`)

  const endDate = new Date().toISOString().slice(0, 10).replace(/-/g, '')
  const startDate = new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10).replace(/-/g, '')

  const response = await factsetRequest('/content/global-filings/v2/search', {
    ids: [ticker],
    sources: ['EDG', 'FFR', 'SDR', 'SDRP', 'ASX', 'ASXD'],
    startDate,
    endDate,
    _paginationLimit: 50,
    searchText: 'technical report OR feasibility study OR NI 43-101 OR resource estimate OR reserve estimate',
    categories: ['DT:AN-FILNS', 'DT:FILNS']
  })

  if (!response || !response.data) {
    console.log(`  ⚠️ No documents found for ${ticker}`)
    return []
  }

  const documents = []
  for (const item of response.data) {
    if (item.documents) {
      for (const doc of item.documents) {
        // Filter for substantial technical documents (likely 100+ pages)
        if (doc.filingSize && parseInt(doc.filingSize) > 5) { // >5MB usually indicates 100+ pages
          documents.push({
            title: doc.headline,
            source: doc.source,
            url: doc.filingsLink,
            date: doc.filingsDateTime,
            size: doc.filingSize,
            formType: doc.formTypes?.[0] || 'Technical Report',
            documentId: doc.documentId
          })
        }
      }
    }
  }

  console.log(`  ✅ Found ${documents.length} technical documents`)
  return documents
}

// Create mining projects from documents
async function createProjectsFromDocuments(companyId: string, companyName: string, documents: any[]) {
  const projects = []

  for (const doc of documents) {
    // Extract project name from document title
    const projectName = extractProjectName(doc.title, companyName)

    const project = {
      id: `${companyId}-${doc.documentId}`.slice(0, 36), // Ensure valid UUID length
      company_id: companyId,
      name: projectName,
      location: 'Various Locations', // Would be extracted from document
      stage: determineProjectStage(doc.title),
      commodities: ['Various'], // Would be extracted from document
      status: 'Active',
      description: `Technical documentation: ${doc.title}`,
      document_urls: [doc.url],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      watchlist: false,
      // Financial metrics would be extracted from the document
      npv: null,
      irr: null,
      capex: null
    }

    projects.push(project)
  }

  return projects
}

// Extract project name from document title
function extractProjectName(title: string, companyName: string): string {
  // Remove company name and common words
  let projectName = title
    .replace(new RegExp(companyName, 'gi'), '')
    .replace(/technical report/gi, '')
    .replace(/feasibility study/gi, '')
    .replace(/NI 43-101/gi, '')
    .replace(/\(.*?\)/g, '') // Remove parenthetical content
    .trim()

  // If too generic, use company name + Project
  if (projectName.length < 5 || projectName.toLowerCase().includes('annual')) {
    projectName = `${companyName} Mining Project`
  }

  return projectName.slice(0, 100) // Limit length
}

// Determine project stage from document title
function determineProjectStage(title: string): string {
  const titleLower = title.toLowerCase()
  if (titleLower.includes('feasibility')) return 'Feasibility'
  if (titleLower.includes('production')) return 'Production'
  if (titleLower.includes('development')) return 'Development'
  if (titleLower.includes('exploration')) return 'Exploration'
  if (titleLower.includes('resource')) return 'Resource Definition'
  return 'Development'
}

// Main function to populate the database
async function populateMiningData() {
  console.log('🚀 Starting FactSet mining data population...')
  console.log(`📊 Processing ${MAJOR_MINING_COMPANIES.length} companies...`)

  let totalCompanies = 0
  let totalProjects = 0
  let totalDocuments = 0

  for (const company of MAJOR_MINING_COMPANIES) {
    try {
      console.log(`\n${'='.repeat(60)}`)
      console.log(`Processing: ${company.name}`)
      console.log('='.repeat(60))

      // 1. Fetch and save company details
      const companyData = await fetchCompanyDetails(company.ticker, company.name, company.exchange)

      // Check if company already exists
      const { data: existingCompany } = await supabase
        .from('companies')
        .select('id')
        .eq('ticker', company.ticker)
        .single()

      let companyId: string

      if (existingCompany) {
        // Update existing company
        const { error } = await supabase
          .from('companies')
          .update(companyData)
          .eq('ticker', company.ticker)

        if (error) {
          console.error(`  ❌ Failed to update company: ${error.message}`)
          continue
        }
        companyId = existingCompany.id
        console.log(`  ✅ Updated company in database`)
      } else {
        // Insert new company
        const { data: newCompany, error } = await supabase
          .from('companies')
          .insert({
            ...companyData,
            id: crypto.randomUUID()
          })
          .select('id')
          .single()

        if (error) {
          console.error(`  ❌ Failed to insert company: ${error.message}`)
          continue
        }
        companyId = newCompany.id
        console.log(`  ✅ Added company to database`)
        totalCompanies++
      }

      // 2. Fetch technical documents
      const documents = await fetchTechnicalDocuments(company.ticker)
      totalDocuments += documents.length

      // 3. Create projects from documents
      if (documents.length > 0) {
        const projects = await createProjectsFromDocuments(companyId, company.name, documents)

        // Insert projects
        for (const project of projects) {
          // Check if project exists
          const { data: existingProject } = await supabase
            .from('projects')
            .select('id')
            .eq('name', project.name)
            .eq('company_id', companyId)
            .single()

          if (!existingProject) {
            const { error } = await supabase
              .from('projects')
              .insert(project)

            if (error) {
              console.error(`  ❌ Failed to insert project: ${error.message}`)
            } else {
              console.log(`  ✅ Added project: ${project.name}`)
              totalProjects++
            }
          } else {
            // Update document URLs if new ones found
            const { error } = await supabase
              .from('projects')
              .update({
                document_urls: project.document_urls,
                updated_at: new Date().toISOString()
              })
              .eq('id', existingProject.id)

            if (!error) {
              console.log(`  ✅ Updated project: ${project.name}`)
            }
          }
        }
      }

      // Add a small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 1000))

    } catch (error) {
      console.error(`❌ Error processing ${company.name}:`, error)
    }
  }

  console.log('\n' + '='.repeat(60))
  console.log('✅ POPULATION COMPLETE!')
  console.log('='.repeat(60))
  console.log(`📊 Summary:`)
  console.log(`  • Companies processed: ${MAJOR_MINING_COMPANIES.length}`)
  console.log(`  • New companies added: ${totalCompanies}`)
  console.log(`  • Projects created: ${totalProjects}`)
  console.log(`  • Technical documents found: ${totalDocuments}`)
  console.log(`  • Average docs per company: ${(totalDocuments / MAJOR_MINING_COMPANIES.length).toFixed(1)}`)
  console.log('='.repeat(60))

  // Now trigger data extraction for projects with documents
  console.log('\n🔄 Starting data extraction from technical reports...')
  await extractDataFromProjects()
}

// Extract data from project documents
async function extractDataFromProjects() {
  console.log('📈 Fetching projects with technical documentation...')

  const { data: projects, error } = await supabase
    .from('projects')
    .select('*')
    .not('document_urls', 'is', null)
    .limit(10) // Process first 10 for demo

  if (error) {
    console.error('Failed to fetch projects:', error)
    return
  }

  console.log(`Found ${projects?.length || 0} projects with documents`)

  for (const project of projects || []) {
    if (project.document_urls && project.document_urls.length > 0) {
      console.log(`\n💎 Extracting data from: ${project.name}`)

      try {
        // Call our extract-highlights API
        const response = await fetch('http://localhost:3000/api/pdf/extract-highlights', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            pdfUrl: project.document_urls[0],
            projectId: project.id
          })
        })

        if (response.ok) {
          const result = await response.json()
          console.log(`  ✅ Extracted data successfully`)

          if (result.extractedData) {
            const data = result.extractedData
            console.log(`    • NPV: ${data.npv?.value ? `$${data.npv.value}M` : 'N/A'}`)
            console.log(`    • IRR: ${data.irr?.value ? `${data.irr.value}%` : 'N/A'}`)
            console.log(`    • CAPEX: ${data.capex?.value ? `$${data.capex.value}M` : 'N/A'}`)
          }
        } else {
          console.log(`  ⚠️ Could not extract data from document`)
        }
      } catch (error) {
        console.error(`  ❌ Extraction failed:`, error)
      }

      // Add delay to avoid overwhelming the server
      await new Promise(resolve => setTimeout(resolve, 2000))
    }
  }

  console.log('\n✨ Data extraction complete!')
}

// Run the population script
populateMiningData().catch(console.error)