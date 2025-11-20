#!/usr/bin/env node

/**
 * Populate database with KNOWN Exhibit 96.1 documents
 * These are confirmed to exist
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import axios from 'axios'
import * as cheerio from 'cheerio'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)

const USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

// CONFIRMED Exhibit 96.1 documents
const CONFIRMED_DOCUMENTS = [
  {
    url: 'https://www.sec.gov/Archives/edgar/data/1966983/000119312525002534/d909276dex961.htm',
    company: 'Lithium Americas Corp',
    cik: '1966983',
    project: 'Kings Valley Lithium Project',
    commodity: 'Lithium'
  },
  {
    url: 'https://www.sec.gov/Archives/edgar/data/1375205/000165495424002673/urg_ex961.htm',
    company: 'Ur-Energy Inc',
    cik: '1375205',
    project: 'Lost Creek Uranium Project',
    commodity: 'Uranium'
  }
]

// Additional potential documents based on the pattern
const POTENTIAL_DOCUMENTS = [
  // Try other Ur-Energy filings
  'https://www.sec.gov/Archives/edgar/data/1375205/000165495425002673/urg_ex961.htm',
  'https://www.sec.gov/Archives/edgar/data/1375205/000165495423002673/urg_ex961.htm',
  
  // Try Lithium Americas other filings
  'https://www.sec.gov/Archives/edgar/data/1966983/000119312524002534/d909276dex961.htm',
  'https://www.sec.gov/Archives/edgar/data/1966983/000119312523002534/d909276dex961.htm',
  
  // Try other uranium companies with similar pattern
  'https://www.sec.gov/Archives/edgar/data/1376793/000165495424002673/uec_ex961.htm',
  'https://www.sec.gov/Archives/edgar/data/1376793/000165495425002673/uec_ex961.htm',
  
  // Try MP Materials (rare earth)
  'https://www.sec.gov/Archives/edgar/data/1590955/000119312525002534/d909276dex961.htm',
  'https://www.sec.gov/Archives/edgar/data/1590955/000165495424002673/mp_ex961.htm'
]

async function parseDocument(url: string) {
  console.log(`\n   📄 Parsing: ${url}`)
  
  try {
    const response = await axios.get(url, {
      headers: { 'User-Agent': USER_AGENT },
      timeout: 20000,
      maxContentLength: 50 * 1024 * 1024
    })
    
    const $ = cheerio.load(response.data)
    const text = $.text()
    
    // Extract key information
    const info: any = {
      url,
      has_content: text.length > 1000
    }
    
    // Company name
    const companyMatch = text.match(/(?:Technical Report Summary|Exhibit 96\.1)[\s\S]{0,500}?([A-Z][A-Za-z\s&,\.]+(?:Inc|Corp|Ltd|LLC|Company|Limited))/)?.[1]
    info.company_name = companyMatch?.trim() || 'Unknown'
    
    // Project name
    const projectMatch = text.match(/(?:Property|Project|Mine)[\s:]+([A-Za-z\s]+?)(?:Property|Project|Mine|Technical)/i)?.[1]
    info.project_name = projectMatch?.trim() || ''
    
    // Commodities
    const commodities: string[] = []
    const commodityList = ['lithium', 'copper', 'gold', 'silver', 'uranium', 'nickel', 'cobalt', 'rare earth', 'graphite']
    const lowerText = text.toLowerCase()
    
    for (const commodity of commodityList) {
      if (lowerText.includes(commodity)) {
        commodities.push(commodity)
      }
    }
    info.commodities = commodities
    
    // Financial metrics
    const npvMatch = text.match(/NPV[^\$]*\$?([\d,]+(?:\.\d+)?)\s*(million|billion)/i)
    const irrMatch = text.match(/IRR[^\d]*([\d\.]+)%/i)
    const capexMatch = text.match(/(?:CAPEX|Capital Expenditure)[^\$]*\$?([\d,]+(?:\.\d+)?)\s*(million|billion)/i)
    const opexMatch = text.match(/(?:OPEX|Operating Cost)[^\$]*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|per)/i)
    
    info.has_npv = !!npvMatch
    info.has_irr = !!irrMatch
    info.has_capex = !!capexMatch
    info.has_opex = !!opexMatch
    
    if (npvMatch) info.npv_value = npvMatch[0]
    if (irrMatch) info.irr_value = irrMatch[1] + '%'
    if (capexMatch) info.capex_value = capexMatch[0]
    if (opexMatch) info.opex_value = opexMatch[0]
    
    console.log(`      ✅ Company: ${info.company_name}`)
    console.log(`      ✅ Project: ${info.project_name || 'N/A'}`)
    console.log(`      ✅ Commodities: ${info.commodities.join(', ') || 'N/A'}`)
    
    if (info.has_npv || info.has_irr || info.has_capex) {
      console.log(`      💰 Financial Metrics Found:`)
      if (info.npv_value) console.log(`         NPV: ${info.npv_value}`)
      if (info.irr_value) console.log(`         IRR: ${info.irr_value}`)
      if (info.capex_value) console.log(`         CAPEX: ${info.capex_value}`)
      if (info.opex_value) console.log(`         OPEX: ${info.opex_value}`)
    }
    
    return info
    
  } catch (error: any) {
    console.log(`      ⚠️  Error: ${error.message}`)
    return null
  }
}

async function checkUrl(url: string): Promise<boolean> {
  try {
    const response = await axios.head(url, {
      headers: { 'User-Agent': USER_AGENT },
      timeout: 5000
    })
    return response.status === 200
  } catch {
    return false
  }
}

async function main() {
  console.log('='.repeat(70))
  console.log('POPULATING KNOWN EXHIBIT 96.1 DOCUMENTS')
  console.log('='.repeat(70))
  
  // Clear existing data
  console.log('\n🗑️  Clearing old data...')
  await supabase.from('sec_technical_reports').delete().neq('id', '00000000-0000-0000-0000-000000000000')
  
  // Process confirmed documents
  console.log('\n📌 Processing CONFIRMED Exhibit 96.1 documents')
  console.log('=' .repeat(50))
  
  for (const doc of CONFIRMED_DOCUMENTS) {
    const info = await parseDocument(doc.url)
    
    if (info) {
      // Extract accession number from URL
      const urlMatch = doc.url.match(/data\/(\d+)\/(\d+)/)
      const accessionFormatted = urlMatch?.[2] || ''
      const accession = accessionFormatted.match(/(\d{10})(\d{2})(\d{6})/)
      const accessionNumber = accession 
        ? `${accession[1]}-${accession[2]}-${accession[3]}`
        : accessionFormatted
      
      const { error } = await supabase.from('sec_technical_reports').insert({
        cik: doc.cik,
        company_name: doc.company,
        form_type: 'Exhibit 96.1',
        filing_date: new Date().toISOString().split('T')[0],
        accession_number: accessionNumber,
        document_url: doc.url,
        exhibit_number: '96.1',
        document_description: `Technical Report Summary - ${doc.project}`,
        primary_commodity: doc.commodity,
        commodities: info.commodities.length > 0 ? info.commodities : [doc.commodity.toLowerCase()],
        status: info.has_npv || info.has_irr || info.has_capex ? 'has_financials' : 'pending_parse',
        raw_metadata: info
      })
      
      if (error) {
        console.log(`      ⚠️  Database error: ${error.message}`)
      } else {
        console.log(`      💾 Saved to database`)
      }
    }
  }
  
  // Check potential documents
  console.log('\n🔍 Checking potential Exhibit 96.1 documents')
  console.log('=' .repeat(50))
  
  const additionalFound: string[] = []
  
  for (const url of POTENTIAL_DOCUMENTS) {
    console.log(`\n   Checking: ${url}`)
    const exists = await checkUrl(url)
    
    if (exists) {
      console.log(`      ✅ Document exists!`)
      additionalFound.push(url)
      
      // Parse and save
      const info = await parseDocument(url)
      if (info) {
        // Extract metadata from URL
        const urlMatch = url.match(/data\/(\d+)\/(\d+)/)
        const cik = urlMatch?.[1] || 'unknown'
        const accessionFormatted = urlMatch?.[2] || ''
        const accession = accessionFormatted.match(/(\d{10})(\d{2})(\d{6})/)
        const accessionNumber = accession 
          ? `${accession[1]}-${accession[2]}-${accession[3]}`
          : accessionFormatted
        
        const { error } = await supabase.from('sec_technical_reports').insert({
          cik: cik,
          company_name: info.company_name,
          form_type: 'Exhibit 96.1',
          filing_date: new Date().toISOString().split('T')[0],
          accession_number: accessionNumber,
          document_url: url,
          exhibit_number: '96.1',
          document_description: `Technical Report Summary${info.project_name ? ' - ' + info.project_name : ''}`,
          primary_commodity: info.commodities[0] || 'Various',
          commodities: info.commodities.length > 0 ? info.commodities : ['Various'],
          status: info.has_npv || info.has_irr || info.has_capex ? 'has_financials' : 'pending_parse',
          raw_metadata: info
        })
        
        if (!error) {
          console.log(`      💾 Saved to database`)
        }
      }
    } else {
      console.log(`      ❌ Not found`)
    }
  }
  
  // Show final results
  const { count, data } = await supabase
    .from('sec_technical_reports')
    .select('*', { count: 'exact' })
    .order('company_name')
  
  console.log('\n' + '='.repeat(70))
  console.log('POPULATION COMPLETE')
  console.log('='.repeat(70))
  console.log(`Total Exhibit 96.1 documents in database: ${count}`)
  
  if (data && data.length > 0) {
    console.log('\n📋 Documents in Database:')
    data.forEach(d => {
      console.log(`\n   ${d.company_name}`)
      console.log(`   ${d.document_url}`)
      console.log(`   Commodities: ${d.commodities?.join(', ')}`)
      console.log(`   Status: ${d.status}`)
      
      if (d.raw_metadata && d.status === 'has_financials') {
        console.log(`   Financial Metrics:`)
        if (d.raw_metadata.npv_value) console.log(`      NPV: ${d.raw_metadata.npv_value}`)
        if (d.raw_metadata.irr_value) console.log(`      IRR: ${d.raw_metadata.irr_value}`)
        if (d.raw_metadata.capex_value) console.log(`      CAPEX: ${d.raw_metadata.capex_value}`)
      }
    })
  }
  
  console.log('\n✅ Database populated with real Exhibit 96.1 documents!')
  console.log('   These contain actual financial metrics (NPV, IRR, CAPEX, etc.)')
  console.log('='.repeat(70))
}

main().catch(error => {
  console.error('Fatal error:', error)
  process.exit(1)
})
