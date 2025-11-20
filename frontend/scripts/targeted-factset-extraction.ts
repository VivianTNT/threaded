#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'
import crypto from 'crypto'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!)

const authHeader = 'Basic ' + Buffer.from(`${process.env.FACTSET_USERNAME}:${process.env.FACTSET_API_KEY}`).toString('base64')

// Target specific projects
const TARGET_PROJECTS = [
  { name: "Kansanshi", company: "First Quantum", ticker: "FM-CA" },
  { name: "Sentinel", company: "First Quantum", ticker: "FM-CA" },
  { name: "Cobre Panama", company: "First Quantum", ticker: "FM-CA" },
  { name: "Kamoa-Kakula", company: "Ivanhoe", ticker: "IVN-CA" },
  { name: "Platreef", company: "Ivanhoe", ticker: "IVN-CA" },
]

async function main() {
  console.log('🎯 TARGETED FACTSET EXTRACTION FOR SPECIFIC PROJECTS')
  console.log('='.repeat(80))
  
  for (const target of TARGET_PROJECTS) {
    console.log(`\n📊 ${target.name} (${target.company})`)
    console.log('-'.repeat(40))
    
    try {
      // Search FactSet
      const searchUrl = `https://api.factset.com/content/global-filings/v2/search?ids=${target.ticker}&sources=SDR&searchText=${encodeURIComponent(target.name + ' technical report ni 43-101')}&sort=-filingsDate&paginationLimit=5`
      
      const searchRes = await fetch(searchUrl, { headers: { Authorization: authHeader } })
      const searchData = await searchRes.json()
      
      const docs = searchData.data?.[0]?.documents || []
      
      if (docs.length === 0) {
        console.log('⚠️  No documents found')
        continue
      }
      
      const doc = docs[0]
      console.log(`📄 Found: ${doc.headline}`)
      console.log(`📅 Date: ${doc.filingsDate}`)
      console.log(`🔗 Link: ${doc.filingsLink}`)
      
      // Download PDF text via filingsLink (get first 50KB for quick extraction)
      const pdfRes = await fetch(doc.filingsLink, { 
        headers: { Authorization: authHeader, Accept: 'application/pdf' }
      })
      
      const pdfBuffer = Buffer.from(await pdfRes.arrayBuffer())
      const sizeMB = pdfBuffer.length / 1024 / 1024
      console.log(`📦 PDF Size: ${sizeMB.toFixed(2)} MB`)
      
      // Save to temp file for text extraction
      const fs = require('fs')
      const tempPath = `/tmp/${target.name.replace(/\s+/g, '_')}.pdf`
      fs.writeFileSync(tempPath, pdfBuffer)
      
      // Extract text using pdfjs (more reliable than pdf-parse)
      const pdfjsLib = require('pdfjs-dist/legacy/build/pdf.js')
      const loadingTask = pdfjsLib.getDocument({ data: pdfBuffer })
      const pdf = await loadingTask.promise
      
      let fullText = ''
      const maxPages = Math.min(pdf.numPages, 50) // First 50 pages for speed
      
      for (let i = 1; i <= maxPages; i++) {
        const page = await pdf.getPage(i)
        const textContent = await page.getTextContent()
        fullText += textContent.items.map((item: any) => item.str).join(' ') + '\n'
      }
      
      console.log(`📝 Extracted ${fullText.length} characters from ${maxPages} pages`)
      
      // Use OpenAI to extract financial metrics
      const OpenAI = require('openai').default
      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
      
      console.log('🤖 Extracting financial data with AI...')
      
      const prompt = `Extract mining project financial metrics from this technical report excerpt. Return ONLY a JSON object with these fields:
{
  "npv": <number in millions USD>,
  "irr": <number as percentage>,
  "capex": <number in millions USD>,
  "resource": <number in millions tonnes>,
  "reserve": <number in millions tonnes>,
  "commodity": "<primary commodity>"
}

If a value is not found, use null. Be precise with units.

Text excerpt:
${fullText.substring(0, 15000)}`

      const completion = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are a mining financial data extraction expert. Return only valid JSON.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.1,
        response_format: { type: 'json_object' }
      })
      
      const extracted = JSON.parse(completion.choices[0].message.content || '{}')
      
      console.log('💰 Extracted Data:')
      console.log(`   NPV: $${extracted.npv}M`)
      console.log(`   IRR: ${extracted.irr}%`)
      console.log(`   CAPEX: $${extracted.capex}M`)
      console.log(`   Resource: ${extracted.resource}Mt`)
      console.log(`   Reserve: ${extracted.reserve}Mt`)
      console.log(`   Commodity: ${extracted.commodity}`)
      
      // Find company in database
      const { data: companies } = await supabase
        .from('companies')
        .select('id')
        .ilike('name', `%${target.company}%`)
        .limit(1)
      
      if (!companies || companies.length === 0) {
        console.log('⚠️  Company not found in database')
        continue
      }
      
      // Insert/update project
      const projectId = crypto.randomUUID()
      const { error } = await supabase.from('projects').insert({
        id: projectId,
        company_id: companies[0].id,
        name: target.name,
        commodities: [extracted.commodity || 'Copper'],
        npv: extracted.npv || null,
        irr: extracted.irr || null,
        capex: extracted.capex || null,
        resource: extracted.resource || null,
        reserve: extracted.reserve || null,
        stage: 'Production',
        status: 'Active',
        urls: [doc.filingsLink],
        description: `${doc.headline.substring(0, 200)}. Date: ${doc.filingsDate}`,
        watchlist: false
      })
      
      if (error) {
        console.log(`⚠️  Database error: ${error.message}`)
      } else {
        console.log(`✅ Project saved to database!`)
      }
      
      // Clean up
      fs.unlinkSync(tempPath)
      
    } catch (err: any) {
      console.error(`❌ Error: ${err.message}`)
    }
  }
  
  console.log('\n' + '='.repeat(80))
  console.log('✅ TARGETED EXTRACTION COMPLETE')
  console.log('='.repeat(80))
}

main()
