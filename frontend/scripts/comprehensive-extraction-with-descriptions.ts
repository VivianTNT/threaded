#!/usr/bin/env npx tsx
/**
 * Comprehensive PDF Data Extraction with Project Descriptions
 * Extracts NPV, IRR, CAPEX, Location, Commodity, Resources, Reserves, and Description
 */

import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'
import { extractText, getDocumentProxy } from 'unpdf'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

const SECTION_PATTERNS = {
  economics: /economic\s+analysis|financial\s+analysis|project\s+economics|economic\s+evaluation/i,
  summary: /executive\s+summary|summary\s+of\s+results|project\s+summary/i,
  npv: /net\s+present\s+value|npv|post-tax\s+npv|after-tax\s+npv/i,
  irr: /internal\s+rate\s+of\s+return|irr/i,
  capex: /capital\s+cost|initial\s+capital|capex|capital\s+expenditure/i,
  location: /location|property\s+location|geographic\s+setting/i,
  resources: /mineral\s+resource|resource\s+estimate/i,
  reserves: /mineral\s+reserve|reserve\s+estimate|ore\s+reserve/i,
}

async function extractPDFText(pdfUrl: string): Promise<{ text: string; pageCount: number }> {
  try {
    console.log(`      📥 Downloading PDF...`)
    const response = await fetch(pdfUrl)
    const arrayBuffer = await response.arrayBuffer()
    const buffer = new Uint8Array(arrayBuffer)

    console.log(`      📖 Extracting text with unpdf...`)
    const pdf = await getDocumentProxy(buffer)
    const { totalPages, text } = await extractText(pdf, { mergePages: true })

    console.log(`      ✅ Extracted ${totalPages} pages, ${(text.length / 1000).toFixed(0)}KB text`)
    return { text, pageCount: totalPages }
  } catch (error) {
    console.log(`      ❌ PDF extraction error: ${error}`)
    throw error
  }
}

function findRelevantSections(fullText: string): string {
  const chunkSize = 5000
  const chunks: string[] = []

  for (let i = 0; i < fullText.length; i += chunkSize) {
    chunks.push(fullText.substring(i, i + chunkSize))
  }

  const relevantChunks = chunks.filter((chunk, idx) => {
    const hasEconomics = SECTION_PATTERNS.economics.test(chunk)
    const hasSummary = SECTION_PATTERNS.summary.test(chunk)
    const hasNPV = SECTION_PATTERNS.npv.test(chunk)
    const hasIRR = SECTION_PATTERNS.irr.test(chunk)
    const hasCAPEX = SECTION_PATTERNS.capex.test(chunk)

    const score = [hasEconomics, hasSummary, hasNPV, hasIRR, hasCAPEX].filter(Boolean).length

    if (score >= 2) {
      console.log(`      📍 Found relevant section at position ${idx} (score: ${score}/5)`)
      return true
    }
    return false
  })

  const combinedRelevant = relevantChunks.join('\n').substring(0, 30000)

  if (combinedRelevant.length < 1000) {
    console.log(`      ⚠️  No specific sections found, using first 30KB`)
    return fullText.substring(0, 30000)
  }

  console.log(`      ✅ Found ${relevantChunks.length} relevant sections, ${(combinedRelevant.length / 1000).toFixed(0)}KB total`)
  return combinedRelevant
}

async function extractDataWithAI(text: string, projectName: string) {
  try {
    console.log(`      🤖 Analyzing with OpenAI GPT-4...`)

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `You are a mining financial analyst extracting data from NI 43-101 technical reports. Extract the following from this ${projectName} project report:

1. NPV (Net Present Value) - in USD millions, post-tax/after-tax preferred
2. IRR (Internal Rate of Return) - as percentage
3. CAPEX (Capital Expenditure / Initial Capital Cost) - in USD millions
4. Location - Country and Province/State/Region
5. Commodity - Primary mineral (e.g., Gold, Copper, Silver, Nickel)
6. Resources - Measured, Indicated, Inferred resource estimates with tonnage and grade
7. Reserves - Proven, Probable reserve estimates with tonnage and grade
8. Description - 2-3 sentence project overview including mine type (open pit/underground), operation status, key features, and production rate

Return ONLY valid JSON:
{
  "npv": number | null,
  "irr": number | null,
  "capex": number | null,
  "location": "Country, Region" | null,
  "commodity": "Primary Mineral" | null,
  "resources": "Resource estimates description" | null,
  "reserves": "Reserve estimates description" | null,
  "description": "Project description" | null
}

If a value is not found, use null. For resources/reserves, provide a concise summary with tonnage and grade.`
        },
        {
          role: 'user',
          content: text
        }
      ],
      response_format: { type: 'json_object' },
      temperature: 0.1
    })

    const result = JSON.parse(response.choices[0].message.content || '{}')
    console.log(`      ✅ AI extraction complete`)

    return {
      npv: result.npv || result.NPV || null,
      irr: result.irr || result.IRR || null,
      capex: result.capex || result.CAPEX || null,
      location: result.location || null,
      commodity: result.commodity || null,
      resources: result.resources || null,
      reserves: result.reserves || null,
      description: result.description || null
    }
  } catch (error) {
    console.log(`      ❌ AI extraction failed: ${error}`)
    return {
      npv: null,
      irr: null,
      capex: null,
      location: null,
      commodity: null,
      resources: null,
      reserves: null,
      description: null
    }
  }
}

async function main() {
  console.log('🚀 COMPREHENSIVE PDF DATA EXTRACTION WITH DESCRIPTIONS')
  console.log('='.repeat(80))

  const { data: projects } = await supabase
    .from('projects')
    .select('*')
    .not('urls', 'is', null)
    .order('name')

  if (!projects || projects.length === 0) {
    console.log('No projects found')
    return
  }

  console.log(`Found ${projects.length} projects with PDFs\n`)

  let extracted = 0
  let failed = 0

  for (const project of projects) {
    console.log(`\n📊 ${project.name}`)

    if (!project.urls || project.urls.length === 0) {
      console.log(`   ⚠️  No URLs`)
      failed++
      continue
    }

    const pdfUrl = project.urls[0]
    const fileName = pdfUrl.substring(pdfUrl.lastIndexOf('/') + 1)
    console.log(`   📄 ${fileName}`)

    try {
      const { text } = await extractPDFText(pdfUrl)
      const relevantText = findRelevantSections(text)
      const data = await extractDataWithAI(relevantText, project.name)

      const updates: any = {}

      if (data.npv && !project.post_tax_npv_usd_m) {
        updates.post_tax_npv_usd_m = data.npv
        console.log(`      💰 NPV: $${data.npv}M`)
      }

      if (data.irr && !project.irr_percent) {
        updates.irr_percent = data.irr
        console.log(`      📈 IRR: ${data.irr}%`)
      }

      if (data.capex && !project.capex_usd_m) {
        updates.capex_usd_m = data.capex
        console.log(`      🏗️  CAPEX: $${data.capex}M`)
      }

      if (data.location) {
        updates.country = data.location
        console.log(`      📍 Location: ${data.location}`)
      }

      if (data.commodity) {
        updates.commodity = data.commodity
        console.log(`      ⛏️  Commodity: ${data.commodity}`)
      }

      if (data.resources) {
        updates.resources = typeof data.resources === 'string' ? data.resources : String(data.resources)
        const preview = updates.resources.substring(0, 80)
        console.log(`      📊 Resources: ${preview}...`)
      }

      if (data.reserves) {
        updates.reserves = typeof data.reserves === 'string' ? data.reserves : String(data.reserves)
        const preview = updates.reserves.substring(0, 80)
        console.log(`      📊 Reserves: ${preview}...`)
      }

      if (data.description) {
        updates.description = data.description
        const preview = data.description.substring(0, 100)
        console.log(`      📝 Description: ${preview}...`)
      }

      if (Object.keys(updates).length > 0) {
        await supabase
          .from('projects')
          .update(updates)
          .eq('id', project.id)

        console.log(`      ✅ Updated ${Object.keys(updates).length} fields in database`)
        extracted++
      } else {
        console.log(`      ⚠️  No new data extracted`)
        failed++
      }

    } catch (error) {
      console.log(`      ❌ Error: ${error}`)
      failed++
    }

    await new Promise(r => setTimeout(r, 3000))
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ Extracted: ${extracted}`)
  console.log(`❌ Failed: ${failed}`)
  console.log(`📊 Total: ${projects.length}`)
  console.log(`${'='.repeat(80)}`)
}

main().catch(console.error)
