#!/usr/bin/env npx tsx
/**
 * FINAL PDF DATA EXTRACTION PIPELINE
 *
 * This script:
 * 1. Fetches all projects with PDF URLs from Supabase
 * 2. Downloads and parses PDFs using unpdf library
 * 3. Uses regex to find relevant financial sections
 * 4. Uses OpenAI GPT-4-mini to extract structured data
 * 5. WRITES data to Supabase with CORRECT column names
 *
 * Columns populated:
 * - npv_usd_millions
 * - irr_percentage
 * - capex_usd_millions
 * - location
 * - commodities (array)
 * - resource_estimate
 * - reserve_estimate
 * - description
 * - financial_metrics_updated_at
 */

import { createClient } from '@supabase/supabase-js'
import { extractText, getDocumentProxy } from 'unpdf'
import OpenAI from 'openai'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

// Regex patterns to identify relevant sections
const SECTION_PATTERNS = {
  economics: /economic\s+analysis|financial\s+analysis|project\s+economics|economic\s+evaluation/i,
  summary: /executive\s+summary|summary\s+of\s+results|highlights|key\s+findings/i,
  npv: /net\s+present\s+value|npv|post-tax\s+npv|after-tax\s+npv|pre-tax\s+npv/i,
  irr: /internal\s+rate\s+of\s+return|irr/i,
  capex: /capital\s+cost|initial\s+capital|capex|capital\s+expenditure/i,
  resources: /mineral\s+resource|resource\s+estimate|measured.*indicated.*inferred/i,
  reserves: /mineral\s+reserve|reserve\s+estimate|proven.*probable/i,
  location: /location|jurisdiction|property\s+location|geographic\s+setting/i,
  commodity: /commodity|metal|mineral|deposit\s+type/i
}

/**
 * Extract text from PDF using unpdf
 */
async function extractPDFText(pdfUrl: string) {
  console.log(`   📥 Downloading PDF...`)

  try {
    const response = await fetch(pdfUrl)
    if (!response.ok) {
      throw new Error(`Failed to fetch PDF: ${response.status}`)
    }

    const arrayBuffer = await response.arrayBuffer()
    const buffer = new Uint8Array(arrayBuffer)

    console.log(`   📖 Parsing PDF with unpdf...`)
    const pdf = await getDocumentProxy(buffer)
    const { totalPages, text } = await extractText(pdf, { mergePages: true })

    console.log(`   ✅ Extracted ${totalPages} pages, ${(text.length / 1024).toFixed(1)}KB text`)
    return { text, pageCount: totalPages }
  } catch (error: any) {
    console.log(`   ❌ PDF extraction failed: ${error.message}`)
    return null
  }
}

/**
 * Find relevant sections using regex scoring
 */
function findRelevantSections(fullText: string): string {
  console.log(`   🔍 Finding relevant sections...`)

  const chunkSize = 5000
  const chunks: { text: string; score: number; index: number }[] = []

  for (let i = 0; i < fullText.length; i += chunkSize) {
    const chunk = fullText.substring(i, Math.min(i + chunkSize, fullText.length))

    // Score chunk based on keyword matches
    const score = [
      SECTION_PATTERNS.economics.test(chunk),
      SECTION_PATTERNS.summary.test(chunk),
      SECTION_PATTERNS.npv.test(chunk),
      SECTION_PATTERNS.irr.test(chunk),
      SECTION_PATTERNS.capex.test(chunk),
      SECTION_PATTERNS.resources.test(chunk),
      SECTION_PATTERNS.reserves.test(chunk)
    ].filter(Boolean).length

    if (score >= 2) {
      chunks.push({ text: chunk, score, index: i / chunkSize })
    }
  }

  // Sort by score (highest first)
  chunks.sort((a, b) => b.score - a.score)

  // Take top chunks (up to 30KB total)
  let totalLength = 0
  const relevantChunks: string[] = []

  for (const chunk of chunks) {
    if (totalLength + chunk.text.length > 30000) break
    relevantChunks.push(chunk.text)
    totalLength += chunk.text.length
  }

  console.log(`   📊 Found ${chunks.length} relevant chunks, using ${relevantChunks.length} (${(totalLength / 1024).toFixed(1)}KB)`)

  return relevantChunks.join('\n\n')
}

/**
 * Extract structured data using OpenAI
 */
async function extractDataWithAI(text: string, projectName: string) {
  console.log(`   🤖 Extracting data with OpenAI GPT-4-mini...`)

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `You are extracting financial and technical data from NI 43-101 technical reports for mining projects.

Extract the following data from the provided text for the project "${projectName}":

1. NPV (Net Present Value) - in millions USD (post-tax or after-tax preferred, but pre-tax acceptable)
2. IRR (Internal Rate of Return) - as percentage (post-tax or after-tax preferred)
3. CAPEX (Capital Expenditure) - in millions USD (initial capital or total capital costs)
4. Location - specific location (e.g., "Quebec, Canada" or "Nevada, USA")
5. Commodity - primary commodity or commodities (e.g., "Gold", "Copper", "Lithium")
6. Resource Estimate - brief summary (e.g., "M&I: 2.5M oz Au @ 1.2g/t, Inferred: 0.8M oz")
7. Reserve Estimate - brief summary (e.g., "P&P: 1.8M oz Au @ 1.1g/t")
8. Description - 2-3 sentence project description including mine type (open pit/underground), operation status, key features

Return a JSON object with these exact keys:
{
  "npv": number or null,
  "irr": number or null,
  "capex": number or null,
  "location": string or null,
  "commodity": string or string[] or null,
  "resource": string or null,
  "reserve": string or null,
  "description": string or null
}

Important:
- Return null for any value not found in the text
- For NPV, IRR, CAPEX: convert to numbers only (no currency symbols or units)
- For commodity: can be a single string or array of strings
- Keep resource/reserve estimates concise (max 200 chars each)
- Description should be 2-3 sentences maximum`
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
    console.log(`   ✅ AI extraction complete`)
    return result
  } catch (error: any) {
    console.log(`   ❌ OpenAI extraction failed: ${error.message}`)
    return null
  }
}

/**
 * Update project in database
 */
async function updateProject(projectId: string, data: any) {
  console.log(`   💾 Writing to database...`)

  const updates: any = {
    financial_metrics_updated_at: new Date().toISOString()
  }

  let fieldsUpdated = 0

  // NPV
  if (data.npv !== null && data.npv !== undefined) {
    updates.npv = data.npv
    console.log(`      💰 NPV: $${data.npv}M`)
    fieldsUpdated++
  }

  // IRR
  if (data.irr !== null && data.irr !== undefined) {
    updates.irr = data.irr
    console.log(`      📈 IRR: ${data.irr}%`)
    fieldsUpdated++
  }

  // CAPEX
  if (data.capex !== null && data.capex !== undefined) {
    updates.capex = data.capex
    console.log(`      🏗️  CAPEX: $${data.capex}M`)
    fieldsUpdated++
  }

  // Location
  if (data.location) {
    updates.location = data.location
    console.log(`      📍 Location: ${data.location}`)
    fieldsUpdated++
  }

  // Commodity (convert to array if needed)
  if (data.commodity) {
    updates.commodities = Array.isArray(data.commodity)
      ? data.commodity
      : [data.commodity]
    console.log(`      ⚡ Commodities: ${updates.commodities.join(', ')}`)
    fieldsUpdated++
  }

  // Resources
  if (data.resource) {
    updates.resource = data.resource
    const preview = data.resource.substring(0, 80)
    console.log(`      📊 Resources: ${preview}${data.resource.length > 80 ? '...' : ''}`)
    fieldsUpdated++
  }

  // Reserves
  if (data.reserve) {
    updates.reserve = data.reserve
    const preview = data.reserve.substring(0, 80)
    console.log(`      💎 Reserves: ${preview}${data.reserve.length > 80 ? '...' : ''}`)
    fieldsUpdated++
  }

  // Description
  if (data.description) {
    updates.description = data.description
    const preview = data.description.substring(0, 100)
    console.log(`      📝 Description: ${preview}${data.description.length > 100 ? '...' : ''}`)
    fieldsUpdated++
  }

  // Execute update
  const { error } = await supabase
    .from('projects')
    .update(updates)
    .eq('id', projectId)

  if (error) {
    console.log(`   ❌ Database update failed: ${error.message}`)
    return false
  }

  console.log(`   ✅ Updated ${fieldsUpdated} fields in database`)
  return true
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 FINAL PDF DATA EXTRACTION PIPELINE')
  console.log('='.repeat(80))

  // Fetch all projects with PDF URLs
  console.log('\n📊 Fetching projects from database...')
  const { data: projects, error } = await supabase
    .from('projects')
    .select('*')
    .not('urls', 'is', null)

  if (error) {
    console.log(`❌ Failed to fetch projects: ${error.message}`)
    return
  }

  const projectsWithPDFs = projects?.filter(p => p.urls && p.urls.length > 0) || []
  console.log(`✅ Found ${projectsWithPDFs.length} projects with PDFs`)

  let successCount = 0
  let failCount = 0

  for (let i = 0; i < projectsWithPDFs.length; i++) {
    const project = projectsWithPDFs[i]
    console.log(`\n${'─'.repeat(80)}`)
    console.log(`[${i + 1}/${projectsWithPDFs.length}] ${project.name}`)
    console.log(`─'.repeat(80)}`)

    // Get first PDF URL
    const pdfUrl = project.urls[0]
    console.log(`   🔗 PDF: ${pdfUrl.substring(0, 60)}...`)

    // Extract PDF text
    const pdfData = await extractPDFText(pdfUrl)
    if (!pdfData) {
      console.log(`   ⚠️  Skipping - PDF extraction failed`)
      failCount++
      continue
    }

    // Find relevant sections
    const relevantText = findRelevantSections(pdfData.text)
    if (relevantText.length < 1000) {
      console.log(`   ⚠️  Skipping - insufficient relevant text found`)
      failCount++
      continue
    }

    // Extract data with AI
    const extractedData = await extractDataWithAI(relevantText, project.name)
    if (!extractedData) {
      console.log(`   ⚠️  Skipping - AI extraction failed`)
      failCount++
      continue
    }

    // Update database
    const success = await updateProject(project.id, extractedData)
    if (success) {
      successCount++
    } else {
      failCount++
    }

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 1000))
  }

  console.log('\n' + '='.repeat(80))
  console.log('📊 EXTRACTION COMPLETE')
  console.log('='.repeat(80))
  console.log(`✅ Successful: ${successCount}`)
  console.log(`❌ Failed: ${failCount}`)
  console.log(`📈 Success rate: ${((successCount / projectsWithPDFs.length) * 100).toFixed(1)}%`)
  console.log('\n✨ Data is now visible in the frontend!')
}

main().catch(console.error)
