#!/usr/bin/env npx tsx
/**
 * OpenAI Direct PDF Financial Extraction
 * Uses OpenAI to analyze PDF URLs directly and extract NPV, IRR, CAPEX
 */

import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

async function extractFinancialMetrics(pdfUrl: string, projectName: string) {
  try {
    console.log(`      🤖 Analyzing PDF with OpenAI...`)

    // Download first few pages of PDF as text
    const pdfResponse = await fetch(pdfUrl)
    const arrayBuffer = await pdfResponse.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    // Convert buffer to base64
    const base64 = buffer.toString('base64')

    // Use OpenAI to analyze the PDF
    // Since we can't use vision API for PDFs, we'll need to extract text first
    // Let's use a simple regex-based extraction on the raw bytes

    const text = buffer.toString('utf8', 0, Math.min(buffer.length, 100000)) // First 100KB

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: `You are a mining financial analyst. Extract NPV (Net Present Value in USD millions), IRR (Internal Rate of Return as percentage), and CAPEX (Capital Expenditure in USD millions) from this NI 43-101 technical report for the ${projectName} project. Look for terms like "Post-Tax NPV", "After-Tax NPV @5%", "IRR", "Internal Rate of Return", "Initial CAPEX", "Total CAPEX", "Capital Cost". Return ONLY valid JSON with numeric values: {"npv": number, "irr": number, "capex": number} or {} if not found.`
        },
        {
          role: 'user',
          content: `Analyze this text from a mining technical report and extract financial metrics:\n\n${text.substring(0, 15000)}`
        }
      ],
      response_format: { type: 'json_object' },
      temperature: 0.1
    })

    const result = JSON.parse(response.choices[0].message.content || '{}')

    return {
      npv: result.npv || result.NPV || null,
      irr: result.irr || result.IRR || null,
      capex: result.capex || result.CAPEX || null
    }
  } catch (error) {
    console.log(`      ⚠️  Extraction failed: ${error}`)
    return { npv: null, irr: null, capex: null }
  }
}

async function main() {
  console.log('💰 OPENAI DIRECT PDF FINANCIAL EXTRACTION')
  console.log('='.repeat(80))

  // Get sample of projects with PDF URLs to test
  const { data: projects } = await supabase
    .from('projects')
    .select('*')
    .not('urls', 'is', null)
    .limit(5)  // Start with just 5 to test
    .order('name')

  if (!projects || projects.length === 0) {
    console.log('No projects found')
    return
  }

  console.log(`Testing with ${projects.length} projects\n`)

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
      const metrics = await extractFinancialMetrics(pdfUrl, project.name)

      // Update database
      const updates: any = {}
      if (metrics.npv) {
        updates.post_tax_npv_usd_m = metrics.npv
        console.log(`      💰 NPV: $${metrics.npv}M`)
      }
      if (metrics.irr) {
        updates.irr_percent = metrics.irr
        console.log(`      📈 IRR: ${metrics.irr}%`)
      }
      if (metrics.capex) {
        updates.capex_usd_m = metrics.capex
        console.log(`      🏗️  CAPEX: $${metrics.capex}M`)
      }

      if (Object.keys(updates).length > 0) {
        await supabase
          .from('projects')
          .update(updates)
          .eq('id', project.id)

        console.log(`      ✅ Updated database`)
        extracted++
      } else {
        console.log(`      ⚠️  No metrics found`)
        failed++
      }

    } catch (error) {
      console.log(`      ❌ Error: ${error}`)
      failed++
    }

    // Rate limit
    await new Promise(r => setTimeout(r, 3000))
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ Extracted: ${extracted}`)
  console.log(`❌ Failed: ${failed}`)
  console.log(`📊 Total: ${projects.length}`)
  console.log(`${'='.repeat(80)}`)

  if (extracted > 0) {
    console.log(`\n✨ Success! Run this script again without the .limit(5) to process all 39 projects`)
  }
}

main().catch(console.error)
