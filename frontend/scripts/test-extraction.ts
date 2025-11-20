#!/usr/bin/env npx tsx
/**
 * TEST EXTRACTION - Extract data from stored PDFs for 2 projects
 */

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

interface ExtractedData {
  npv?: number
  irr?: number
  capex?: number
}

async function extractDataFromStoredPDF(storedUrl: string): Promise<ExtractedData> {
  try {
    console.log(`      🔍 Extracting financial data...`)

    // Call the extraction API endpoint
    const apiUrl = `http://localhost:3000/api/pdf/extract-highlights`

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pdfUrl: storedUrl })
    })

    if (!response.ok) {
      console.log(`      ⚠️  API returned ${response.status}`)
      return {}
    }

    const result = await response.json()

    if (result.error) {
      console.log(`      ⚠️  ${result.error}`)
      return {}
    }

    // Extract financial metrics from the result
    const data: ExtractedData = {}

    if (result.npv || result.post_tax_npv_usd_m) {
      data.npv = result.npv || result.post_tax_npv_usd_m
    }
    if (result.irr || result.irr_percent) {
      data.irr = result.irr || result.irr_percent
    }
    if (result.capex || result.capex_usd_m) {
      data.capex = result.capex || result.capex_usd_m
    }

    if (data.npv || data.irr || data.capex) {
      console.log(`      💰 NPV: $${data.npv}M, IRR: ${data.irr}%, CAPEX: $${data.capex}M`)
    } else {
      console.log(`      ⚠️  No financial data found`)
    }

    return data
  } catch (error) {
    console.log(`      ❌ Extraction error:`, error)
    return {}
  }
}

async function main() {
  console.log('🧪 TEST EXTRACTION')
  console.log('='.repeat(80))

  // Wait for server to start
  console.log('Waiting 5 seconds for dev server to start...')
  await new Promise(r => setTimeout(r, 5000))

  // Get 2 projects
  const { data: projects, error } = await supabase
    .from('projects')
    .select('*')
    .limit(2)

  if (error || !projects || projects.length === 0) {
    console.log('❌ No projects found')
    return
  }

  console.log(`\n📊 Testing with ${projects.length} projects\n`)

  for (const project of projects) {
    console.log(`\n🏗️  ${project.name}`)

    if (!project.urls || project.urls.length === 0) {
      console.log(`   ⚠️  No URLs, skipping`)
      continue
    }

    console.log(`   Testing first PDF: ${project.urls[0]}`)

    const extractedData = await extractDataFromStoredPDF(project.urls[0])

    if (extractedData.npv || extractedData.irr || extractedData.capex) {
      console.log(`   ✅ Extraction successful!`)

      // Update project
      const updateData: any = {}
      if (extractedData.npv) updateData.post_tax_npv_usd_m = extractedData.npv
      if (extractedData.irr) updateData.irr_percent = extractedData.irr
      if (extractedData.capex) updateData.capex_usd_m = extractedData.capex

      const { error: updateError } = await supabase
        .from('projects')
        .update(updateData)
        .eq('id', project.id)

      if (updateError) {
        console.log(`   ❌ Update error:`, updateError.message)
      } else {
        console.log(`   ✅ Database updated`)
      }
    }

    await new Promise(r => setTimeout(r, 3000))
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ Test complete`)
}

main().catch(console.error)
