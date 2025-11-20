#!/usr/bin/env npx tsx
/**
 * COMPREHENSIVE EXTRACTION PIPELINE
 *
 * 1. Get all projects from database
 * 2. For each project, download PDFs from FactSet
 * 3. Upload PDFs to Supabase Storage
 * 4. Extract financial data from PDFs using API
 * 5. Update project with extracted NPV, IRR, CAPEX
 */

import { createClient } from '@supabase/supabase-js'
import pdf from 'pdf-parse'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const FACTSET_USERNAME = process.env.FACTSET_USERNAME!
const FACTSET_API_KEY = process.env.FACTSET_API_KEY!
const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64')

interface ExtractedData {
  npv?: number
  irr?: number
  capex?: number
}

async function extractTextFromPDF(pdfBuffer: Buffer): Promise<string> {
  try {
    const data = await pdf(pdfBuffer)
    return data.text
  } catch (error) {
    console.error('PDF parsing error:', error)
    return ''
  }
}

async function extractFinancialDataWithAI(text: string): Promise<ExtractedData> {
  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: 'Extract NPV (Net Present Value in USD millions), IRR (Internal Rate of Return as percentage), and CAPEX (Capital Expenditure in USD millions) from mining technical reports. Return ONLY valid JSON with numeric values.'
          },
          {
            role: 'user',
            content: `Extract financial metrics from this text (first 8000 chars):\n\n${text.substring(0, 8000)}`
          }
        ],
        response_format: { type: 'json_object' },
        temperature: 0.1
      })
    })

    if (!response.ok) {
      console.error('OpenAI API error:', response.status)
      return {}
    }

    const result = await response.json()
    const extracted = JSON.parse(result.choices[0].message.content)

    return {
      npv: extracted.npv || extracted.NPV,
      irr: extracted.irr || extracted.IRR,
      capex: extracted.capex || extracted.CAPEX || extracted.capex_usd_m
    }
  } catch (error) {
    console.error('AI extraction error:', error)
    return {}
  }
}

async function downloadAndStorePDF(pdfUrl: string, projectId: string, docIndex: number): Promise<string | null> {
  try {
    console.log(`      📥 Downloading PDF ${docIndex + 1}...`)

    const response = await fetch(pdfUrl, {
      headers: { 'Authorization': authHeader }
    })

    if (!response.ok) {
      console.log(`      ❌ Download failed: ${response.status}`)
      return null
    }

    const arrayBuffer = await response.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)
    const sizeInMB = (buffer.length / 1024 / 1024).toFixed(2)
    console.log(`      📦 Downloaded ${sizeInMB}MB`)

    // Upload to Supabase Storage
    const fileName = `${projectId}/doc-${docIndex + 1}.pdf`
    const { error: uploadError } = await supabase.storage
      .from('mining-documents')
      .upload(fileName, buffer, {
        contentType: 'application/pdf',
        upsert: true
      })

    if (uploadError) {
      console.log(`      ❌ Upload error:`, uploadError.message)
      return null
    }

    const { data: { publicUrl } } = supabase.storage
      .from('mining-documents')
      .getPublicUrl(fileName)

    console.log(`      ✅ Stored in bucket`)
    return publicUrl
  } catch (error) {
    console.log(`      ❌ Error:`, error)
    return null
  }
}

async function extractDataFromStoredPDF(storedUrl: string): Promise<ExtractedData> {
  try {
    console.log(`      🔍 Extracting financial data from stored PDF...`)

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

async function mergeFinancialData(allData: ExtractedData[]): ExtractedData {
  const merged: ExtractedData = {}

  // Take the most recent/highest NPV
  const npvValues = allData.filter(d => d.npv && !isNaN(d.npv)).map(d => d.npv!)
  if (npvValues.length > 0) {
    merged.npv = Math.max(...npvValues)
  }

  // Take the most recent/highest IRR
  const irrValues = allData.filter(d => d.irr && !isNaN(d.irr)).map(d => d.irr!)
  if (irrValues.length > 0) {
    merged.irr = Math.max(...irrValues)
  }

  // Take the most recent/highest CAPEX
  const capexValues = allData.filter(d => d.capex && !isNaN(d.capex)).map(d => d.capex!)
  if (capexValues.length > 0) {
    merged.capex = Math.max(...capexValues)
  }

  return merged
}

async function main() {
  console.log('🏭 COMPREHENSIVE EXTRACTION PIPELINE')
  console.log('='.repeat(80))
  console.log('Downloading PDFs → Storing in bucket → Extracting data → Updating DB')
  console.log('='.repeat(80))

  // Get all projects
  const { data: projects, error } = await supabase
    .from('projects')
    .select('*')
    .order('created_at', { ascending: false })

  if (error || !projects || projects.length === 0) {
    console.log('❌ No projects found')
    return
  }

  console.log(`\n📊 Found ${projects.length} projects\n`)

  let processed = 0
  let updated = 0

  for (const project of projects) {
    console.log(`\n🏗️  ${project.name}`)

    if (!project.urls || project.urls.length === 0) {
      console.log(`   ⚠️  No URLs found, skipping`)
      continue
    }

    const storedUrls: string[] = []
    const extractedDataArray: ExtractedData[] = []

    // Process each PDF
    for (let i = 0; i < Math.min(project.urls.length, 3); i++) {
      const pdfUrl = project.urls[i]

      console.log(`   📄 Document ${i + 1}/${Math.min(project.urls.length, 3)}`)

      // Download and store in bucket
      const storedUrl = await downloadAndStorePDF(pdfUrl, project.id, i)
      if (storedUrl) {
        storedUrls.push(storedUrl)

        // Extract financial data from the stored PDF
        const extractedData = await extractDataFromStoredPDF(storedUrl)
        extractedDataArray.push(extractedData)
      }

      // Rate limiting
      await new Promise(r => setTimeout(r, 3000))
    }

    // Merge all extracted data
    const mergedData = mergeFinancialData(extractedDataArray)

    // Update project with stored URLs and financial data
    const updateData: any = {}

    if (storedUrls.length > 0) {
      updateData.urls = storedUrls
    }

    if (mergedData.npv) updateData.post_tax_npv_usd_m = mergedData.npv
    if (mergedData.irr) updateData.irr_percent = mergedData.irr
    if (mergedData.capex) updateData.capex_usd_m = mergedData.capex

    if (Object.keys(updateData).length > 0) {
      const { error: updateError } = await supabase
        .from('projects')
        .update(updateData)
        .eq('id', project.id)

      if (updateError) {
        console.log(`   ❌ Update error:`, updateError.message)
      } else {
        console.log(`   ✅ Updated with ${Object.keys(updateData).length} fields`)
        updated++
      }
    }

    processed++
    console.log(`   Progress: ${processed}/${projects.length}`)

    await new Promise(r => setTimeout(r, 3000))
  }

  console.log(`\n${'='.repeat(80)}`)
  console.log(`✅ Processed ${processed} projects`)
  console.log(`✅ Updated ${updated} projects with financial data`)
  console.log(`${'='.repeat(80)}`)
}

main().catch(console.error)
