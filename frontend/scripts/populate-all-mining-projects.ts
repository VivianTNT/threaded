#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'
import OpenAI from 'openai'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!
})

interface ProjectData {
  name: string
  companyName: string
  location: string
  country: string
  commodities: string[]
  stage: string
  resourceTonnes?: number
  resourceGrade?: number
  reserveTonnes?: number
  reserveGrade?: number
  capex?: number
  opex?: number
  aisc?: number
  latitude?: number
  longitude?: number
}

async function downloadPDF(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to download PDF: ${response.statusText}`)
  }
  return await response.arrayBuffer()
}

async function extractProjectData(pdfUrl: string, filename: string): Promise<ProjectData | null> {
  try {
    console.log(`   📄 Downloading PDF...`)
    const pdfBuffer = await downloadPDF(pdfUrl)

    // Convert to base64 for OpenAI
    const base64Pdf = Buffer.from(pdfBuffer).toString('base64')

    console.log(`   🤖 Extracting data with OpenAI...`)

    const response = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `Analyze this mining technical report and extract the following information in JSON format:
{
  "name": "Project name",
  "companyName": "Operating/owning company name",
  "location": "State/province/region",
  "country": "Country",
  "commodities": ["List of commodities like Gold, Copper, etc"],
  "stage": "One of: Exploration, Pre-feasibility, Feasibility, Development, Production",
  "resourceTonnes": "Total mineral resources in tonnes (number only)",
  "resourceGrade": "Average resource grade (g/t or % as a number)",
  "reserveTonnes": "Total mineral reserves in tonnes (number only)",
  "reserveGrade": "Average reserve grade (g/t or % as a number)",
  "capex": "Capital expenditure in millions USD (number only)",
  "opex": "Operating cost per tonne in USD (number only)",
  "aisc": "All-in sustaining cost per ounce/tonne in USD (number only)",
  "latitude": "Latitude coordinate (number)",
  "longitude": "Longitude coordinate (number)"
}

Return ONLY valid JSON. If a field is not found, use null. Extract numbers without units.`
            },
            {
              type: 'image_url',
              image_url: {
                url: `data:application/pdf;base64,${base64Pdf}`
              }
            }
          ]
        }
      ],
      max_tokens: 1000,
      temperature: 0.1
    })

    const content = response.choices[0]?.message?.content
    if (!content) {
      throw new Error('No response from OpenAI')
    }

    // Extract JSON from response
    const jsonMatch = content.match(/\{[\s\S]*\}/)
    if (!jsonMatch) {
      throw new Error('No JSON found in response')
    }

    const data = JSON.parse(jsonMatch[0]) as ProjectData
    console.log(`   ✅ Extracted: ${data.name} - ${data.companyName}`)

    return data

  } catch (error) {
    console.error(`   ❌ Error extracting data:`, error)
    return null
  }
}

async function findOrCreateCompany(companyName: string, ticker?: string): Promise<string> {
  // Try to find existing company
  const { data: existing, error: findError } = await supabase
    .from('companies')
    .select('id')
    .ilike('name', companyName)
    .single()

  if (existing) {
    return existing.id
  }

  // Create new company
  const { data: newCompany, error: createError } = await supabase
    .from('companies')
    .insert({
      name: companyName,
      ticker: ticker || null,
      description: `Mining company operating ${companyName}`,
      created_at: new Date().toISOString()
    })
    .select('id')
    .single()

  if (createError) {
    throw new Error(`Failed to create company: ${createError.message}`)
  }

  return newCompany.id
}

async function main() {
  console.log('🚀 Starting comprehensive mining project population...\n')

  // First, clear existing projects
  console.log('🗑️  Clearing existing projects...')
  const { error: deleteError } = await supabase
    .from('projects')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000') // Delete all

  if (deleteError) {
    console.error('Error clearing projects:', deleteError.message)
  } else {
    console.log('✅ Cleared existing projects\n')
  }

  // Get all PDFs from technical-documents/mining-documents
  console.log('📦 Fetching PDFs from Supabase storage...')
  const { data: files, error: filesError } = await supabase.storage
    .from('technical-documents')
    .list('mining-documents', { limit: 1000 })

  if (filesError) {
    console.error('Error listing files:', filesError.message)
    return
  }

  console.log(`Found ${files.length} PDF files\n`)

  let successCount = 0
  let failCount = 0

  // Process each PDF
  for (let i = 0; i < files.length; i++) {
    const file = files[i]
    const pdfUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/technical-documents/mining-documents/${file.name}`

    console.log(`\n[${i + 1}/${files.length}] Processing: ${file.name}`)

    // Extract project data from PDF
    const projectData = await extractProjectData(pdfUrl, file.name)

    if (!projectData) {
      console.log(`   ⚠️  Skipping - extraction failed`)
      failCount++
      continue
    }

    try {
      // Find or create company
      const companyId = await findOrCreateCompany(projectData.companyName)
      console.log(`   🏢 Company ID: ${companyId}`)

      // Insert project
      const { data: newProject, error: insertError } = await supabase
        .from('projects')
        .insert({
          name: projectData.name,
          company_id: companyId,
          location: projectData.location,
          country: projectData.country,
          commodities: projectData.commodities,
          stage: projectData.stage,
          resource: projectData.resourceTonnes,
          resource_grade: projectData.resourceGrade,
          reserve: projectData.reserveTonnes,
          reserve_grade: projectData.reserveGrade,
          capex: projectData.capex,
          opex: projectData.opex,
          aisc: projectData.aisc,
          latitude: projectData.latitude,
          longitude: projectData.longitude,
          document_storage_path: pdfUrl,
          urls: [pdfUrl],
          created_at: new Date().toISOString()
        })
        .select('id')
        .single()

      if (insertError) {
        console.error(`   ❌ Failed to insert project:`, insertError.message)
        failCount++
      } else {
        console.log(`   ✨ Successfully created project`)
        successCount++
      }

    } catch (error) {
      console.error(`   ❌ Error processing project:`, error)
      failCount++
    }

    // Rate limiting - wait 2 seconds between API calls
    if (i < files.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 2000))
    }
  }

  console.log('\n\n📊 Summary:')
  console.log(`   Total PDFs processed: ${files.length}`)
  console.log(`   ✅ Successfully created: ${successCount}`)
  console.log(`   ❌ Failed: ${failCount}`)
  console.log('\n✅ Population complete!')
}

main()
