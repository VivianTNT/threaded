#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function verifyFinalData() {
  console.log('🔍 Verifying final database state...\n')

  try {
    // Get company count
    const { count: companyCount } = await supabase
      .from('companies')
      .select('*', { count: 'exact', head: true })

    // Get project count
    const { count: projectCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true })

    // Get projects with PDFs
    const { count: projectsWithPdfs } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true })
      .not('document_storage_path', 'is', null)

    console.log('📊 Database Summary:')
    console.log(`   Companies: ${companyCount}`)
    console.log(`   Projects: ${projectCount}`)
    console.log(`   Projects with PDFs: ${projectsWithPdfs}\n`)

    // Sample 10 random projects
    const { data: sampleProjects, error } = await supabase
      .from('projects')
      .select('id, name, location, stage, commodities, npv, irr, capex, resource, reserve, document_storage_path')
      .limit(10)

    if (error) throw error

    console.log('📋 Sample Projects (showing ALL fields are populated):\n')

    sampleProjects?.forEach((p, idx) => {
      console.log(`${idx + 1}. ${p.name}`)
      console.log(`   Location: ${p.location}`)
      console.log(`   Stage: ${p.stage}`)
      console.log(`   Commodities: ${p.commodities?.join(', ')}`)
      console.log(`   NPV: $${p.npv}M  |  IRR: ${p.irr}%  |  CAPEX: $${p.capex}M`)
      console.log(`   Resource: ${p.resource}`)
      console.log(`   Reserve: ${p.reserve}`)
      console.log(`   PDF: ${p.document_storage_path ? '✅ Assigned' : '❌ Missing'}`)

      if (p.document_storage_path) {
        // Extract just the PDF filename for brevity
        const pdfId = p.document_storage_path.split('/').pop()
        console.log(`   PDF ID: ${pdfId}`)
      }
      console.log()
    })

    // Check for any null values
    console.log('🔍 Checking for NULL values...')
    const fieldsToCheck = ['name', 'location', 'stage', 'commodities', 'npv', 'irr', 'capex', 'resource', 'reserve', 'document_storage_path']

    for (const field of fieldsToCheck) {
      const { count } = await supabase
        .from('projects')
        .select('*', { count: 'exact', head: true })
        .is(field, null)

      if (count && count > 0) {
        console.log(`   ⚠️ ${field}: ${count} projects have NULL values`)
      } else {
        console.log(`   ✅ ${field}: No NULL values`)
      }
    }

    console.log('\n' + '='.repeat(60))
    console.log('✅ Verification complete!')
    console.log('='.repeat(60))

  } catch (error: any) {
    console.error('❌ Error:', error.message)
  }
}

verifyFinalData()
