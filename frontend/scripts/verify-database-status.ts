#!/usr/bin/env npx tsx

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('📊 DATABASE STATUS REPORT\n')
  console.log('='.repeat(50))

  // Check companies count
  const { count: companyCount, error: companyError } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true })

  if (!companyError) {
    console.log(`\n✅ Companies: ${companyCount}`)

    // Get sample companies
    const { data: sampleCompanies } = await supabase
      .from('companies')
      .select('name, ticker')
      .limit(5)

    if (sampleCompanies && sampleCompanies.length > 0) {
      console.log('   Sample companies:')
      sampleCompanies.forEach(c => {
        console.log(`   - ${c.name} (${c.ticker})`)
      })
    }
  } else {
    console.log(`\n❌ Companies error: ${companyError.message}`)
  }

  // Check projects count
  const { count: projectCount, error: projectError } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  if (!projectError) {
    console.log(`\n✅ Projects: ${projectCount}`)

    // Get sample projects with company names
    const { data: sampleProjects } = await supabase
      .from('projects')
      .select(`
        name,
        location,
        stage,
        npv,
        irr,
        capex,
        companies!projects_company_id_fkey(name)
      `)
      .limit(5)

    if (sampleProjects && sampleProjects.length > 0) {
      console.log('   Sample projects:')
      sampleProjects.forEach(p => {
        const companyName = (p as any).companies?.name || 'Unknown'
        console.log(`   - ${p.name} (${companyName}) - ${p.location}`)
        if (p.npv || p.irr || p.capex) {
          console.log(`     Financial: NPV: $${p.npv ? (p.npv/1000000).toFixed(0) + 'M' : 'N/A'}, IRR: ${p.irr || 'N/A'}%, CAPEX: $${p.capex ? (p.capex/1000000).toFixed(0) + 'M' : 'N/A'}`)
        }
      })
    }
  } else {
    console.log(`\n❌ Projects error: ${projectError.message}`)
  }

  // Check PDF highlights count
  const { count: highlightCount, error: highlightError } = await supabase
    .from('pdf_highlights')
    .select('*', { count: 'exact', head: true })

  if (!highlightError) {
    console.log(`\n✅ PDF Highlights: ${highlightCount}`)
  } else {
    console.log(`\n❌ PDF Highlights error: ${highlightError.message}`)
  }

  console.log('\n' + '='.repeat(50))
  console.log('\n📈 SUMMARY:')
  console.log(`   Total Companies: ${companyCount || 0}`)
  console.log(`   Total Projects: ${projectCount || 0}`)
  console.log(`   Total PDF Highlights: ${highlightCount || 0}`)
  console.log('\n✨ Database population status verified!')
}

main().catch(console.error)