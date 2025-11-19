#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('🔍 CHECKING YAMANA GOLD DATA')
  console.log('='.repeat(80))

  // Find Yamana company
  const { data: companies } = await supabase
    .from('companies')
    .select('*')
    .ilike('name', '%yamana%')

  console.log('\n📊 COMPANIES:')
  if (companies && companies.length > 0) {
    for (const company of companies) {
      console.log(`\nID: ${company.id}`)
      console.log(`Name: ${company.name}`)
      console.log(`Ticker: ${company.ticker}`)
      console.log(`Description: ${company.description?.substring(0, 100)}...`)
    }
  } else {
    console.log('No Yamana companies found')
  }

  // Find all projects for Yamana
  const { data: projects } = await supabase
    .from('projects')
    .select('*')
    .ilike('company_name', '%yamana%')

  console.log('\n\n📁 PROJECTS:')
  if (projects && projects.length > 0) {
    for (const project of projects) {
      console.log('\n' + '-'.repeat(80))
      console.log(`Project: ${project.name}`)
      console.log(`Company: ${project.company_name}`)
      console.log(`Stage: ${project.stage || 'N/A'}`)
      console.log(`Commodity: ${project.commodity || 'N/A'}`)
      console.log(`Country: ${project.country || 'N/A'}`)
      console.log(`\n💰 Financial Metrics:`)
      console.log(`  NPV: ${project.post_tax_npv_usd_m ? '$' + project.post_tax_npv_usd_m + 'M' : 'NULL'}`)
      console.log(`  IRR: ${project.irr_percent ? project.irr_percent + '%' : 'NULL'}`)
      console.log(`  CAPEX: ${project.capex_usd_m ? '$' + project.capex_usd_m + 'M' : 'NULL'}`)

      console.log(`\n📄 Documents:`)
      if (project.urls && project.urls.length > 0) {
        for (const url of project.urls) {
          const fileName = url.substring(url.lastIndexOf('/') + 1)
          console.log(`  - ${fileName}`)
          console.log(`    ${url}`)
        }
      } else {
        console.log('  No documents')
      }

      if (project.document_urls && project.document_urls.length > 0) {
        console.log(`\n📎 Document URLs (legacy):`)
        for (const url of project.document_urls) {
          console.log(`  - ${url}`)
        }
      }
    }
  } else {
    console.log('No Yamana projects found')
  }

  console.log('\n' + '='.repeat(80))
}

main().catch(console.error)
