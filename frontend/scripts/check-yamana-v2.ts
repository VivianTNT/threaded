#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('🔍 CHECKING YAMANA GOLD DATA')
  console.log('='.repeat(80))

  // Get ALL projects and filter for Yamana
  const { data: allProjects } = await supabase
    .from('projects')
    .select('*')
    .order('name')

  const yamanaProjects = allProjects?.filter(p =>
    (p.company_name && p.company_name.toLowerCase().includes('yamana')) ||
    (p.name && p.name.toLowerCase().includes('yamana'))
  ) || []

  console.log(`\nTotal projects in database: ${allProjects?.length || 0}`)
  console.log(`Yamana Gold projects found: ${yamanaProjects.length}`)

  if (yamanaProjects.length > 0) {
    console.log('\n📁 YAMANA GOLD PROJECTS:')

    for (const project of yamanaProjects) {
      console.log('\n' + '-'.repeat(80))
      console.log(`Project: ${project.name}`)
      console.log(`Company: ${project.company_name}`)
      console.log(`Company ID: ${project.company_id}`)
      console.log(`Stage: ${project.stage || 'N/A'}`)
      console.log(`Commodity: ${project.commodity || 'N/A'}`)
      console.log(`Country: ${project.country || 'N/A'}`)

      console.log(`\n💰 Financial Metrics:`)
      console.log(`  NPV: ${project.post_tax_npv_usd_m ? '$' + project.post_tax_npv_usd_m + 'M' : 'NULL'}`)
      console.log(`  IRR: ${project.irr_percent ? project.irr_percent + '%' : 'NULL'}`)
      console.log(`  CAPEX: ${project.capex_usd_m ? '$' + project.capex_usd_m + 'M' : 'NULL'}`)

      console.log(`\n📄 Documents (urls):`)
      if (project.urls && project.urls.length > 0) {
        for (let i = 0; i < project.urls.length; i++) {
          const url = project.urls[i]
          const fileName = url.substring(url.lastIndexOf('/') + 1)

          // Try to get file size
          try {
            const response = await fetch(url, { method: 'HEAD' })
            const sizeInBytes = response.headers.get('content-length')
            const sizeInMB = sizeInBytes ? (parseInt(sizeInBytes) / 1024 / 1024).toFixed(2) : '?'
            console.log(`  ${i + 1}. ${fileName} (${sizeInMB}MB)`)
          } catch {
            console.log(`  ${i + 1}. ${fileName}`)
          }
          console.log(`     ${url}`)
        }
      } else {
        console.log('  No documents in urls field')
      }

      if (project.document_urls && project.document_urls.length > 0) {
        console.log(`\n📎 Document URLs (legacy field):`)
        for (const url of project.document_urls) {
          console.log(`  - ${url}`)
        }
      }
    }
  } else {
    console.log('\n⚠️  No Yamana projects found in database')
    console.log('\nSearching for any project that might be Yamana...')

    // Show first 10 projects as sample
    const sample = allProjects?.slice(0, 10) || []
    console.log(`\nShowing first 10 projects as sample:`)
    for (const p of sample) {
      console.log(`  - ${p.name} (${p.company_name})`)
    }
  }

  console.log('\n' + '='.repeat(80))
}

main().catch(console.error)
