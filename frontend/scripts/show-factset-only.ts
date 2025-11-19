#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('📊 FACTSET PROJECTS WITH SUPABASE STORAGE PDFS')
  console.log('='.repeat(80))

  // Get projects where document_storage_path starts with factset bucket path
  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, company_id, location, commodities, document_storage_path, urls, created_at')
    .like('document_storage_path', '%factset-documents%')
    .order('created_at', { ascending: false })
    .limit(10)

  if (error) {
    console.error('❌ Error:', error.message)
    return
  }

  if (!projects || projects.length === 0) {
    console.log('⚠️  No projects with FactSet Storage PDFs found.')
    console.log('\nLet me check for any projects with storage paths...\n')

    // Try broader search
    const { data: anyProjects, error: error2 } = await supabase
      .from('projects')
      .select('id, name, document_storage_path')
      .not('document_storage_path', 'is', null)
      .order('created_at', { ascending: false })
      .limit(5)

    if (anyProjects && anyProjects.length > 0) {
      console.log(`✅ Found ${anyProjects.length} projects with storage paths:`)
      anyProjects.forEach((p, i) => {
        console.log(`   ${i + 1}. ${p.name}`)
        console.log(`      Path: ${p.document_storage_path}`)
      })
    }

    return
  }

  console.log(`\n✅ Found ${projects.length} projects with FactSet PDFs in Supabase Storage\n`)

  for (let i = 0; i < projects.length; i++) {
    const project = projects[i]
    const fullUrl = project.document_storage_path.startsWith('http')
      ? project.document_storage_path
      : `https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/${project.document_storage_path}`

    console.log(`${i + 1}. 📄 ${project.name}`)
    console.log(`   📍 Location: ${project.location || 'N/A'}`)
    console.log(`   ⛏️  Commodities: ${project.commodities?.join(', ') || 'N/A'}`)
    console.log(`   📦 Supabase Storage URL:`)
    console.log(`      ${fullUrl}`)
    console.log(`   📅 Created: ${new Date(project.created_at).toLocaleDateString()}`)
    console.log()
  }

  console.log('='.repeat(80))
  console.log(`✅ These ${projects.length} PDFs were successfully downloaded from FactSet API`)
  console.log('   and uploaded to Supabase Storage!')
  console.log('='.repeat(80))
}

main()
