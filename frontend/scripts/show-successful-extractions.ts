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
  console.log('📊 SHOWING SUCCESSFULLY EXTRACTED FACTSET PROJECTS')
  console.log('='.repeat(80))

  // Get all projects with document_storage_path (those from FactSet)
  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, company_id, location, commodities, document_storage_path, urls, created_at')
    .not('document_storage_path', 'is', null)
    .order('created_at', { ascending: false })

  if (error) {
    console.error('❌ Error:', error.message)
    return
  }

  if (!projects || projects.length === 0) {
    console.log('⚠️  No projects with FactSet PDFs found yet.')
    return
  }

  console.log(`\n✅ Found ${projects.length} projects with uploaded FactSet PDFs\n`)

  for (let i = 0; i < projects.length; i++) {
    const project = projects[i]
    console.log(`${i + 1}. ${project.name}`)
    console.log(`   📍 Location: ${project.location || 'N/A'}`)
    console.log(`   ⛏️  Commodities: ${project.commodities?.join(', ') || 'N/A'}`)
    console.log(`   📄 Supabase PDF URL:`)
    console.log(`      ${project.document_storage_path}`)
    console.log(`   🔗 Additional URLs: ${project.urls?.length || 0}`)
    if (project.urls && project.urls.length > 0) {
      project.urls.forEach((url: string, idx: number) => {
        console.log(`      ${idx + 1}. ${url}`)
      })
    }
    console.log(`   📅 Created: ${new Date(project.created_at).toLocaleDateString()}`)
    console.log()
  }

  console.log('='.repeat(80))
  console.log('📈 Summary:')
  console.log(`   Total projects with FactSet PDFs: ${projects.length}`)
  console.log('='.repeat(80))
}

main()
