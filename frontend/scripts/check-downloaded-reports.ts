#!/usr/bin/env node

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkReports() {
  console.log('📊 Checking downloaded FactSet technical reports...\n')

  // List all files in factset-documents bucket
  const folders = ['IVN', 'LUN', 'FM', 'TECK']

  for (const folder of folders) {
    console.log(`\n📁 ${folder}/`)
    console.log('='.repeat(80))

    const { data: files, error } = await supabase.storage
      .from('factset-documents')
      .list(folder, { limit: 100 })

    if (error) {
      console.log(`   ❌ Error: ${error.message}`)
      continue
    }

    if (!files || files.length === 0) {
      console.log('   (empty)')
      continue
    }

    files.forEach((file, idx) => {
      const sizeKB = Math.round((file.metadata?.size || 0) / 1024)
      const sizeMB = (sizeKB / 1024).toFixed(2)
      const { data: urlData } = supabase.storage
        .from('factset-documents')
        .getPublicUrl(`${folder}/${file.name}`)

      console.log(`\n   ${idx + 1}. ${file.name}`)
      console.log(`      Size: ${sizeKB} KB (${sizeMB} MB)`)
      console.log(`      URL: ${urlData.publicUrl}`)
    })
  }

  // Check if any projects have these documents linked
  console.log('\n\n📋 Checking which projects have these documents...\n')
  console.log('='.repeat(80))

  const { data: projects, error: projectError } = await supabase
    .from('projects')
    .select('id, name, document_urls')
    .not('document_urls', 'is', null)

  if (projectError) {
    console.log(`❌ Error fetching projects: ${projectError.message}`)
    return
  }

  const factsetProjects = projects?.filter(p =>
    p.document_urls?.some((url: string) => url.includes('factset-documents'))
  )

  if (!factsetProjects || factsetProjects.length === 0) {
    console.log('\n⚠️  No projects currently have FactSet documents linked!\n')
    console.log('These large technical reports are stored in Supabase but not')
    console.log('linked to any project in the database.\n')
    console.log('To view them, we need to either:')
    console.log('  1. Link them to existing projects')
    console.log('  2. Create new projects for these reports')
  } else {
    console.log(`\n✅ Found ${factsetProjects.length} projects with FactSet documents:\n`)
    factsetProjects.forEach((project, idx) => {
      console.log(`${idx + 1}. ${project.name}`)
      console.log(`   ID: ${project.id}`)
      console.log(`   Documents: ${project.document_urls?.length || 0}`)
      project.document_urls?.forEach((url: string, i: number) => {
        if (url.includes('factset-documents')) {
          console.log(`      ${i + 1}. ${url}`)
        }
      })
      console.log()
    })
  }
}

checkReports().catch(console.error)
