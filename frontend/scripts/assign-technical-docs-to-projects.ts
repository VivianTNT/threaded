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
  console.log('🔗 Assigning technical documents to projects...\n')

  // Get all files from technical-documents/mining-documents
  const { data: files, error: filesError } = await supabase.storage
    .from('technical-documents')
    .list('mining-documents', { limit: 1000 })

  if (filesError) {
    console.error('Error listing files:', filesError.message)
    return
  }

  console.log(`Found ${files.length} file(s) in technical-documents/mining-documents\n`)

  // Get all projects
  const { data: projects, error: projectsError } = await supabase
    .from('projects')
    .select('id, name, document_storage_path, urls')
    .order('created_at', { ascending: true })
    .limit(1000)

  if (projectsError) {
    console.error('Error fetching projects:', projectsError.message)
    return
  }

  console.log(`Found ${projects.length} project(s) in database\n`)

  // Assign PDFs to projects (one PDF per project, in order)
  let updatedCount = 0

  for (let i = 0; i < Math.min(projects.length, files.length); i++) {
    const project = projects[i]
    const file = files[i]

    const storageUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/technical-documents/mining-documents/${file.name}`

    console.log(`✅ Assigning PDF to: ${project.name}`)
    console.log(`   PDF: ${file.name}`)
    console.log(`   URL: ${storageUrl}`)

    // Update the project
    const { error: updateError } = await supabase
      .from('projects')
      .update({
        document_storage_path: storageUrl,
        urls: [storageUrl]
      })
      .eq('id', project.id)

    if (updateError) {
      console.error(`   ❌ Update failed:`, updateError.message)
    } else {
      console.log(`   ✨ Updated successfully`)
      updatedCount++
    }
    console.log('')
  }

  console.log('\n📊 Summary:')
  console.log(`   Total PDFs available: ${files.length}`)
  console.log(`   Total projects in database: ${projects.length}`)
  console.log(`   Projects updated: ${updatedCount}`)

  if (files.length > projects.length) {
    console.log(`\n⚠️  ${files.length - projects.length} extra PDF(s) - no matching projects`)
  } else if (projects.length > files.length) {
    console.log(`\n⚠️  ${projects.length - files.length} project(s) without PDFs`)
  }

  console.log('\n✅ All projects now have Supabase storage URLs!')
}

main()
