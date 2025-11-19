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
  console.log('🔗 Assigning PDF files to projects...\n')

  // Get all files from mining-documents bucket
  const { data: files, error: filesError } = await supabase.storage
    .from('mining-documents')
    .list('', { limit: 1000 })

  if (filesError) {
    console.error('Error listing files:', filesError.message)
    return
  }

  // Filter to get only UUID-like files (not folders)
  const pdfFiles = files.filter(file =>
    file.name !== 'factset-documents' &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(file.name)
  )

  console.log(`Found ${pdfFiles.length} PDF file(s) in mining-documents bucket\n`)

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

  for (let i = 0; i < Math.min(projects.length, pdfFiles.length); i++) {
    const project = projects[i]
    const file = pdfFiles[i]

    const storageUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/mining-documents/${file.name}`

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
  console.log(`   Total PDFs available: ${pdfFiles.length}`)
  console.log(`   Total projects in database: ${projects.length}`)
  console.log(`   Projects updated: ${updatedCount}`)

  if (pdfFiles.length > projects.length) {
    console.log(`\n⚠️  ${pdfFiles.length - projects.length} extra PDF(s) - no matching projects`)
  } else if (projects.length > pdfFiles.length) {
    console.log(`\n⚠️  ${projects.length - pdfFiles.length} project(s) without PDFs`)
  }
}

main()
