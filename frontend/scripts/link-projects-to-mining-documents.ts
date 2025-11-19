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
  console.log('🔗 Linking projects to mining-documents storage...\n')

  // Get all files from mining-documents bucket (these are UUID-named files)
  const { data: files, error: filesError } = await supabase.storage
    .from('mining-documents')
    .list('', { limit: 1000 })

  if (filesError) {
    console.error('Error listing files:', filesError.message)
    return
  }

  console.log(`Found ${files.length} file(s) in mining-documents bucket\n`)

  // Filter out folders and only keep actual files (UUIDs)
  const pdfFiles = files.filter(file =>
    file.id &&
    file.name !== 'factset-documents' &&
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(file.name)
  )

  console.log(`Found ${pdfFiles.length} UUID-named PDF file(s)\n`)

  // Get all projects
  const { data: projects, error: projectsError } = await supabase
    .from('projects')
    .select('id, name, document_storage_path, urls')
    .limit(1000)

  if (projectsError) {
    console.error('Error fetching projects:', projectsError.message)
    return
  }

  console.log(`Found ${projects.length} project(s) in database\n`)

  // Try to match projects to files by their ID
  let updatedCount = 0
  let matchedCount = 0

  for (const project of projects) {
    // Check if there's a file with this project's ID
    const matchingFile = pdfFiles.find(file => file.name === project.id)

    if (matchingFile) {
      matchedCount++
      const storageUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/mining-documents/${matchingFile.name}`

      console.log(`✅ Found PDF for: ${project.name}`)
      console.log(`   Project ID: ${project.id}`)
      console.log(`   Storage URL: ${storageUrl}`)

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
  }

  console.log('\n📊 Summary:')
  console.log(`   Total files in storage: ${pdfFiles.length}`)
  console.log(`   Total projects in database: ${projects.length}`)
  console.log(`   Projects with matching PDFs: ${matchedCount}`)
  console.log(`   Projects updated successfully: ${updatedCount}`)

  if (matchedCount < pdfFiles.length) {
    console.log(`\n⚠️  ${pdfFiles.length - matchedCount} PDF file(s) have no matching project`)
  }

  if (updatedCount === 0 && pdfFiles.length > 0) {
    console.log('\n💡 Tip: PDF files are named with UUIDs that should match project IDs')
    console.log('   If no matches were found, the PDFs might be from different projects')
  }
}

main()
