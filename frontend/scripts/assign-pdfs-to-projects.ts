#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function assignPdfsToProjects() {
  console.log('🚀 Starting PDF assignment to projects...\n')

  try {
    // Step 1: Fetch all PDF files from mining-documents bucket
    console.log('📁 Fetching PDF files from storage...')
    const { data: files, error: storageError } = await supabase.storage
      .from('mining-documents')
      .list('', { limit: 1000 })

    if (storageError) {
      throw new Error(`Storage error: ${storageError.message}`)
    }

    if (!files || files.length === 0) {
      throw new Error('No PDF files found in mining-documents bucket')
    }

    console.log(`✅ Found ${files.length} PDF files\n`)

    // Step 2: Fetch all projects
    console.log('📊 Fetching projects from database...')
    const { data: projects, error: projectsError } = await supabase
      .from('projects')
      .select('id, name')
      .order('created_at')

    if (projectsError) {
      throw new Error(`Projects fetch error: ${projectsError.message}`)
    }

    if (!projects || projects.length === 0) {
      throw new Error('No projects found in database')
    }

    console.log(`✅ Found ${projects.length} projects\n`)

    // Step 3: Assign PDFs to projects (rotating through available PDFs)
    console.log('🔄 Assigning PDFs to projects...\n')

    const baseUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/mining-documents`
    let successCount = 0
    let errorCount = 0

    for (let i = 0; i < projects.length; i++) {
      const project = projects[i]
      const pdfFile = files[i % files.length] // Rotate through available PDFs
      const documentPath = `${baseUrl}/${pdfFile.name}`

      const { error: updateError } = await supabase
        .from('projects')
        .update({
          document_storage_path: documentPath,
          urls: [documentPath] // Also update urls array for consistency
        })
        .eq('id', project.id)

      if (updateError) {
        console.error(`❌ Error updating project ${project.name}: ${updateError.message}`)
        errorCount++
      } else {
        successCount++
        if ((i + 1) % 50 === 0) {
          console.log(`   ✅ Assigned PDFs to ${i + 1}/${projects.length} projects...`)
        }
      }
    }

    console.log('\n' + '='.repeat(60))
    console.log('✅ PDF assignment complete!')
    console.log(`   Successfully updated: ${successCount} projects`)
    console.log(`   Errors: ${errorCount}`)
    console.log('='.repeat(60))

    // Step 4: Verify assignments
    console.log('\n🔍 Verifying assignments...')
    const { data: verifyData, error: verifyError } = await supabase
      .from('projects')
      .select('id, name, document_storage_path')
      .not('document_storage_path', 'is', null)

    if (verifyError) {
      console.error('Verification error:', verifyError.message)
    } else {
      console.log(`✅ ${verifyData?.length || 0} projects have assigned PDFs\n`)

      // Show sample assignments
      console.log('📋 Sample assignments:')
      verifyData?.slice(0, 5).forEach((p, idx) => {
        console.log(`   ${idx + 1}. ${p.name}`)
        console.log(`      → ${p.document_storage_path}\n`)
      })
    }

  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message)
    process.exit(1)
  }
}

assignPdfsToProjects()
