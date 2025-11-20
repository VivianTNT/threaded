#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkStorage() {
  console.log('=' .repeat(80))
  console.log('📦 SUPABASE STORAGE CHECK')
  console.log('=' .repeat(80))

  try {
    // List all buckets
    const { data: buckets, error: bucketsError } = await supabase.storage.listBuckets()

    if (bucketsError) {
      console.error('Error listing buckets:', bucketsError)
      return
    }

    console.log(`\n📂 Found ${buckets?.length || 0} storage bucket(s):\n`)

    if (!buckets || buckets.length === 0) {
      console.log('  No storage buckets found')
      return
    }

    let totalFiles = 0
    let totalSize = 0

    // Check each bucket
    for (const bucket of buckets) {
      console.log(`\n🗂️ Bucket: ${bucket.name}`)
      console.log(`  ID: ${bucket.id}`)
      console.log(`  Public: ${bucket.public}`)
      console.log(`  Created: ${bucket.created_at}`)

      // List files in bucket
      const { data: files, error: filesError } = await supabase.storage
        .from(bucket.name)
        .list('', {
          limit: 100,
          offset: 0
        })

      if (filesError) {
        console.log(`  ❌ Error listing files: ${filesError.message}`)
        continue
      }

      if (!files || files.length === 0) {
        console.log('  📄 No files in this bucket')
        continue
      }

      console.log(`\n  📄 Files (${files.length}):\n`)

      // Check subdirectories
      const folders = files.filter(f => !f.id)
      const directFiles = files.filter(f => f.id)

      // List files in subdirectories
      for (const folder of folders) {
        const { data: subFiles } = await supabase.storage
          .from(bucket.name)
          .list(folder.name, {
            limit: 100,
            offset: 0
          })

        if (subFiles && subFiles.length > 0) {
          console.log(`\n  📁 ${folder.name}/ (${subFiles.length} files):`)

          for (const file of subFiles.slice(0, 10)) { // Show first 10
            if (file.id) {
              const sizeMB = file.metadata?.size ? (file.metadata.size / 1024 / 1024).toFixed(2) : '?'
              console.log(`    - ${file.name} (${sizeMB} MB)`)
              totalFiles++
              totalSize += file.metadata?.size || 0
            }
          }

          if (subFiles.length > 10) {
            console.log(`    ... and ${subFiles.length - 10} more files`)
          }
        }
      }

      // List direct files
      if (directFiles.length > 0) {
        console.log(`\n  📄 Direct files:`)
        for (const file of directFiles.slice(0, 10)) {
          const sizeMB = file.metadata?.size ? (file.metadata.size / 1024 / 1024).toFixed(2) : '?'
          console.log(`    - ${file.name} (${sizeMB} MB)`)
          totalFiles++
          totalSize += file.metadata?.size || 0
        }
      }
    }

    console.log('\n' + '=' .repeat(80))
    console.log('📊 STORAGE SUMMARY')
    console.log('=' .repeat(80))
    console.log(`Total buckets: ${buckets.length}`)
    console.log(`Total files: ${totalFiles}`)
    console.log(`Total size: ${(totalSize / 1024 / 1024).toFixed(2)} MB`)

  } catch (error) {
    console.error('Error:', error)
  }
}

// Also check database for projects with document links
async function checkProjectDocuments() {
  console.log('\n' + '=' .repeat(80))
  console.log('📄 DATABASE DOCUMENT CHECK')
  console.log('=' .repeat(80))

  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, document_storage_path, urls')
    .not('document_storage_path', 'is', null)
    .limit(20)

  if (error) {
    console.error('Error checking projects:', error)
    return
  }

  console.log(`\n📊 Projects with document paths: ${projects?.length || 0}\n`)

  if (projects && projects.length > 0) {
    for (const project of projects.slice(0, 10)) {
      console.log(`📍 ${project.name}`)
      console.log(`   Storage: ${project.document_storage_path || 'None'}`)
      console.log(`   URLs: ${project.urls?.length || 0} link(s)`)
      console.log()
    }
  } else {
    console.log('No projects with document storage paths found')
  }
}

async function main() {
  await checkStorage()
  await checkProjectDocuments()
}

main().catch(console.error)