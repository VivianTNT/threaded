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
  console.log('🔍 Listing all storage buckets and files...\n')

  // List all buckets
  const { data: buckets, error: bucketsError } = await supabase.storage.listBuckets()

  if (bucketsError) {
    console.error('Error listing buckets:', bucketsError)
    return
  }

  console.log(`Found ${buckets.length} bucket(s):\n`)

  const allFiles: Array<{
    bucket: string
    path: string
    fullUrl: string
    name: string
    size: number
  }> = []

  // List files in each bucket
  for (const bucket of buckets) {
    console.log(`📦 Bucket: ${bucket.name} (${bucket.public ? 'Public' : 'Private'})`)

    try {
      // List root files
      const { data: rootFiles, error: rootError } = await supabase.storage
        .from(bucket.name)
        .list('', { limit: 100 })

      if (rootError) {
        console.error(`   Error listing files in ${bucket.name}:`, rootError.message)
        continue
      }

      if (rootFiles && rootFiles.length > 0) {
        for (const file of rootFiles) {
          if (!file.id) {
            // It's a folder, list its contents
            const { data: folderFiles, error: folderError } = await supabase.storage
              .from(bucket.name)
              .list(file.name, { limit: 100 })

            if (!folderError && folderFiles) {
              for (const subFile of folderFiles) {
                if (subFile.id) {
                  const filePath = `${file.name}/${subFile.name}`
                  const fullUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/${bucket.name}/${filePath}`

                  allFiles.push({
                    bucket: bucket.name,
                    path: filePath,
                    fullUrl,
                    name: subFile.name,
                    size: subFile.metadata?.size || 0
                  })

                  console.log(`   📄 ${filePath} (${(subFile.metadata?.size / 1024 / 1024).toFixed(2)} MB)`)
                  console.log(`      ${fullUrl}`)
                }
              }
            }
          } else {
            // It's a file
            const fullUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/${bucket.name}/${file.name}`

            allFiles.push({
              bucket: bucket.name,
              path: file.name,
              fullUrl,
              name: file.name,
              size: file.metadata?.size || 0
            })

            console.log(`   📄 ${file.name} (${(file.metadata?.size / 1024 / 1024).toFixed(2)} MB)`)
            console.log(`      ${fullUrl}`)
          }
        }
      }
    } catch (err) {
      console.error(`   Error processing bucket ${bucket.name}:`, err)
    }

    console.log('')
  }

  console.log(`\n📊 Total files found: ${allFiles.length}\n`)

  if (allFiles.length === 0) {
    console.log('⚠️  No files found in storage. Upload some PDFs first.')
    return
  }

  // Now get all projects that need updating
  console.log('🔍 Fetching projects from database...\n')

  const { data: projects, error: projectsError } = await supabase
    .from('projects')
    .select('id, name, document_storage_path, urls')
    .limit(100)

  if (projectsError) {
    console.error('Error fetching projects:', projectsError)
    return
  }

  console.log(`Found ${projects.length} project(s) to potentially update\n`)

  // Update each project
  let updatedCount = 0

  for (const project of projects) {
    // For now, assign the first PDF from storage if the project doesn't have a valid storage path
    const needsUpdate = !project.document_storage_path ||
                        (!project.document_storage_path.startsWith('http') &&
                         !project.document_storage_path.includes('storage/v1/object/public'))

    if (needsUpdate && allFiles.length > 0) {
      // Assign a file (for demo purposes, using the first one, but you could match by project name)
      const file = allFiles[0] // Could be smarter about matching

      const { error: updateError } = await supabase
        .from('projects')
        .update({
          document_storage_path: file.fullUrl,
          urls: [file.fullUrl]
        })
        .eq('id', project.id)

      if (updateError) {
        console.error(`❌ Failed to update ${project.name}:`, updateError.message)
      } else {
        console.log(`✅ Updated ${project.name}`)
        console.log(`   → ${file.fullUrl}`)
        updatedCount++
      }
    } else if (project.document_storage_path) {
      console.log(`⏭️  Skipping ${project.name} (already has storage path)`)
    }
  }

  console.log(`\n✨ Updated ${updatedCount} project(s)`)
}

main()
