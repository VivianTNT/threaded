#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Minimum file size for 100+ page documents (500KB)
const MIN_SIZE_KB = 500

async function consolidateDocuments() {
  console.log('=' .repeat(80))
  console.log('📦 CONSOLIDATING TECHNICAL DOCUMENTS INTO ONE BUCKET')
  console.log('=' .repeat(80))

  const sourceBuckets = ['factset-documents', 'mining-documents', 'technical-reports']
  const targetBucket = 'technical-documents'

  // Create target bucket if it doesn't exist
  const { data: buckets } = await supabase.storage.listBuckets()
  if (!buckets?.find(b => b.name === targetBucket)) {
    // Create bucket without file size limit option that's causing the error
    const { error: createError } = await supabase.storage.createBucket(targetBucket, {
      public: true
    })
    if (createError) {
      console.error('Error creating bucket:', createError)
      return
    }
    console.log(`✅ Created new bucket: ${targetBucket}\n`)
  } else {
    console.log(`📂 Target bucket exists: ${targetBucket}\n`)
  }

  let totalMoved = 0
  let totalSizeMB = 0
  const movedFiles: Array<{ source: string, name: string, sizeMB: number, path: string }> = []

  // Process each source bucket
  for (const sourceBucket of sourceBuckets) {
    console.log(`\n📁 Scanning bucket: ${sourceBucket}`)
    console.log('-' .repeat(60))

    try {
      // Get top-level items
      const { data: topLevel, error: topError } = await supabase.storage
        .from(sourceBucket)
        .list('', { limit: 1000 })

      if (topError || !topLevel) {
        console.log(`  ❌ Error listing bucket: ${topError?.message}`)
        continue
      }

      let totalFiles = 0
      let largeFiles = 0

      // Process each top-level item
      for (const item of topLevel) {
        // Skip special files
        if (item.name === '.emptyFolderPlaceholder' || item.name.startsWith('.')) {
          continue
        }

        // Check if it's a folder
        const isFolder = !item.metadata && !item.id && !item.updated_at

        if (isFolder) {
          // List contents of folder
          const { data: folderContents } = await supabase.storage
            .from(sourceBucket)
            .list(item.name, { limit: 1000 })

          if (folderContents) {
            for (const file of folderContents) {
              if (file.name.toLowerCase().endsWith('.pdf') && file.metadata?.size) {
                totalFiles++
                const sizeMB = file.metadata.size / (1024 * 1024)

                if (file.metadata.size >= MIN_SIZE_KB * 1024) {
                  largeFiles++
                  const fullPath = `${item.name}/${file.name}`

                  // Download and re-upload large file
                  console.log(`  📄 Processing: ${fullPath} (${sizeMB.toFixed(2)} MB)`)

                  const { data: downloadData, error: downloadError } = await supabase.storage
                    .from(sourceBucket)
                    .download(fullPath)

                  if (!downloadError && downloadData) {
                    // Create clean filename
                    const cleanName = fullPath.replace(/[^a-zA-Z0-9_.-]/g, '_')
                    const newPath = `${sourceBucket}/${cleanName}`

                    const { error: uploadError } = await supabase.storage
                      .from(targetBucket)
                      .upload(newPath, downloadData, {
                        contentType: 'application/pdf',
                        upsert: true
                      })

                    if (!uploadError) {
                      console.log(`     ✅ Moved to: ${targetBucket}/${newPath}`)
                      totalMoved++
                      totalSizeMB += sizeMB
                      movedFiles.push({
                        source: sourceBucket,
                        name: file.name,
                        sizeMB: parseFloat(sizeMB.toFixed(2)),
                        path: newPath
                      })
                    } else {
                      console.log(`     ❌ Upload error: ${uploadError.message}`)
                    }
                  } else {
                    console.log(`     ❌ Download error: ${downloadError?.message}`)
                  }
                }
              }
            }
          }
        } else if (item.name.toLowerCase().endsWith('.pdf')) {
          // It's a PDF file at root level
          totalFiles++
          if (item.metadata?.size) {
            const sizeMB = item.metadata.size / (1024 * 1024)

            if (item.metadata.size >= MIN_SIZE_KB * 1024) {
              largeFiles++
              console.log(`  📄 Processing: ${item.name} (${sizeMB.toFixed(2)} MB)`)

              const { data: downloadData, error: downloadError } = await supabase.storage
                .from(sourceBucket)
                .download(item.name)

              if (!downloadError && downloadData) {
                const cleanName = item.name.replace(/[^a-zA-Z0-9_.-]/g, '_')
                const newPath = `${sourceBucket}/${cleanName}`

                const { error: uploadError } = await supabase.storage
                  .from(targetBucket)
                  .upload(newPath, downloadData, {
                    contentType: 'application/pdf',
                    upsert: true
                  })

                if (!uploadError) {
                  console.log(`     ✅ Moved to: ${targetBucket}/${newPath}`)
                  totalMoved++
                  totalSizeMB += sizeMB
                  movedFiles.push({
                    source: sourceBucket,
                    name: item.name,
                    sizeMB: parseFloat(sizeMB.toFixed(2)),
                    path: newPath
                  })
                } else {
                  console.log(`     ❌ Upload error: ${uploadError.message}`)
                }
              }
            }
          }
        }
      }

      console.log(`  📊 Found ${totalFiles} PDFs, ${largeFiles} are >${MIN_SIZE_KB}KB`)
    } catch (error: any) {
      console.log(`  ❌ Error processing bucket: ${error.message}`)
    }
  }

  console.log('\n' + '=' .repeat(80))
  console.log('📊 CONSOLIDATION SUMMARY')
  console.log('=' .repeat(80))
  console.log(`Documents consolidated: ${totalMoved}`)
  console.log(`Total size: ${totalSizeMB.toFixed(2)} MB`)
  console.log(`Average size: ${totalMoved > 0 ? (totalSizeMB / totalMoved).toFixed(2) : 0} MB per document`)

  if (movedFiles.length > 0) {
    console.log('\n📋 Top 10 Largest Consolidated Files:')
    movedFiles
      .sort((a, b) => b.sizeMB - a.sizeMB)
      .slice(0, 10)
      .forEach(file => {
        console.log(`  📄 ${file.sizeMB.toFixed(2)} MB - ${file.path}`)
      })
  }

  // Verify final contents
  console.log(`\n📂 Verifying ${targetBucket} contents:`)
  const { data: finalContents } = await supabase.storage
    .from(targetBucket)
    .list('', { limit: 100 })

  if (finalContents) {
    const folders = finalContents.filter(f => !f.metadata && !f.id)
    const files = finalContents.filter(f => f.metadata || f.id)

    console.log(`  📁 ${folders.length} folders`)
    console.log(`  📄 ${files.length} files at root`)

    // Check each folder
    for (const folder of folders) {
      const { data: subFiles } = await supabase.storage
        .from(targetBucket)
        .list(folder.name, { limit: 100 })

      if (subFiles) {
        const pdfCount = subFiles.filter(f => f.name.endsWith('.pdf')).length
        if (pdfCount > 0) {
          console.log(`     ${folder.name}/: ${pdfCount} PDFs`)
        }
      }
    }
  }
}

consolidateDocuments().catch(console.error)