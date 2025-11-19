#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Minimum file size for 100+ page documents (approx 300KB)
const MIN_SIZE_BYTES = 300 * 1024 // 300KB

async function consolidateDocuments() {
  console.log('=' .repeat(80))
  console.log('📦 CONSOLIDATING TECHNICAL DOCUMENTS')
  console.log('=' .repeat(80))

  const sourceBuckets = ['factset-documents', 'mining-documents', 'technical-reports']
  const targetBucket = 'technical-documents'

  // Create target bucket if it doesn't exist
  const { data: buckets } = await supabase.storage.listBuckets()
  if (!buckets?.find(b => b.name === targetBucket)) {
    const { error: createError } = await supabase.storage.createBucket(targetBucket, {
      public: true,
      fileSizeLimit: 104857600 // 100MB
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
  let totalSize = 0

  for (const sourceBucket of sourceBuckets) {
    console.log(`\n📁 Processing bucket: ${sourceBucket}`)
    console.log('-' .repeat(60))

    // List all files in bucket recursively
    async function listAllFiles(prefix: string = ''): Promise<any[]> {
      const { data, error } = await supabase.storage
        .from(sourceBucket)
        .list(prefix, { limit: 1000 })

      if (error || !data) {
        console.error(`  ❌ Error listing ${prefix}:`, error)
        return []
      }

      let allFiles: any[] = []

      for (const item of data) {
        if (!item.id) {
          // It's a folder, recurse into it
          const subPath = prefix ? `${prefix}/${item.name}` : item.name
          const subFiles = await listAllFiles(subPath)
          allFiles = allFiles.concat(subFiles)
        } else {
          // It's a file
          const fullPath = prefix ? `${prefix}/${item.name}` : item.name
          allFiles.push({
            ...item,
            fullPath,
            size: item.metadata?.size || 0
          })
        }
      }

      return allFiles
    }

    const files = await listAllFiles()
    console.log(`  Found ${files.length} total files`)

    // Filter for large PDF files (100+ pages)
    const largePDFs = files.filter(file => {
      const isPDF = file.name.toLowerCase().endsWith('.pdf')
      const isLarge = file.size >= MIN_SIZE_BYTES
      return isPDF && isLarge
    })

    console.log(`  Found ${largePDFs.length} PDFs larger than ${MIN_SIZE_BYTES / 1024}KB`)

    // Move/copy each large PDF to the consolidated bucket
    for (const file of largePDFs) {
      const sizeMB = (file.size / 1024 / 1024).toFixed(2)
      console.log(`\n  📄 ${file.name} (${sizeMB} MB)`)

      try {
        // Download the file
        const { data: downloadData, error: downloadError } = await supabase.storage
          .from(sourceBucket)
          .download(file.fullPath)

        if (downloadError) {
          console.log(`    ❌ Download error: ${downloadError.message}`)
          continue
        }

        // Create a clean filename
        const cleanName = file.fullPath.replace(/\//g, '_').replace(/[^a-zA-Z0-9_.-]/g, '')
        const newPath = `${sourceBucket}/${cleanName}`

        // Upload to consolidated bucket
        const { error: uploadError } = await supabase.storage
          .from(targetBucket)
          .upload(newPath, downloadData, {
            contentType: 'application/pdf',
            upsert: true
          })

        if (uploadError) {
          console.log(`    ❌ Upload error: ${uploadError.message}`)
        } else {
          console.log(`    ✅ Moved to: ${targetBucket}/${newPath}`)
          totalMoved++
          totalSize += file.size

          // Optionally delete from source (commented out for safety)
          // await supabase.storage.from(sourceBucket).remove([file.fullPath])
        }
      } catch (error: any) {
        console.log(`    ❌ Error: ${error.message}`)
      }
    }
  }

  console.log('\n' + '=' .repeat(80))
  console.log('📊 CONSOLIDATION SUMMARY')
  console.log('=' .repeat(80))
  console.log(`Documents consolidated: ${totalMoved}`)
  console.log(`Total size: ${(totalSize / 1024 / 1024).toFixed(2)} MB`)

  // List final contents of consolidated bucket
  console.log(`\n📂 Contents of ${targetBucket}:`)
  const { data: finalFiles } = await supabase.storage
    .from(targetBucket)
    .list('', { limit: 100 })

  if (finalFiles) {
    // Count files in each source folder
    const sourceCounts: Record<string, number> = {}
    for (const item of finalFiles) {
      if (!item.id) {
        const { data: subFiles } = await supabase.storage
          .from(targetBucket)
          .list(item.name, { limit: 1000 })

        sourceCounts[item.name] = subFiles?.filter(f => f.id).length || 0
      }
    }

    for (const [folder, count] of Object.entries(sourceCounts)) {
      console.log(`  📁 ${folder}/: ${count} documents`)
    }
  }
}

consolidateDocuments().catch(console.error)