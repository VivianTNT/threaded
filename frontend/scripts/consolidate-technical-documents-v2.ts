#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Minimum file size for 100+ page documents (500KB to be more inclusive)
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
  let totalSizeMB = 0
  const movedFiles: Array<{ source: string, name: string, sizeMB: number }> = []

  for (const sourceBucket of sourceBuckets) {
    console.log(`\n📁 Processing bucket: ${sourceBucket}`)
    console.log('-' .repeat(60))

    // First, get all files with their public URLs to check actual sizes
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
        if (item.name === '.emptyFolderPlaceholder') continue

        const fullPath = prefix ? `${prefix}/${item.name}` : item.name

        // Check if it's a folder (no metadata or id)
        if (!item.metadata && !item.id) {
          // It's a folder, recurse into it
          const subFiles = await listAllFiles(fullPath)
          allFiles = allFiles.concat(subFiles)
        } else {
          // It's a file - check if it's a PDF
          if (item.name.toLowerCase().endsWith('.pdf')) {
            // Try to get the actual file size
            const { data: publicUrl } = supabase.storage
              .from(sourceBucket)
              .getPublicUrl(fullPath)

            // Estimate size from metadata or use a default for PDFs
            let estimatedSize = 0

            // If the item has metadata with size, use it
            if (item.metadata?.size) {
              estimatedSize = item.metadata.size
            } else if (item.size) {
              estimatedSize = item.size
            } else {
              // For PDFs without size info, check if they're real files
              // Placeholder PDFs are usually very small or have specific patterns
              if (item.name.includes('placeholder') || item.name.includes('empty')) {
                estimatedSize = 0
              } else {
                // Assume it's a real PDF if it exists without size metadata
                // (this can happen with older uploads)
                estimatedSize = 1024 * 1024 // Default to 1MB for unknown sizes
              }
            }

            allFiles.push({
              name: item.name,
              fullPath,
              size: estimatedSize,
              bucket: sourceBucket
            })
          }
        }
      }

      return allFiles
    }

    const files = await listAllFiles()
    const sizeableFiles = files.filter(f => f.size >= MIN_SIZE_KB * 1024)

    console.log(`  Found ${files.length} PDFs total`)
    console.log(`  Found ${sizeableFiles.length} PDFs larger than ${MIN_SIZE_KB}KB`)

    // Process each sizeable PDF
    for (const file of sizeableFiles) {
      const sizeMB = (file.size / 1024 / 1024).toFixed(2)
      console.log(`\n  📄 Processing: ${file.name} (${sizeMB} MB)`)
      console.log(`     Path: ${file.fullPath}`)

      try {
        // Download the file
        const { data: downloadData, error: downloadError } = await supabase.storage
          .from(sourceBucket)
          .download(file.fullPath)

        if (downloadError) {
          console.log(`    ❌ Download error: ${downloadError.message}`)
          continue
        }

        // Check actual downloaded size
        const actualSize = downloadData.size
        const actualSizeMB = (actualSize / 1024 / 1024).toFixed(2)

        // Only proceed if the actual file is large enough
        if (actualSize < MIN_SIZE_KB * 1024) {
          console.log(`    ⚠️  Actual size (${actualSizeMB} MB) below threshold, skipping`)
          continue
        }

        // Create organized path in target bucket
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
          console.log(`    📏 Actual size: ${actualSizeMB} MB`)
          totalMoved++
          totalSizeMB += parseFloat(actualSizeMB)
          movedFiles.push({
            source: sourceBucket,
            name: file.name,
            sizeMB: parseFloat(actualSizeMB)
          })
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
  console.log(`Total size: ${totalSizeMB.toFixed(2)} MB`)
  console.log(`Average size: ${totalMoved > 0 ? (totalSizeMB / totalMoved).toFixed(2) : 0} MB per document`)

  if (movedFiles.length > 0) {
    console.log('\n📋 Consolidated Files:')
    movedFiles
      .sort((a, b) => b.sizeMB - a.sizeMB)
      .forEach(file => {
        console.log(`  📄 ${file.sizeMB.toFixed(2)} MB - ${file.source}/${file.name}`)
      })
  }

  // List final contents
  console.log(`\n📂 Verifying ${targetBucket} contents:`)
  const { data: verifyFiles } = await supabase.storage
    .from(targetBucket)
    .list('', { limit: 100 })

  if (verifyFiles) {
    const folders = verifyFiles.filter(f => !f.id)
    console.log(`  Found ${folders.length} source folders:`)

    for (const folder of folders) {
      const { data: subFiles } = await supabase.storage
        .from(targetBucket)
        .list(folder.name, { limit: 1000 })

      const pdfCount = subFiles?.filter(f => f.name.endsWith('.pdf')).length || 0
      if (pdfCount > 0) {
        console.log(`    📁 ${folder.name}/: ${pdfCount} PDFs`)
      }
    }
  }
}

consolidateDocuments().catch(console.error)