#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

// Minimum size for 100+ page documents (roughly 1MB for good quality PDFs)
const MIN_SIZE_MB = 1.0

async function listLargeDocs() {
  console.log('=' .repeat(80))
  console.log('📊 LISTING LARGE TECHNICAL DOCUMENTS (100+ pages)')
  console.log('=' .repeat(80))

  const buckets = ['factset-documents', 'mining-documents', 'technical-reports']
  const largeDocs: any[] = []

  for (const bucket of buckets) {
    console.log(`\n📂 Checking bucket: ${bucket}`)

    // List files recursively
    async function listFiles(prefix: string = ''): Promise<any[]> {
      const { data, error } = await supabase.storage
        .from(bucket)
        .list(prefix, { limit: 1000 })

      if (error || !data) return []

      let files: any[] = []
      for (const item of data) {
        if (!item.id) {
          // It's a folder
          const subPath = prefix ? `${prefix}/${item.name}` : item.name
          const subFiles = await listFiles(subPath)
          files = files.concat(subFiles)
        } else {
          // It's a file
          const fullPath = prefix ? `${prefix}/${item.name}` : item.name
          const sizeMB = (item.metadata?.size || 0) / (1024 * 1024)

          if (sizeMB >= MIN_SIZE_MB && item.name.toLowerCase().endsWith('.pdf')) {
            files.push({
              bucket,
              path: fullPath,
              name: item.name,
              sizeMB: sizeMB.toFixed(2)
            })
          }
        }
      }
      return files
    }

    const files = await listFiles()
    largeDocs.push(...files)
    console.log(`  Found ${files.length} PDFs larger than ${MIN_SIZE_MB}MB`)
  }

  console.log('\n' + '=' .repeat(80))
  console.log('📚 LARGE TECHNICAL DOCUMENTS FOUND')
  console.log('=' .repeat(80))

  // Sort by size
  largeDocs.sort((a, b) => parseFloat(b.sizeMB) - parseFloat(a.sizeMB))

  console.log(`\nTotal: ${largeDocs.length} documents\n`)

  // Show top 20 largest
  for (const doc of largeDocs.slice(0, 20)) {
    console.log(`📄 ${doc.sizeMB} MB - ${doc.bucket}/${doc.path}`)
  }

  if (largeDocs.length > 20) {
    console.log(`\n... and ${largeDocs.length - 20} more documents`)
  }

  // Summary by bucket
  console.log('\n📊 Summary by bucket:')
  const bucketSummary: Record<string, number> = {}
  for (const doc of largeDocs) {
    bucketSummary[doc.bucket] = (bucketSummary[doc.bucket] || 0) + 1
  }

  for (const [bucket, count] of Object.entries(bucketSummary)) {
    console.log(`  ${bucket}: ${count} documents`)
  }
}

listLargeDocs().catch(console.error)