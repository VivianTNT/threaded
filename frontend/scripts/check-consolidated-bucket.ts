#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function check() {
  console.log('📂 Checking technical-documents bucket...\n')

  // List top-level folders
  const { data: topLevel, error } = await supabase.storage
    .from('technical-documents')
    .list('', { limit: 100 })

  if (error) {
    console.error('Error:', error)
    return
  }

  let totalFiles = 0
  let totalSizeMB = 0
  const folderStats: Array<{ name: string, count: number, sizeMB: number }> = []

  for (const item of topLevel || []) {
    // Check if it's a folder
    const isFolder = !item.metadata && !item.id && !item.updated_at

    if (isFolder) {
      const { data: files } = await supabase.storage
        .from('technical-documents')
        .list(item.name, { limit: 1000 })

      if (files) {
        const pdfFiles = files.filter(f => f.name.endsWith('.pdf'))
        const folderSize = pdfFiles.reduce((sum, f) => sum + (f.metadata?.size || 0), 0) / 1024 / 1024

        if (pdfFiles.length > 0) {
          folderStats.push({
            name: item.name,
            count: pdfFiles.length,
            sizeMB: folderSize
          })
          totalFiles += pdfFiles.length
          totalSizeMB += folderSize
        }
      }
    }
  }

  console.log('📊 CONSOLIDATED BUCKET SUMMARY')
  console.log('=' .repeat(60))

  folderStats.forEach(folder => {
    console.log(`📁 ${folder.name}:`)
    console.log(`   Files: ${folder.count} PDFs`)
    console.log(`   Size: ${folder.sizeMB.toFixed(2)} MB`)
    console.log(`   Avg: ${(folder.sizeMB / folder.count).toFixed(2)} MB per file`)
  })

  console.log('\n' + '=' .repeat(60))
  console.log(`TOTAL: ${totalFiles} PDFs, ${totalSizeMB.toFixed(2)} MB`)
  console.log(`Average file size: ${totalFiles > 0 ? (totalSizeMB / totalFiles).toFixed(2) : 0} MB`)
  console.log('=' .repeat(60))

  // Show a few sample files
  if (folderStats.length > 0) {
    console.log('\n📄 Sample files from first folder:')
    const { data: samples } = await supabase.storage
      .from('technical-documents')
      .list(folderStats[0].name, { limit: 5 })

    samples?.filter(f => f.name.endsWith('.pdf')).forEach(file => {
      const sizeMB = (file.metadata?.size || 0) / 1024 / 1024
      console.log(`  - ${file.name} (${sizeMB.toFixed(2)} MB)`)
    })
  }
}

check().catch(console.error)