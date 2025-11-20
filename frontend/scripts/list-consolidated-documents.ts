#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function listDocuments() {
  console.log('📚 TECHNICAL DOCUMENTS IN CONSOLIDATED BUCKET')
  console.log('=' .repeat(80))

  const folders = ['factset-documents', 'mining-documents']

  for (const folder of folders) {
    console.log(`\n📁 ${folder}/`)
    console.log('-' .repeat(60))

    const { data: files, error } = await supabase.storage
      .from('technical-documents')
      .list(folder, { limit: 100 })

    if (error) {
      console.log(`Error: ${error.message}`)
      continue
    }

    const pdfFiles = files?.filter(f => f.name.endsWith('.pdf')) || []

    // Sort by size (largest first)
    pdfFiles.sort((a, b) => (b.metadata?.size || 0) - (a.metadata?.size || 0))

    pdfFiles.forEach((file, index) => {
      const sizeMB = ((file.metadata?.size || 0) / 1024 / 1024).toFixed(2)

      // Extract meaningful name from filename
      let displayName = file.name

      // Clean up the name for better readability
      if (file.name.includes('_')) {
        const parts = file.name.split('_')
        if (parts[0].includes('-')) {
          // Company ticker format (e.g., AGI-US, LUN)
          displayName = parts[0]
        } else if (parts.length > 1) {
          // UUID format - try to extract company/project info
          displayName = parts.slice(0, -1).join('_')
        }
      }

      // Determine document type based on name patterns
      let docType = '📄 Technical Report'
      if (file.name.includes('10-K') || file.name.includes('10K')) {
        docType = '📊 10-K Annual Report'
      } else if (file.name.includes('43-101') || file.name.includes('NI43101')) {
        docType = '⛏️ NI 43-101 Technical Report'
      } else if (file.name.includes('filer')) {
        docType = '📋 SEC Filing'
      } else if (sizeMB > '15') {
        docType = '📚 Comprehensive Technical Report'
      }

      console.log(`${index + 1}. ${docType}`)
      console.log(`   File: ${displayName}`)
      console.log(`   Size: ${sizeMB} MB`)

      // Add company/project context for known tickers
      const companyMap: Record<string, string> = {
        'AGI-US': 'Alamos Gold Inc.',
        'FM': 'Freeport-McMoRan',
        'LI-CA': 'Lithium Americas Corp.',
        'LUN': 'Lundin Mining',
        'NEM': 'Newmont Corporation',
        'BHP': 'BHP Group',
        'RIO': 'Rio Tinto'
      }

      for (const [ticker, company] of Object.entries(companyMap)) {
        if (file.name.includes(ticker)) {
          console.log(`   Company: ${company}`)
          break
        }
      }

      console.log()
    })

    console.log(`Total: ${pdfFiles.length} documents`)
  }

  console.log('\n' + '=' .repeat(80))
  console.log('📊 DOCUMENT HIGHLIGHTS:')
  console.log('=' .repeat(80))

  // Get all files for summary
  const allFiles: any[] = []
  for (const folder of folders) {
    const { data } = await supabase.storage
      .from('technical-documents')
      .list(folder, { limit: 100 })
    if (data) {
      allFiles.push(...data.filter(f => f.name.endsWith('.pdf')))
    }
  }

  // Find largest documents
  const largestDocs = allFiles
    .sort((a, b) => (b.metadata?.size || 0) - (a.metadata?.size || 0))
    .slice(0, 5)

  console.log('\n🏆 Largest Technical Documents:')
  largestDocs.forEach((doc, i) => {
    const sizeMB = ((doc.metadata?.size || 0) / 1024 / 1024).toFixed(2)
    console.log(`  ${i + 1}. ${doc.name.split('_')[0]} - ${sizeMB} MB`)
  })

  console.log('\n📈 Document Categories:')
  console.log('  • SEC Filings (10-K, 10-Q)')
  console.log('  • NI 43-101 Technical Reports')
  console.log('  • Feasibility Studies')
  console.log('  • Resource & Reserve Reports')
  console.log('  • Annual Information Forms')
}

listDocuments().catch(console.error)