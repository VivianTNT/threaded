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
  console.log('📦 Checking technical-documents bucket...\n')

  // List all buckets first
  const { data: buckets, error: bucketsError } = await supabase.storage.listBuckets()

  if (bucketsError) {
    console.error('Error listing buckets:', bucketsError)
    return
  }

  console.log('Available buckets:')
  buckets.forEach(bucket => console.log(`  - ${bucket.name}`))
  console.log('')

  // Check if technical-documents exists
  const hasTechnicalDocs = buckets.some(b => b.name === 'technical-documents')

  if (!hasTechnicalDocs) {
    console.log('❌ technical-documents bucket not found')
    return
  }

  // List files in technical-documents/mining-documents
  const { data: files, error: filesError } = await supabase.storage
    .from('technical-documents')
    .list('mining-documents', { limit: 1000 })

  if (filesError) {
    console.error('Error listing files:', filesError.message)
    return
  }

  console.log(`\n📄 Files in technical-documents/mining-documents:\n`)

  files.forEach((file, index) => {
    const fullUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/technical-documents/mining-documents/${file.name}`
    const sizeMB = file.metadata?.size ? (file.metadata.size / 1024 / 1024).toFixed(2) : 'Unknown'

    console.log(`${index + 1}. ${file.name}`)
    console.log(`   Size: ${sizeMB} MB`)
    console.log(`   URL: ${fullUrl}`)
    console.log('')
  })

  console.log(`\nTotal: ${files.length} file(s)`)
}

main()
