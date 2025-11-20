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
  console.log('📦 Checking mining-documents bucket...\n')

  // List files in mining-documents bucket
  const { data: files, error } = await supabase.storage
    .from('mining-documents')
    .list('', { limit: 1000 })

  if (error) {
    console.error('Error:', error.message)
    return
  }

  console.log(`Found ${files.length} file(s) in mining-documents:\n`)

  files.forEach((file, index) => {
    const fullUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/mining-documents/${file.name}`
    const sizeMB = file.metadata?.size ? (file.metadata.size / 1024 / 1024).toFixed(2) : 'Unknown'

    console.log(`${index + 1}. ${file.name}`)
    console.log(`   Size: ${sizeMB} MB`)
    console.log(`   URL: ${fullUrl}`)
    console.log('')
  })
}

main()
