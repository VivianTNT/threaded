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
  console.log('Checking buckets...')

  const { data: buckets, error } = await supabase.storage.listBuckets()

  if (error) {
    console.error('Error:', error)
    return
  }

  console.log('Existing buckets:')
  buckets?.forEach(b => console.log(`  - ${b.name}`))

  // Check if technical-documents exists
  const exists = buckets?.find(b => b.name === 'technical-documents')
  console.log(`\ntechnical-documents bucket ${exists ? 'EXISTS' : 'DOES NOT EXIST'}`)
}

main().catch(console.error)