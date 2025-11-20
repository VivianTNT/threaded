#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkStorageFiles() {
  console.log('🔍 Checking actual storage files...\n')

  try {
    // List all buckets
    const { data: buckets, error: bucketsError } = await supabase.storage.listBuckets()

    if (bucketsError) throw bucketsError

    console.log('📦 Available buckets:')
    buckets?.forEach(b => console.log(`   - ${b.name}`))
    console.log()

    // Check mining-documents bucket
    const { data: files, error: filesError } = await supabase.storage
      .from('mining-documents')
      .list('', { limit: 100 })

    if (filesError) {
      console.error('❌ Error listing mining-documents:', filesError.message)
    } else {
      console.log(`📁 mining-documents bucket: ${files?.length || 0} files`)
      files?.slice(0, 10).forEach((f, idx) => {
        console.log(`   ${idx + 1}. ${f.name}`)
      })
    }
    console.log()

    // Check technical-documents bucket
    const { data: techFiles, error: techError } = await supabase.storage
      .from('technical-documents')
      .list('', { limit: 100 })

    if (techError) {
      console.log('ℹ️  technical-documents bucket not accessible or empty')
    } else {
      console.log(`📁 technical-documents bucket: ${techFiles?.length || 0} files`)
      techFiles?.slice(0, 10).forEach((f, idx) => {
        console.log(`   ${idx + 1}. ${f.name}`)
      })
    }

  } catch (error: any) {
    console.error('❌ Error:', error.message)
  }
}

checkStorageFiles()
