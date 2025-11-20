#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function createStorageBucket() {
  console.log('🪣 Creating Supabase Storage Bucket')
  console.log('='.repeat(60))

  // Create the bucket
  const { data, error } = await supabase.storage.createBucket('mining-documents', {
    public: true,
    fileSizeLimit: 524288000, // 500MB
    allowedMimeTypes: ['text/html', 'application/pdf', 'application/octet-stream']
  })

  if (error) {
    if (error.message.includes('already exists')) {
      console.log('✅ Bucket already exists!')
    } else {
      console.log('❌ Failed to create bucket:', error.message)
      return
    }
  } else {
    console.log('✅ Successfully created bucket: mining-documents')
  }

  console.log('\n✨ Storage bucket ready for document uploads!')
  console.log('📁 Bucket: mining-documents')
  console.log('🔒 Access: Public')
  console.log('📏 Size limit: 500MB per file')
}

createStorageBucket().catch(console.error)