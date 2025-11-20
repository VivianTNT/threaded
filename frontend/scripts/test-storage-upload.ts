#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function testUpload() {
  console.log('🧪 Testing Supabase Storage Upload\n')

  // Test 1: Check bucket exists
  console.log('1️⃣  Checking bucket existence...')
  const { data: buckets, error: bucketsError } = await supabase.storage.listBuckets()

  if (bucketsError) {
    console.error('❌ Error listing buckets:', bucketsError.message)
    return
  }

  console.log('✅ Found buckets:', buckets?.map(b => b.name).join(', '))

  const factsetBucket = buckets?.find(b => b.name === 'factset-documents')
  if (!factsetBucket) {
    console.log('⚠️  factset-documents bucket not found!')
    console.log('📝 Creating bucket...')

    const { error: createError } = await supabase.storage.createBucket('factset-documents', {
      public: false,
      fileSizeLimit: 52428800 // 50 MB
    })

    if (createError) {
      console.error('❌ Error creating bucket:', createError.message)
      return
    }
    console.log('✅ Bucket created')
  } else {
    console.log(`✅ Bucket exists (public: ${factsetBucket.public})`)
  }

  // Test 2: Small file upload via SDK
  console.log('\n2️⃣  Testing small file upload (SDK)...')
  const smallBuffer = Buffer.from('Test content for small file upload')

  const { data: smallData, error: smallError } = await supabase.storage
    .from('factset-documents')
    .upload('test/small-test.txt', smallBuffer, {
      contentType: 'text/plain',
      upsert: true
    })

  if (smallError) {
    console.error('❌ Small file upload failed:', smallError.message)
  } else {
    console.log('✅ Small file uploaded successfully')
  }

  // Test 3: Large file upload via HTTP API (simulated with 1MB buffer)
  console.log('\n3️⃣  Testing large file upload (HTTP API)...')
  const largeBuffer = Buffer.alloc(1024 * 1024, 'A') // 1 MB test file

  const uploadUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/factset-documents/test/large-test.bin`

  try {
    const uploadResponse = await fetch(uploadUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.SUPABASE_SERVICE_ROLE_KEY}`,
        'Content-Type': 'application/octet-stream',
        'x-upsert': 'true'
      },
      body: largeBuffer,
      // @ts-ignore
      signal: AbortSignal.timeout(60000) // 60 second timeout
    })

    if (!uploadResponse.ok) {
      const errorText = await uploadResponse.text()
      console.error(`❌ HTTP upload failed: ${uploadResponse.status} - ${errorText}`)
    } else {
      console.log('✅ Large file uploaded successfully via HTTP')
    }
  } catch (error: any) {
    console.error('❌ HTTP upload error:', error.message)
  }

  // Test 4: List uploaded files
  console.log('\n4️⃣  Listing uploaded test files...')
  const { data: files, error: listError } = await supabase.storage
    .from('factset-documents')
    .list('test')

  if (listError) {
    console.error('❌ Error listing files:', listError.message)
  } else {
    console.log('✅ Files in test folder:', files?.map(f => f.name).join(', ') || 'none')
  }

  console.log('\n✅ Test complete')
}

testUpload().catch(console.error)
