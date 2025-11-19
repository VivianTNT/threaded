import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function fixHTMLContentType() {
  console.log('🔧 Fixing HTML file content-type...\n')

  const htmlUrl = 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/freeport-mcmoran/2025/0000831259-25-000031-1.html'
  const storagePath = 'factset/edg/freeport-mcmoran/2025/0000831259-25-000031-1.html'

  console.log('📄 Downloading HTML file...')
  const response = await fetch(htmlUrl)
  const htmlBuffer = await response.arrayBuffer()
  console.log(`✅ Downloaded: ${htmlBuffer.byteLength} bytes`)

  console.log('\n🗑️  Removing old file...')
  const { error: removeError } = await supabase.storage
    .from('refinitiv')
    .remove([storagePath])

  if (removeError) {
    console.log('⚠️  Remove error (file might not exist):', removeError.message)
  } else {
    console.log('✅ Old file removed')
  }

  console.log('\n📤 Re-uploading with correct content-type...')
  const { data, error } = await supabase.storage
    .from('refinitiv')
    .upload(storagePath, htmlBuffer, {
      contentType: 'text/html; charset=utf-8',
      upsert: true,
      cacheControl: '3600'
    })

  if (error) {
    console.error('❌ Upload error:', error)
    return
  }

  console.log('✅ File re-uploaded successfully!')
  console.log('📊 Storage path:', data.path)

  // Verify content-type
  console.log('\n🔍 Verifying content-type...')
  const verifyResponse = await fetch(htmlUrl, { method: 'HEAD' })
  const contentType = verifyResponse.headers.get('content-type')
  console.log('Content-Type:', contentType)

  if (contentType?.includes('text/html')) {
    console.log('✅ Content-Type is correct!')
  } else {
    console.log('❌ Content-Type is still wrong:', contentType)
    console.log('\n💡 Alternative solution: Use srcdoc attribute instead of src')
  }
}

fixHTMLContentType().catch(console.error)
