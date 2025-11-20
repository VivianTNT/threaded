#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function testPdfUrlAccess() {
  console.log('🧪 Testing PDF URL access...\n')

  try {
    // Get a random project with PDF
    const { data: project, error } = await supabase
      .from('projects')
      .select('id, name, document_storage_path')
      .not('document_storage_path', 'is', null)
      .limit(1)
      .single()

    if (error || !project) {
      throw new Error('No project found')
    }

    console.log(`📋 Testing Project: ${project.name}`)
    console.log(`🔗 PDF URL: ${project.document_storage_path}\n`)

    // Try to fetch the PDF URL
    console.log('🌐 Attempting to fetch PDF...')
    const response = await fetch(project.document_storage_path)

    console.log(`📊 Response Status: ${response.status} ${response.statusText}`)
    console.log(`📝 Content-Type: ${response.headers.get('content-type')}`)

    if (response.status === 200) {
      const contentLength = response.headers.get('content-length')
      console.log(`✅ PDF accessible! Size: ${contentLength ? (parseInt(contentLength) / 1024).toFixed(2) + ' KB' : 'Unknown'}`)
    } else if (response.status === 404) {
      console.error('❌ PDF not found (404)')
      const text = await response.text()
      console.error('Response:', text.substring(0, 200))

      // Check if file exists in storage
      console.log('\n🔍 Checking storage bucket...')
      const fileName = project.document_storage_path.split('/').pop()
      const { data: fileData, error: fileError } = await supabase.storage
        .from('mining-documents')
        .list('', { limit: 1000 })

      if (fileError) {
        console.error('Error listing files:', fileError.message)
      } else {
        const fileExists = fileData?.some(f => f.name === fileName)
        console.log(`File ${fileName} exists in bucket: ${fileExists ? '✅ Yes' : '❌ No'}`)

        if (fileExists) {
          console.log('\n💡 File exists but URL returns 404. This suggests a bucket policy issue.')
          console.log('The mining-documents bucket may not be publicly accessible.')
        }
      }
    } else {
      console.error(`❌ Unexpected status: ${response.status}`)
    }

  } catch (error: any) {
    console.error('❌ Error:', error.message)
  }
}

testPdfUrlAccess()
