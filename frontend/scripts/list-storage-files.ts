import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!)

async function listFiles() {
  const { data, error } = await supabase.storage.from('factset-documents').list('LI-CA', { limit: 10 })
  
  if (error) {
    console.error('Error:', error.message)
    return
  }
  
  console.log('📁 Files in factset-documents/LI-CA:')
  data?.forEach(file => {
    console.log(`   ${file.name} (${(file.metadata.size / 1024 / 1024).toFixed(2)} MB)`)
    const url = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/factset-documents/LI-CA/${file.name}`
    console.log(`   🔗 ${url}\n`)
  })
  
  console.log(`\nTotal: ${data?.length || 0} file(s)`)
}

listFiles()
