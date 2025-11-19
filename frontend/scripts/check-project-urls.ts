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
  console.log('📊 Checking project URLs...\n')

  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, document_storage_path, urls')
    .limit(5)

  if (error) {
    console.error('Error:', error.message)
    return
  }

  projects.forEach((project, i) => {
    console.log(`${i + 1}. ${project.name}`)
    console.log(`   document_storage_path: ${project.document_storage_path}`)
    console.log(`   urls: ${JSON.stringify(project.urls)}`)
    console.log('')
  })
}

main()
