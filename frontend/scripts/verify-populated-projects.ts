#!/usr/bin/env npx tsx

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('🔍 Verifying Populated Projects\n')

  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, company_id, document_storage_path, urls')
    .limit(5)

  if (error) {
    console.error('❌ Error:', error.message)
    return
  }

  console.log(`✅ Found ${projects.length} sample projects:\n`)

  for (const project of projects) {
    console.log(`📄 ${project.name}`)
    console.log(`   Company ID: ${project.company_id}`)
    console.log(`   Document Path: ${project.document_storage_path ? '✅ ' + project.document_storage_path : '❌ None'}`)
    console.log(`   URLs: ${project.urls?.length || 0} links`)
    console.log()
  }
}

main()
