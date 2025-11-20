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
  console.log('🔍 Recent Projects\n')

  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, company_id, document_storage_path, created_at')
    .order('created_at', { ascending: false })
    .limit(10)

  if (error) {
    console.error('❌ Error:', error.message)
    return
  }

  console.log(`✅ Latest ${projects.length} projects:\n`)

  for (const project of projects) {
    console.log(`📄 ${project.name}`)
    console.log(`   Created: ${new Date(project.created_at).toLocaleString()}`)
    console.log(`   Document: ${project.document_storage_path ? '✅ Yes' : '❌ No'}`)
    console.log()
  }
}

main()
