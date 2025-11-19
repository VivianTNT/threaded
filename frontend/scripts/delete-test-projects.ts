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
  console.log('🗑️  Deleting test projects...\n')

  // Get Alamos Gold company ID
  const { data: alamos } = await supabase
    .from('companies')
    .select('id, name')
    .ilike('name', '%Alamos Gold%')
    .single()

  if (alamos) {
    console.log(`Found: ${alamos.name} (${alamos.id})`)

    const { data: projects, error: fetchError } = await supabase
      .from('projects')
      .select('id, name')
      .eq('company_id', alamos.id)

    if (projects) {
      console.log(`\nDeleting ${projects.length} Alamos Gold projects...\n`)

      const { error: deleteError } = await supabase
        .from('projects')
        .delete()
        .eq('company_id', alamos.id)

      if (deleteError) {
        console.error('❌ Error:', deleteError.message)
      } else {
        console.log(`✅ Deleted ${projects.length} projects`)
      }
    }
  }

  // Also delete any other test projects (certificates, consents)
  console.log('\n🗑️  Deleting certificate/consent test projects...\n')

  const { data: testProjects } = await supabase
    .from('projects')
    .select('id, name')
    .or('name.ilike.%certificate%,name.ilike.%consent%')

  if (testProjects && testProjects.length > 0) {
    console.log(`Found ${testProjects.length} test projects to delete\n`)

    const { error: deleteError } = await supabase
      .from('projects')
      .delete()
      .or('name.ilike.%certificate%,name.ilike.%consent%')

    if (deleteError) {
      console.error('❌ Error:', deleteError.message)
    } else {
      console.log(`✅ Deleted ${testProjects.length} test projects`)
    }
  }

  // Check remaining count
  const { count } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  console.log(`\n📊 Remaining projects: ${count}`)
}

main()
