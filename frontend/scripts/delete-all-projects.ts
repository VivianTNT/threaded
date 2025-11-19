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
  console.log('🗑️  Deleting all projects...')

  // First check how many projects exist
  const { count } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  console.log(`📊 Found ${count} projects to delete`)

  // Delete all projects (use id column, not project_id)
  const { error } = await supabase
    .from('projects')
    .delete()
    .gte('id', '00000000-0000-0000-0000-000000000000')

  if (error) {
    console.error('❌ Error:', error.message)
  } else {
    console.log('✅ All projects deleted successfully')

    // Verify deletion
    const { count: finalCount } = await supabase
      .from('projects')
      .select('*', { count: 'exact', head: true })

    console.log(`📊 Remaining projects: ${finalCount}`)
  }
}

main()
