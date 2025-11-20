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
  console.log('🔍 Checking actual projects table schema...\n')

  // Method 1: Try to select all columns from projects
  const { data: projects, error: projectsError } = await supabase
    .from('projects')
    .select('*')
    .limit(1)

  if (projectsError) {
    console.error('❌ Error fetching projects:', projectsError.message)
  } else {
    console.log('✅ Projects table exists')
    console.log('📊 Sample project:', projects?.[0] ? 'Found' : 'No projects')

    if (projects?.[0]) {
      console.log('\n🔑 Column names in actual table:')
      Object.keys(projects[0]).forEach(col => {
        console.log(`   - ${col}`)
      })
    }
  }

  // Method 2: Get count
  const { count, error: countError } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  if (!countError) {
    console.log(`\n📈 Total projects: ${count}`)
  }
}

main()
