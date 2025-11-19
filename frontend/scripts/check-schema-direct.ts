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
  console.log('🔍 Checking projects table schema...\n')

  // Get a sample project to see actual columns
  const { data: sample, error } = await supabase
    .from('projects')
    .select('*')
    .limit(1)
    .maybeSingle()

  if (error) {
    console.error('Error:', error.message)
    return
  }

  if (sample) {
    console.log('✅ Sample project columns:')
    Object.keys(sample).forEach(key => {
      console.log(`   - ${key}: ${typeof sample[key]}`)
    })
  } else {
    console.log('No projects found in database')
  }
}

main()
