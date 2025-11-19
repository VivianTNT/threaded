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
  const { data: project, error } = await supabase
    .from('projects')
    .select('*')
    .eq('name', 'Sinkhole at the Alcaparrosa Mine')
    .single()

  if (error) {
    console.error('Error:', error.message)
  } else {
    console.log('\n📋 Project Details:')
    console.log(JSON.stringify(project, null, 2))
  }
}

main()
