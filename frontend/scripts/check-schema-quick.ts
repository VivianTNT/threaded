#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkSchema() {
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .limit(1)

  if (error) {
    console.error('Error:', error.message)
    return
  }

  if (data && data.length > 0) {
    console.log('Projects table columns:')
    console.log(Object.keys(data[0]).sort())
  }
}

checkSchema()
