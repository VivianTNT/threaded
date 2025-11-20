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
  const { count, error } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true })

  if (error) {
    console.error('Error:', error.message)
  } else {
    console.log(`Companies in database: ${count}`)
  }
}

main()
