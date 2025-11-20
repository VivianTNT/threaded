#!/usr/bin/env node
import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
const supabase = createClient(SUPABASE_URL, SUPABASE_KEY)

async function main() {
  console.log('\n📊 VERIFYING FACTSET-POPULATED PROJECTS\n')
  console.log('='.repeat(70))

  const { data: projects, error } = await supabase
    .from('projects')
    .select('name, urls, companies(name, ticker)')
    .not('urls', 'is', null)
    .order('created_at', { ascending: false })
    .limit(15)

  if (error) {
    console.error('Error:', error)
    process.exit(1)
  }

  console.log(`\n✅ Found ${projects.length} recent projects with filing URLs:\n`)

  projects.forEach((p: any, i: number) => {
    console.log(`${(i+1).toString().padStart(2)}. ${p.name}`)
    console.log(`    Company: ${p.companies?.name} (${p.companies?.ticker})`)
    console.log(`    URLs: ${p.urls?.length || 0} filing link(s)`)
    if (p.urls && p.urls.length > 0) {
      console.log(`    Sample: ${p.urls[0].substring(0, 80)}...`)
    }
    console.log('')
  })

  console.log('='.repeat(70))
}

main()
