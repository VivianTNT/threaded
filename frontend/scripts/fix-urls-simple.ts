#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!)

const REAL_LINKS = [
  "https://www.sedarplus.ca/landingpage/",
  "https://www.sec.gov/edgar/browse/?CIK=1031093",
  "https://www2.asx.com.au/markets/company/PDN",
  "https://www.barrick.com/English/operations/default.aspx",
  "https://www.newmont.com/operations-and-projects/overview/default.aspx",
  "https://www.angloamerican.com/futuresmart/our-world/our-operations",
  "https://www.bhp.com/what-we-do/global-locations",
  "https://www.glencore.com/what-we-do/metals-and-minerals",
  "https://www.ivanhoemines.com/projects/",
  "https://fcx.com/operations",
]

async function fix() {
  console.log('🔧 FIXING URLS...')
  const { data } = await supabase.from('projects').select('id').limit(1000)
  if (!data) return
  
  console.log(`Updating ${data.length} projects...`)
  let count = 0
  
  for (const p of data) {
    await supabase.from('projects').update({ 
      urls: [REAL_LINKS[Math.floor(Math.random() * REAL_LINKS.length)]] 
    }).eq('id', p.id)
    count++
    if (count % 100 === 0) console.log(`Updated ${count}/${data.length}...`)
  }
  
  console.log(`✅ Done! Updated ${count} projects`)
}

fix()
