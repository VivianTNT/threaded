#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkUrls() {
  const { data } = await supabase
    .from('projects')
    .select('name, location, commodities, stage, npv, irr, urls, description')
    .not('urls', 'is', null)
    .order('created_at', { ascending: false })
    .limit(10)

  console.log('📋 LATEST 10 PROJECTS WITH TECHNICAL DOCUMENTATION:\n')
  data?.forEach((p, i) => {
    console.log(`${i + 1}. ${p.name}`)
    console.log(`   📍 Location: ${p.location}`)
    console.log(`   ⛏️  Commodity: ${p.commodities?.[0]}`)
    console.log(`   📊 Stage: ${p.stage}`)
    console.log(`   💰 NPV: $${p.npv}M | IRR: ${p.irr}%`)
    console.log(`   📄 Tech Doc: ${p.urls?.[0]}`)
    console.log('')
  })

  const { count: totalWithUrls } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .not('urls', 'is', null)

  const { count: total } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  console.log(`\n✅ ${totalWithUrls} out of ${total} projects have technical documentation links!`)
}

checkUrls()
