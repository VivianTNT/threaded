#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(process.env.NEXT_PUBLIC_SUPABASE_URL!, process.env.SUPABASE_SERVICE_ROLE_KEY!)

async function check() {
  console.log('🔍 CHECKING DATABASE FOR PROJECTS...\n')
  
  // Total count
  const { count } = await supabase.from('projects').select('*', { count: 'exact', head: true })
  console.log(`📊 Total projects: ${count}\n`)
  
  // Recent 10 with all fields
  const { data } = await supabase
    .from('projects')
    .select('id, name, location, commodities, stage, npv, irr, capex, resource, reserve, urls, document_storage_path, created_at, is_private, user_id')
    .order('created_at', { ascending: false })
    .limit(10)
  
  console.log('📋 LATEST 10 PROJECTS:\n')
  data?.forEach((p, i) => {
    console.log(`${i+1}. ${p.name}`)
    console.log(`   Location: ${p.location || 'N/A'}`)
    console.log(`   Commodity: ${p.commodities?.[0] || 'N/A'}`)
    console.log(`   Stage: ${p.stage || 'N/A'}`)
    console.log(`   NPV: $${p.npv}M | IRR: ${p.irr}%`)
    console.log(`   CAPEX: $${p.capex}M | Resource: ${p.resource}Mt | Reserve: ${p.reserve}Mt`)
    console.log(`   is_private: ${p.is_private}`)
    console.log(`   user_id: ${p.user_id}`)
    console.log(`   PDF: ${p.urls?.[0] || p.document_storage_path || 'N/A'}`)
    console.log(`   Created: ${p.created_at}`)
    console.log('')
  })
  
  // Count by is_private
  const { count: publicCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .is('is_private', false)
  
  const { count: privateCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .is('is_private', true)
    
  const { count: nullPrivateCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .is('is_private', null)
  
  console.log('📊 VISIBILITY:')
  console.log(`   Public (is_private=false): ${publicCount}`)
  console.log(`   Private (is_private=true): ${privateCount}`)
  console.log(`   NULL (is_private=null): ${nullPrivateCount}`)
}

check()
