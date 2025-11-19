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
  console.log('📊 Population Progress Check\n')

  const { count, error } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  if (error) {
    console.error('❌ Error:', error.message)
    return
  }

  console.log(`✅ Total projects: ${count}`)

  // Get recent projects
  const { data: recent } = await supabase
    .from('projects')
    .select('name, created_at')
    .order('created_at', { ascending: false })
    .limit(5)

  if (recent && recent.length > 0) {
    console.log('\n📄 Most recent projects:')
    recent.forEach(p => {
      const timeAgo = Math.floor((Date.now() - new Date(p.created_at).getTime()) / 1000)
      console.log(`   • ${p.name} (${timeAgo}s ago)`)
    })
  }

  // Count by company
  const { data: byCompany } = await supabase
    .from('projects')
    .select('company_id')

  if (byCompany) {
    const uniqueCompanies = new Set(byCompany.map(p => p.company_id)).size
    console.log(`\n🏢 Companies with projects: ${uniqueCompanies}`)
  }
}

main()
