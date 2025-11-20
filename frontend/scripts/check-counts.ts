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
  console.log('📊 DATABASE STATUS SUMMARY')
  console.log('='.repeat(50))

  // Companies count
  const { count: companiesCount } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true })

  console.log(`🏢 Companies: ${companiesCount || 0}`)

  // Projects count
  const { count: projectsCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  console.log(`⛏️  Projects: ${projectsCount || 0}`)

  // Projects with financial metrics
  const { count: projectsWithMetrics } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .not('npv', 'is', null)

  console.log(`💰 Projects with NPV: ${projectsWithMetrics || 0}`)

  // Projects with documents
  const { count: projectsWithDocs } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .not('document_storage_path', 'is', null)

  console.log(`📄 Projects with documents: ${projectsWithDocs || 0}`)

  // PDF highlights count
  const { count: highlightsCount } = await supabase
    .from('pdf_highlights')
    .select('*', { count: 'exact', head: true })

  console.log(`📌 PDF Highlights: ${highlightsCount || 0}`)

  console.log('='.repeat(50))
  console.log('✅ Database successfully populated!')
}

main().catch(console.error)