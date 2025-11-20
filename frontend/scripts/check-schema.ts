#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('Testing project insert...\n')

  const testCompanyId = crypto.randomUUID()

  // First insert a test company
  const { data: company, error: companyError } = await supabase
    .from('companies')
    .insert({
      id: testCompanyId,
      name: 'Test Company',
      ticker_symbol: 'TEST',
      primary_commodity: 'Copper'
    })
    .select()

  if (companyError) {
    console.error('Company insert error:', companyError)
    return
  }

  console.log('Company created:', testCompanyId)

  // Try to insert a test project
  const testProjectId = crypto.randomUUID()
  const { data: project, error: projectError } = await supabase
    .from('projects')
    .insert({
      id: testProjectId,
      company_id: testCompanyId,
      name: 'Test Project',
      location: 'Test Location',
      stage: 'Operating',
      commodities: ['Copper'],
      status: 'Active',
      urls: ['https://example.com/doc1.pdf', 'https://example.com/doc2.pdf'],
      description: 'Test project',
      watchlist: false
    })
    .select()

  if (projectError) {
    console.error('Project insert error:', projectError)
  } else {
    console.log('Project created successfully!')
    console.log('Project data:', project)
  }

  // Clean up
  await supabase.from('projects').delete().eq('id', testProjectId)
  await supabase.from('companies').delete().eq('id', testCompanyId)
}

main().catch(console.error)
