#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkSchema() {
  console.log('Checking database schema...\n')

  // Check projects table columns
  const { data: projectSample, error: projectError } = await supabase
    .from('projects')
    .select('*')
    .limit(1)

  if (projectSample && projectSample.length > 0) {
    console.log('Projects table columns:')
    console.log(Object.keys(projectSample[0]))
  }

  // Check companies table columns
  const { data: companySample, error: companyError } = await supabase
    .from('companies')
    .select('*')
    .limit(1)

  if (companySample && companySample.length > 0) {
    console.log('\nCompanies table columns:')
    console.log(Object.keys(companySample[0]))
  }
}

checkSchema()