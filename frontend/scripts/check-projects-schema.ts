#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkProjectsSchema() {
  console.log('Checking projects table schema...\n')

  // Try to insert a dummy project to see what columns exist
  const testProject = {
    id: crypto.randomUUID(),
    name: 'Test Project Schema',
    owner: 'Test Owner',
    location: 'Test Location',
    npv: 1000,
    irr: 20,
    capex: 500,
    urls: ['https://example.com/test.pdf'],
    watchlist: false,
    stage: 'Production',
    description: 'Test project',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }

  const { data, error } = await supabase
    .from('projects')
    .insert(testProject)
    .select()

  if (error) {
    console.log('Schema test error:', error.message)

    // Try without problematic fields
    const minimalProject = {
      id: crypto.randomUUID(),
      name: 'Minimal Test Project',
      owner: 'Test Owner',
      location: 'Test Location',
      npv: 1000,
      irr: 20,
      capex: 500,
      urls: ['https://example.com/test.pdf'],
      watchlist: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    const { data: minimalData, error: minimalError } = await supabase
      .from('projects')
      .insert(minimalProject)
      .select()

    if (!minimalError && minimalData) {
      console.log('\n✅ Minimal project created successfully!')
      console.log('Available columns:', Object.keys(minimalData[0]))

      // Clean up test project
      await supabase.from('projects').delete().eq('id', minimalData[0].id)
    } else {
      console.log('Minimal project error:', minimalError?.message)
    }
  } else if (data) {
    console.log('✅ Full project created successfully!')
    console.log('Available columns:', Object.keys(data[0]))

    // Clean up test project
    await supabase.from('projects').delete().eq('id', data[0].id)
  }

  // Also try to fetch an existing project to see its structure
  const { data: existingProject } = await supabase
    .from('projects')
    .select('*')
    .limit(1)
    .single()

  if (existingProject) {
    console.log('\n📊 Existing project structure:')
    console.log('Columns:', Object.keys(existingProject))
    console.log('\nSample data:')
    Object.entries(existingProject).forEach(([key, value]) => {
      if (value !== null) {
        console.log(`  ${key}: ${typeof value === 'object' ? JSON.stringify(value) : value}`)
      }
    })
  }
}

checkProjectsSchema().catch(console.error)