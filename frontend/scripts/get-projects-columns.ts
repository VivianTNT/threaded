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
  console.log('🔍 Getting projects table columns...\n')

  // Fetch one project with all columns
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .limit(1)

  if (error && error.code !== 'PGRST116') {
    console.error('❌ Error:', error.message)
    return
  }

  // If we have data, show the columns
  if (data && data.length > 0) {
    console.log('📊 Columns found in projects table:')
    Object.keys(data[0]).forEach(col => {
      console.log(`   - ${col}: ${typeof data[0][col]}`)
    })
  } else {
    // Try to insert a dummy record to see what columns are expected
    console.log('No data found. Trying to insert a test record to discover schema...')

    const testData = {
      id: 'test-' + Date.now(),
      name: 'Test Project',
      company_id: null,
      location: 'Test Location',
      stage: 'exploration',
      commodities: ['Gold'],
      resource_estimate: '1.0 Moz',
      reserve_estimate: '0.5 Moz',
      ownership_percentage: 100,
      status: 'active',
      description: 'Test',
      urls: ['https://example.com'],
      npv: 1000000,
      irr: 15,
      capex: 500000,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }

    const { data: insertData, error: insertError } = await supabase
      .from('projects')
      .insert(testData)
      .select()

    if (insertError) {
      console.log('❌ Insert error (this helps us understand the schema):')
      console.log(insertError.message)

      // The error message often reveals the actual column names
      if (insertError.message.includes('column')) {
        console.log('\n📝 This error indicates the actual schema structure')
      }
    } else {
      console.log('✅ Test insert successful!')
      console.log('📊 Actual columns:')
      if (insertData && insertData[0]) {
        Object.keys(insertData[0]).forEach(col => {
          console.log(`   - ${col}`)
        })

        // Clean up test data
        await supabase.from('projects').delete().eq('id', testData.id)
      }
    }
  }
}

main()