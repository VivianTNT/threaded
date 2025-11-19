import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://dfxauievbyqwcynwtvib.supabase.co'
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ''

const supabase = createClient(supabaseUrl, supabaseKey)

async function checkColumns() {
  console.log('Checking actual database columns...\n')

  // Get one project to see all columns
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .limit(1)
    .single()

  if (error) {
    console.error('Error:', error)
    return
  }

  if (data) {
    console.log('Available columns in projects table:')
    Object.keys(data).sort().forEach(key => {
      console.log(`  - ${key}: ${typeof data[key]} (${data[key] === null ? 'NULL' : 'has value'})`)
    })
  }
}

checkColumns().catch(console.error)
