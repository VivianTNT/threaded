import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseKey)

async function main() {
  console.log('🔍 Checking actual database data...\n')

  // Get a specific project
  const { data, error } = await supabase
    .from('projects')
    .select('id, name, resource, reserve, npv, irr, capex')
    .limit(5)

  if (error) {
    console.error('❌ Error:', error)
    return
  }

  console.log('Raw database response:')
  console.log(JSON.stringify(data, null, 2))
}

main()
