import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseKey)

async function main() {
  console.log('🗑️  Clearing existing companies...')

  // Delete all existing companies
  const { error: deleteError } = await supabase
    .from('companies')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000') // Delete all rows

  if (deleteError) {
    console.error('Error clearing companies:', deleteError)
    return
  }

  console.log('✅ Companies table cleared')
  console.log('\n🔍 Starting comprehensive research for global mining companies...')
  console.log('This will take a few minutes to gather accurate data...\n')
}

main()
