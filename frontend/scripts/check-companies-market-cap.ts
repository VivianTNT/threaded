import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseKey)

async function main() {
  console.log('Checking companies table...\n')

  // Get a sample company to see the data structure
  const { data, error } = await supabase
    .from('companies')
    .select('*')
    .limit(3)

  if (error) {
    console.error('Error:', error)
    return
  }

  console.log('Sample companies:')
  console.log(JSON.stringify(data, null, 2))
}

main()
