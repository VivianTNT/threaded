import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

const supabase = createClient(supabaseUrl, anonKey)

async function main() {
  console.log('🔍 Testing with ANON key (same as frontend)...\n')

  const { data, error } = await supabase
    .from('projects')
    .select('id, name, resource, reserve')
    .limit(3)

  if (error) {
    console.error('❌ Error:', error)
    return
  }

  console.log('Response with ANON key:')
  console.log(JSON.stringify(data, null, 2))
}

main()
