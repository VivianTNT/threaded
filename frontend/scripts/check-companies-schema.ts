import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
)

async function checkSchema() {
  const { data, error } = await supabase
    .from('companies')
    .select('*')
    .limit(1)

  if (error) {
    console.error('Error:', error)
  } else {
    console.log('Sample company record:')
    console.log(JSON.stringify(data, null, 2))

    if (data && data[0]) {
      console.log('\nCurrent fields:')
      console.log(Object.keys(data[0]).join(', '))
    }
  }
}

checkSchema()
