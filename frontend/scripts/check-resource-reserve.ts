import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseKey)

async function main() {
  console.log('📊 Checking resource and reserve values in projects table...\n')

  const { data, error } = await supabase
    .from('projects')
    .select('id, name, resource, reserve')
    .limit(10)

  if (error) {
    console.error('❌ Error:', error)
    return
  }

  console.log(`Found ${data.length} projects:\n`)

  data.forEach((project, index) => {
    console.log(`${index + 1}. ${project.name}`)
    console.log(`   ID: ${project.id}`)
    console.log(`   Resource: ${project.resource || 'NULL'}`)
    console.log(`   Reserve: ${project.reserve || 'NULL'}`)
    console.log()
  })

  // Count projects with values
  const withResource = data.filter(p => p.resource).length
  const withReserve = data.filter(p => p.reserve).length

  console.log(`\n📈 Summary:`)
  console.log(`   Projects with resource values: ${withResource}/${data.length}`)
  console.log(`   Projects with reserve values: ${withReserve}/${data.length}`)
}

main()
