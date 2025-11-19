import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function verifyAndFixChuquicamata() {
  console.log('🔍 Checking Chuquicamata...\n')

  // Find the project
  const { data: project } = await supabase
    .from('projects')
    .select('id, name, company_id')
    .eq('name', 'Chuquicamata Copper Mine')
    .single()

  if (!project) {
    console.log('❌ Project not found')
    return
  }

  console.log(`Found project: ${project.name}`)
  console.log(`Company ID: ${project.company_id || 'NULL'}\n`)

  // Check if Codelco exists
  const { data: codelco } = await supabase
    .from('companies')
    .select('id, name')
    .eq('name', 'Codelco')
    .single()

  if (!codelco) {
    console.log('Creating Codelco company...')
    const { data: newCompany } = await supabase
      .from('companies')
      .insert({
        name: 'Codelco',
        description: 'Chilean state-owned copper mining company, largest copper producer in the world'
      })
      .select()
      .single()

    if (newCompany) {
      console.log(`✓ Created Codelco (ID: ${newCompany.id})`)

      // Link to project
      await supabase
        .from('projects')
        .update({ company_id: newCompany.id })
        .eq('name', 'Chuquicamata Copper Mine')

      console.log('✓ Linked Chuquicamata to Codelco')
    }
  } else {
    console.log(`Codelco exists (ID: ${codelco.id})`)

    if (project.company_id !== codelco.id) {
      // Update the link
      await supabase
        .from('projects')
        .update({ company_id: codelco.id })
        .eq('name', 'Chuquicamata Copper Mine')

      console.log('✓ Updated link to Codelco')
    } else {
      console.log('✓ Already linked correctly')
    }
  }

  // Final verification
  const { data: final } = await supabase
    .from('projects')
    .select('name, companies(name)')
    .eq('name', 'Chuquicamata Copper Mine')
    .single()

  console.log(`\n✅ Final result: ${final?.name} → ${(final as any)?.companies?.name}`)
}

verifyAndFixChuquicamata()
