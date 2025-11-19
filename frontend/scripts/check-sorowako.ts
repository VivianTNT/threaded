import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function checkSorowako() {
  // Search for Sorowako
  const { data: projects } = await supabase
    .from('projects')
    .select('id, name, company_id')
    .or('name.ilike.%Sorowako%,name.ilike.%nickel%')

  console.log('Sorowako-related projects:', projects?.filter(p => p.name.toLowerCase().includes('sorowako')))

  // Check the added projects
  const { data: addedProjects } = await supabase
    .from('projects')
    .select('name, company_id, companies(name)')
    .in('name', [
      'Greenbushes Lithium Mine',
      'Pilgangoora Lithium-Tantalum Project',
      'Mount Marion Lithium Project',
      'Sorowako Nickel Mine',
      'Weda Bay Nickel Project'
    ])

  console.log('\nAdded projects with companies:')
  addedProjects?.forEach(p => {
    console.log(`- ${p.name}: ${(p as any).companies?.name || 'NO COMPANY'}`)
  })
}

checkSorowako()
