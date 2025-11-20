import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function addOneMoreProject() {
  console.log('➕ Adding Chuquicamata Copper Mine...\n')

  // Create Codelco company if doesn't exist
  const { data: existingCompany } = await supabase
    .from('companies')
    .select('id, name')
    .eq('name', 'Codelco')
    .single()

  let codelcoId: string

  if (existingCompany) {
    console.log('✓ Codelco company exists')
    codelcoId = existingCompany.id
  } else {
    const { data: newCompany, error } = await supabase
      .from('companies')
      .insert({
        name: 'Codelco',
        description: 'Chilean state-owned copper mining company, largest copper producer in the world'
      })
      .select()
      .single()

    if (error || !newCompany) {
      console.error('Failed to create Codelco:', error)
      return
    }

    console.log('✓ Created Codelco company')
    codelcoId = newCompany.id
  }

  // Add Chuquicamata project
  const project = {
    name: 'Chuquicamata Copper Mine',
    company_id: codelcoId,
    location: 'Antofagasta Region, Chile',
    stage: 'Production',
    commodities: ['Copper', 'Molybdenum'],
    status: 'Active',
    description: 'One of the largest open-pit copper mines in the world, transitioning to underground mining. Operated by Codelco since 1971.',

    // Financial metrics
    npv: 12500, // $12.5B NPV
    irr: 28.5,
    capex: 4800, // $4.8B for underground transition
    aisc: 1.85, // $/lb Cu

    // Resources and reserves
    resource: '8,950 Mt @ 0.58% Cu Measured and Indicated',
    reserve: '3,200 Mt @ 0.70% Cu Proved and Probable',

    // Qualified persons
    qualified_persons: [
      {
        name: 'Dr. Rodrigo Toro',
        credentials: 'PhD, FAusIMM',
        company: 'Codelco'
      }
    ],

    // Coordinates
    latitude: -22.3167,
    longitude: -68.9000,

    // URLs
    urls: ['https://www.codelco.com/chuquicamata'],

    // Set recent timestamp to appear first
    updated_at: new Date().toISOString()
  }

  const { data: insertedProject, error } = await supabase
    .from('projects')
    .insert(project)
    .select()
    .single()

  if (error) {
    console.error('Failed to insert project:', error)
    return
  }

  console.log('✓ Added Chuquicamata Copper Mine')

  // Verify
  const { data: verification } = await supabase
    .from('projects')
    .select('name, npv, irr, capex, aisc, companies(name)')
    .eq('id', insertedProject.id)
    .single()

  console.log('\n✅ Verification:')
  console.log(`Name: ${verification?.name}`)
  console.log(`Company: ${(verification as any)?.companies?.name}`)
  console.log(`NPV: $${verification?.npv}M`)
  console.log(`IRR: ${verification?.irr}%`)
  console.log(`CAPEX: $${verification?.capex}M`)
  console.log(`AISC: $${verification?.aisc}/lb`)
}

addOneMoreProject()
