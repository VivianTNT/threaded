import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function linkProjectsToCompanies() {
  console.log('Linking projects to their companies...')

  // Project to company mappings
  const mappings = [
    { project: 'Greenbushes Lithium Mine', company: 'Talison Lithium (Tianqi/IGO/Albemarle JV)' },
    { project: 'Pilgangoora Lithium-Tantalum Project', company: 'Pilbara Minerals' },
    { project: 'Mount Marion Lithium Project', company: 'Mineral Resources' },
    { project: 'Wodgina Lithium Mine', company: 'Mineral Resources' },
    { project: 'Salar de Atacama Lithium Operations', company: 'Albemarle Corporation' },
    { project: 'Salar del Carmen Lithium Operations', company: 'Sociedad Quimica y Minera (SQM)' },
    { project: 'Mount Holland (Earl Grey)', company: 'Sociedad Quimica y Minera (SQM)' },
    { project: 'Goulamina Lithium Project', company: 'Ganfeng Lithium' },
    { project: 'Escondida Copper Mine', company: 'BHP Group' },
    { project: 'Collahuasi Copper Mine', company: 'Glencore' },
    { project: 'Grasberg Copper-Gold Mine', company: 'Freeport-McMoRan' },
    { project: 'Norilsk-Talnakh Nickel Complex', company: 'Nornickel' },
    { project: 'Weda Bay Nickel Project', company: 'Tsingshan Holding Group' },
    { project: 'Sorowako Nickel Mine', company: 'Vale SA' },
    { project: 'Tenke Fungurume Copper-Cobalt Mine', company: 'CMOC Group' },
    { project: 'Mutanda Copper-Cobalt Mine', company: 'Glencore' },
    { project: 'Bayan Obo Iron-REE Mine', company: 'Tsingshan Holding Group' },
    { project: 'Mountain Pass Rare Earth Mine', company: 'MP Materials' },
    { project: 'Mount Weld Rare Earth Mine', company: 'Lynas Rare Earths' }
  ]

  for (const mapping of mappings) {
    // Get company ID
    const { data: company } = await supabase
      .from('companies')
      .select('id')
      .eq('name', mapping.company)
      .single()

    if (!company) {
      console.log(`✗ Company not found: ${mapping.company}`)
      continue
    }

    // Update project with company ID
    const { error } = await supabase
      .from('projects')
      .update({ company_id: company.id })
      .eq('name', mapping.project)

    if (error) {
      console.log(`✗ Error updating ${mapping.project}: ${error.message}`)
    } else {
      console.log(`✓ Linked ${mapping.project} → ${mapping.company}`)
    }
  }
}

linkProjectsToCompanies()
  .then(() => {
    console.log('\nAll projects linked successfully!')
    console.log('Refresh the browser to see company names.')
    process.exit(0)
  })
  .catch((error) => {
    console.error('Error:', error)
    process.exit(1)
  })
