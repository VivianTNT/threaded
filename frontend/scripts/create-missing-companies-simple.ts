import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function createMissingCompanies() {
  console.log('Creating missing companies (simple version)...')

  const companies = [
    'Talison Lithium (Tianqi/IGO/Albemarle JV)',
    'Pilbara Minerals',
    'Mineral Resources',
    'Albemarle Corporation',
    'Sociedad Quimica y Minera (SQM)',
    'Ganfeng Lithium',
    'Nornickel',
    'Tsingshan Holding Group',
    'Vale SA',
    'CMOC Group',
    'MP Materials',
    'Lynas Rare Earths'
  ]

  for (const companyName of companies) {
    const { data: existing } = await supabase
      .from('companies')
      .select('id, name')
      .eq('name', companyName)
      .maybeSingle()

    if (!existing) {
      const { data, error } = await supabase
        .from('companies')
        .insert({ name: companyName })
        .select('id, name')
        .single()

      if (error) {
        console.log(`✗ Error creating ${companyName}: ${error.message}`)
      } else {
        console.log(`✓ Created: ${data.name}`)
      }
    } else {
      console.log(`  Already exists: ${existing.name}`)
    }
  }

  // Now link projects
  const mappings = [
    { project: 'Greenbushes Lithium Mine', company: 'Talison Lithium (Tianqi/IGO/Albemarle JV)' },
    { project: 'Pilgangoora Lithium-Tantalum Project', company: 'Pilbara Minerals' },
    { project: 'Mount Marion Lithium Project', company: 'Mineral Resources' },
    { project: 'Salar de Atacama Lithium Operations', company: 'Albemarle Corporation' },
    { project: 'Salar del Carmen Lithium Operations', company: 'Sociedad Quimica y Minera (SQM)' },
    { project: 'Goulamina Lithium Project', company: 'Ganfeng Lithium' },
    { project: 'Norilsk-Talnakh Nickel Complex', company: 'Nornickel' },
    { project: 'Weda Bay Nickel Project', company: 'Tsingshan Holding Group' },
    { project: 'Sorowako Nickel Mine', company: 'Vale SA' },
    { project: 'Tenke Fungurume Copper-Cobalt Mine', company: 'CMOC Group' },
    { project: 'Bayan Obo Iron-REE Mine', company: 'Tsingshan Holding Group' },
    { project: 'Mountain Pass Rare Earth Mine', company: 'MP Materials' },
    { project: 'Mount Weld Rare Earth Mine', company: 'Lynas Rare Earths' }
  ]

  console.log('\nLinking projects to companies...')
  for (const mapping of mappings) {
    const { data: company } = await supabase
      .from('companies')
      .select('id')
      .eq('name', mapping.company)
      .single()

    if (!company) {
      console.log(`✗ Company not found: ${mapping.company}`)
      continue
    }

    const { error } = await supabase
      .from('projects')
      .update({ company_id: company.id })
      .eq('name', mapping.project)

    if (error) {
      console.log(`✗ ${mapping.project}: ${error.message}`)
    } else {
      console.log(`✓ ${mapping.project} → ${mapping.company}`)
    }
  }

  console.log('\n✅ Done! Company names should now appear in the UI.')
}

createMissingCompanies()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error)
    process.exit(1)
  })
