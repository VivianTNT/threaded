import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function verifyCompanyData() {
  console.log('Verifying company data for new projects...')

  // Get the newly added projects
  const projectNames = [
    'Greenbushes Lithium Mine',
    'Pilgangoora Lithium-Tantalum Project',
    'Mount Marion Lithium Project',
    'Escondida Copper Mine',
    'Tenke Fungurume Copper-Cobalt Mine'
  ]

  for (const name of projectNames) {
    const { data: project } = await supabase
      .from('projects')
      .select('id, name, company_id')
      .eq('name', name)
      .single()

    if (project) {
      if (project.company_id) {
        const { data: company } = await supabase
          .from('companies')
          .select('id, name')
          .eq('id', project.company_id)
          .single()

        if (company) {
          console.log(`✓ ${project.name} -> ${company.name}`)
        } else {
          console.log(`✗ ${project.name} -> Company ID ${project.company_id} NOT FOUND`)
        }
      } else {
        console.log(`✗ ${project.name} -> NO COMPANY_ID`)
      }
    }
  }
}

verifyCompanyData()
  .then(() => {
    console.log('\nVerification complete!')
    console.log('If companies are properly linked, they should display in the UI.')
    console.log('Try refreshing the browser if they still show as "unknown".')
    process.exit(0)
  })
  .catch((error) => {
    console.error('Error:', error)
    process.exit(1)
  })
