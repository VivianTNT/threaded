import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function fixAllProjectCompanies() {
  console.log('Creating missing companies and linking all projects...')

  // All companies needed
  const companies = [
    { name: 'Talison Lithium (Tianqi/IGO/Albemarle JV)', country: 'Australia', stock_symbol: 'IGO.AX' },
    { name: 'Pilbara Minerals', country: 'Australia', stock_symbol: 'PLS.AX' },
    { name: 'Mineral Resources', country: 'Australia', stock_symbol: 'MIN.AX' },
    { name: 'Albemarle Corporation', country: 'United States', stock_symbol: 'ALB' },
    { name: 'Sociedad Quimica y Minera (SQM)', country: 'Chile', stock_symbol: 'SQM' },
    { name: 'Ganfeng Lithium', country: 'China', stock_symbol: '002460.SZ' },
    { name: 'Nornickel', country: 'Russia', stock_symbol: 'GMKN.ME' },
    { name: 'Tsingshan Holding Group', country: 'China', stock_symbol: null },
    { name: 'Vale SA', country: 'Brazil', stock_symbol: 'VALE' },
    { name: 'CMOC Group', country: 'China', stock_symbol: '603993.SS' },
    { name: 'MP Materials', country: 'United States', stock_symbol: 'MP' },
    { name: 'Lynas Rare Earths', country: 'Australia', stock_symbol: 'LYC.AX' }
  ]

  // Create companies if they don't exist
  for (const company of companies) {
    const { data: existing } = await supabase
      .from('companies')
      .select('id, name')
      .eq('name', company.name)
      .maybeSingle()

    if (!existing) {
      const { data, error } = await supabase
        .from('companies')
        .insert({
          name: company.name,
          country: company.country,
          stock_symbol: company.stock_symbol,
          website: null,
          description: null
        })
        .select('id, name')
        .single()

      if (error) {
        console.log(`✗ Error creating ${company.name}: ${error.message}`)
      } else {
        console.log(`✓ Created company: ${data.name}`)
      }
    } else {
      console.log(`  Company already exists: ${existing.name}`)
    }
  }

  // Now link all projects to companies
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
      console.log(`✗ Company still not found: ${mapping.company}`)
      continue
    }

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

  console.log('\n✅ All done! Refresh the browser to see company names.')
}

fixAllProjectCompanies()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('Error:', error)
    process.exit(1)
  })
