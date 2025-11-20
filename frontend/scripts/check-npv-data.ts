import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://dfxauievbyqwcynwtvib.supabase.co'
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmeGF1aWV2Ynlxd2N5bnd0dmliIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NDgyNjEsImV4cCI6MjA2MzQyNDI2MX0.7eJHAHhftweScDfEXqK9mxZRApvm-f2XRETFIxM6ELY'

const supabase = createClient(supabaseUrl, supabaseKey)

async function checkNPVData() {
  console.log('Checking NPV data in database...\n')

  // Check total projects with NPV data
  const { data: allProjects, count: totalCount } = await supabase
    .from('projects')
    .select('name, npv, irr, capex', { count: 'exact' })
    .not('npv', 'is', null)
    .order('npv', { ascending: false })
    .limit(10)

  console.log(`Total projects with NPV data: ${totalCount}\n`)

  if (allProjects && allProjects.length > 0) {
    console.log('Top 10 projects by NPV:')
    allProjects.forEach((p, i) => {
      console.log(`${i + 1}. ${p.name}`)
      console.log(`   NPV: $${p.npv}M, IRR: ${p.irr}%, CAPEX: $${p.capex}M`)
    })
  }

  console.log('\n---\n')

  // Check specific projects
  const projectNames = ['Mountain Pass', 'Hammerdown', 'Queensway']

  for (const projectName of projectNames) {
    const { data } = await supabase
      .from('projects')
      .select('name, location, stage, npv, irr, capex, commodities')
      .ilike('name', `%${projectName}%`)
      .limit(1)
      .single()

    if (data) {
      console.log(`${data.name}:`)
      console.log(`  Location: ${data.location}`)
      console.log(`  Stage: ${data.stage}`)
      console.log(`  Commodities: ${data.commodities?.join(', ')}`)
      console.log(`  NPV: ${data.npv !== null ? `$${data.npv}M` : 'N/A'}`)
      console.log(`  IRR: ${data.irr !== null ? `${data.irr}%` : 'N/A'}`)
      console.log(`  CAPEX: ${data.capex !== null ? `$${data.capex}M` : 'N/A'}`)
      console.log('')
    } else {
      console.log(`${projectName}: Not found`)
      console.log('')
    }
  }
}

checkNPVData().catch(console.error)
