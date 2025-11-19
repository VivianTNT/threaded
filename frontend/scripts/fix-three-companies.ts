import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function fixThreeCompanies() {
  console.log('🔧 Fixing companies for Mutanda, Grasberg, and Collahuasi...\n')

  // Create the three companies if they don't exist
  const companies = [
    { name: 'Glencore', description: 'Swiss multinational commodity trading and mining company' },
    { name: 'Freeport-McMoRan', description: 'American mining company, major copper producer' },
    { name: 'Anglo American (44%) / Glencore (44%)', description: 'Joint venture for Collahuasi copper mine' }
  ]

  const companyMap: Record<string, string> = {}

  for (const company of companies) {
    // Check if exists
    const { data: existing } = await supabase
      .from('companies')
      .select('id, name')
      .eq('name', company.name)
      .single()

    if (existing) {
      console.log(`✓ Company exists: ${company.name}`)
      companyMap[company.name] = existing.id
    } else {
      const { data: newCompany, error } = await supabase
        .from('companies')
        .insert(company)
        .select()
        .single()

      if (error) {
        console.error(`✗ Failed to create ${company.name}:`, error)
      } else {
        console.log(`✓ Created: ${company.name}`)
        companyMap[company.name] = newCompany.id
      }
    }
  }

  console.log('\n🔗 Linking projects to companies...\n')

  // Link the projects
  const projectMappings = [
    { project: 'Mutanda Copper-Cobalt Mine', company: 'Glencore' },
    { project: 'Grasberg Copper-Gold Mine', company: 'Freeport-McMoRan' },
    { project: 'Collahuasi Copper Mine', company: 'Anglo American (44%) / Glencore (44%)' }
  ]

  for (const mapping of projectMappings) {
    const companyId = companyMap[mapping.company]

    const { error } = await supabase
      .from('projects')
      .update({ company_id: companyId })
      .eq('name', mapping.project)

    if (error) {
      console.error(`✗ Failed to link ${mapping.project}:`, error)
    } else {
      console.log(`✓ ${mapping.project} → ${mapping.company}`)
    }
  }

  // Verify
  console.log('\n✅ Verification:\n')

  const { data: projects } = await supabase
    .from('projects')
    .select('name, companies(name)')
    .in('name', ['Mutanda Copper-Cobalt Mine', 'Grasberg Copper-Gold Mine', 'Collahuasi Copper Mine'])

  projects?.forEach(p => {
    console.log(`${p.name}: ${(p as any).companies?.name || 'NO COMPANY'}`)
  })
}

fixThreeCompanies()
