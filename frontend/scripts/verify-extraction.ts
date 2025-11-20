#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('🔍 Verifying extracted data in database...\n')

  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, npv, irr, capex, resource, reserve, location, commodities, description')
    .not('urls', 'is', null)
    .order('name')

  if (error) {
    console.error('❌ Error fetching projects:', error)
    return
  }

  console.log(`📊 Found ${projects?.length || 0} projects with PDFs\n`)

  let withNPV = 0
  let withIRR = 0
  let withCAPEX = 0
  let withResource = 0
  let withReserve = 0
  let withLocation = 0
  let withCommodities = 0
  let withDescription = 0

  console.log('─'.repeat(100))
  console.log('PROJECT NAME'.padEnd(40) + 'NPV'.padEnd(10) + 'IRR'.padEnd(10) + 'CAPEX'.padEnd(15) + 'RES/REV')
  console.log('─'.repeat(100))

  for (const project of projects || []) {
    const hasNPV = project.npv !== null
    const hasIRR = project.irr !== null
    const hasCAPEX = project.capex !== null
    const hasResource = project.resource !== null
    const hasReserve = project.reserve !== null

    if (hasNPV) withNPV++
    if (hasIRR) withIRR++
    if (hasCAPEX) withCAPEX++
    if (hasResource) withResource++
    if (hasReserve) withReserve++
    if (project.location) withLocation++
    if (project.commodities && project.commodities.length > 0) withCommodities++
    if (project.description) withDescription++

    const name = project.name.substring(0, 38).padEnd(40)
    const npv = hasNPV ? `$${project.npv}M`.padEnd(10) : 'N/A'.padEnd(10)
    const irr = hasIRR ? `${project.irr}%`.padEnd(10) : 'N/A'.padEnd(10)
    const capex = hasCAPEX ? `$${project.capex}M`.padEnd(15) : 'N/A'.padEnd(15)
    const resrev = (hasResource ? '✓' : '✗') + '/' + (hasReserve ? '✓' : '✗')

    console.log(name + npv + irr + capex + resrev)
  }

  console.log('─'.repeat(100))
  console.log('\n📈 EXTRACTION STATISTICS:')
  console.log(`   ✅ NPV: ${withNPV}/${projects?.length || 0} (${((withNPV / (projects?.length || 1)) * 100).toFixed(1)}%)`)
  console.log(`   ✅ IRR: ${withIRR}/${projects?.length || 0} (${((withIRR / (projects?.length || 1)) * 100).toFixed(1)}%)`)
  console.log(`   ✅ CAPEX: ${withCAPEX}/${projects?.length || 0} (${((withCAPEX / (projects?.length || 1)) * 100).toFixed(1)}%)`)
  console.log(`   ✅ Resources: ${withResource}/${projects?.length || 0} (${((withResource / (projects?.length || 1)) * 100).toFixed(1)}%)`)
  console.log(`   ✅ Reserves: ${withReserve}/${projects?.length || 0} (${((withReserve / (projects?.length || 1)) * 100).toFixed(1)}%)`)
  console.log(`   ✅ Location: ${withLocation}/${projects?.length || 0} (${((withLocation / (projects?.length || 1)) * 100).toFixed(1)}%)`)
  console.log(`   ✅ Commodities: ${withCommodities}/${projects?.length || 0} (${((withCommodities / (projects?.length || 1)) * 100).toFixed(1)}%)`)
  console.log(`   ✅ Description: ${withDescription}/${projects?.length || 0} (${((withDescription / (projects?.length || 1)) * 100).toFixed(1)}%)`)

  console.log('\n✨ Data is now available in the frontend!\n')
}

main().catch(console.error)
