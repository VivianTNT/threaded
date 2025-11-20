#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function testExtraction() {
  console.log('🧪 Testing PDF Extraction on Updated Projects')
  console.log('=' .repeat(60))

  // Get project with real document URL
  const { data: project, error } = await supabase
    .from('projects')
    .select('*')
    .eq('name', 'Escondida')
    .single()

  if (error || !project) {
    console.log('❌ Failed to fetch project')
    return
  }

  console.log(`\n📄 Testing extraction for: ${project.name}`)
  console.log('-' .repeat(60))
  console.log(`  Document URL: ${project.urls[0].substring(0, 80)}...`)

  try {
    // Call our extraction API
    const response = await fetch('http://localhost:3000/api/pdf/extract-highlights', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pdfUrl: project.urls[0],
        projectId: project.id
      })
    })

    if (response.ok) {
      const result = await response.json()
      console.log(`  ✅ Extraction successful!`)

      if (result.extractedData) {
        const data = result.extractedData
        console.log(`  📊 Extracted metrics:`)
        if (data.npv?.value) console.log(`    • NPV: $${data.npv.value}M`)
        if (data.irr?.value) console.log(`    • IRR: ${data.irr.value}%`)
        if (data.capex?.value) console.log(`    • CAPEX: $${data.capex.value}M`)
      }
    } else {
      console.log(`  ⚠️ Extraction failed`)
    }
  } catch (error) {
    console.log(`  ❌ Error: ${error}`)
  }

  console.log('\n✅ Testing complete!')
}

testExtraction().catch(console.error)
