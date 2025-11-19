#!/usr/bin/env npx tsx

// Test script for the simplified Cmd+F style highlighting

async function testHighlighting() {
  const testPdfUrl = 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/FM/21aa219db185f147ad46b0554b8352e3-filer.pdf'

  console.log('🧪 Testing simplified auto-highlight extraction...')
  console.log('📄 PDF URL:', testPdfUrl)
  console.log('---')

  try {
    // Call the extraction API
    const response = await fetch('http://localhost:3000/api/pdf/extract-highlights', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pdfUrl: testPdfUrl,
        projectId: 'test-project-id',
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    console.log('✅ Extraction completed!')
    console.log('📊 Results:')
    console.log('  - Success:', result.success)
    console.log('  - Highlights found:', result.highlights?.length || 0)
    console.log('  - Number of pages:', result.numPages)
    console.log('  - Relevant pages analyzed:', result.relevantPages?.length || 0)

    if (result.highlights && result.highlights.length > 0) {
      console.log('\n📍 Highlights with coordinates:')
      result.highlights.forEach((h: any) => {
        const hasCoords = h.highlightAreas && h.highlightAreas.length > 0
        console.log(`  - ${h.dataType}: ${hasCoords ? '✅ Has coordinates' : '❌ No coordinates'}`)
        if (hasCoords) {
          const area = h.highlightAreas[0]
          console.log(`    Page ${area.pageIndex + 1}: [${area.left.toFixed(1)}%, ${area.top.toFixed(1)}%, ${area.width.toFixed(1)}%×${area.height.toFixed(1)}%]`)
        }
      })
    }

    if (result.extractedData) {
      console.log('\n💰 Extracted financial data:')
      const data = result.extractedData
      if (data.npv) console.log(`  - NPV: $${data.npv.value}M`)
      if (data.irr) console.log(`  - IRR: ${data.irr.value}%`)
      if (data.capex) console.log(`  - CAPEX: $${data.capex.value}M`)
    }

  } catch (error) {
    console.error('❌ Error:', error)
  }
}

// Run the test
testHighlighting()