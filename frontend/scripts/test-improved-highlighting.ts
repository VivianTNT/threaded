#!/usr/bin/env npx tsx

// Test the improved text matching for highlighting

async function testImprovedHighlighting() {
  const testPdfUrl = 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/FM/21aa219db185f147ad46b0554b8352e3-filer.pdf'

  console.log('🧪 Testing improved text matching for highlighting...')
  console.log('📄 PDF URL:', testPdfUrl)
  console.log('---')

  // Test case from the user - this should match
  const testText = "The updated Measured and Indicated Mineral Resource estimate (as at the end of December 2023) now stands at 1,160.9 Mt at an average grade of 0.61 %TCu (excluding stockpiles)."

  console.log('🔍 Test text to find:')
  console.log(`   "${testText}"`)
  console.log('---')

  try {
    // Call the extraction API with a manual highlight to test
    const response = await fetch('http://localhost:3000/api/pdf/extract-highlights', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pdfUrl: testPdfUrl,
        projectId: null, // No project update needed for test
      }),
    })

    if (!response.ok) {
      const error = await response.text()
      throw new Error(`HTTP error! status: ${response.status}, error: ${error}`)
    }

    const result = await response.json()

    console.log('\n✅ Extraction completed!')
    console.log('📊 Results:')
    console.log('  - Success:', result.success)
    console.log('  - Total highlights:', result.highlights?.length || 0)
    console.log('  - Total pages:', result.numPages)

    if (result.highlights && result.highlights.length > 0) {
      console.log('\n📍 Highlight Coordinate Results:')

      result.highlights.forEach((highlight: any) => {
        const hasCoords = highlight.highlightAreas && highlight.highlightAreas.length > 0
        const status = hasCoords ? '✅' : '❌'

        console.log(`\n${status} ${highlight.dataType?.toUpperCase() || 'UNKNOWN'}`)
        console.log(`   Page: ${highlight.page}`)
        console.log(`   Text: "${highlight.content?.substring(0, 60)}..."`)

        if (hasCoords) {
          const area = highlight.highlightAreas[0]
          console.log(`   📐 Coordinates: [${area.left.toFixed(1)}%, ${area.top.toFixed(1)}%] - ${area.width.toFixed(1)}% × ${area.height.toFixed(1)}%`)
        } else {
          console.log(`   ⚠️  No coordinates found - text matching failed`)
        }
      })

      // Summary
      const withCoords = result.highlights.filter((h: any) => h.highlightAreas?.length > 0).length
      const total = result.highlights.length

      console.log('\n📊 Summary:')
      console.log(`   Successfully highlighted: ${withCoords}/${total} (${((withCoords/total)*100).toFixed(0)}%)`)

      if (withCoords < total) {
        console.log('\n💡 Text matching improved with:')
        console.log('   - Space normalization (handles multiple spaces)')
        console.log('   - Partial matching (first 30 chars)')
        console.log('   - Key phrase extraction (numbers, percentages)')
      }
    }

  } catch (error) {
    console.error('\n❌ Error:', error)
  }
}

// Run the test
testImprovedHighlighting()