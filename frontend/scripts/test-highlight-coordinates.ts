import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function testHighlightCoordinates() {
  console.log('🧪 Testing PDF highlight coordinates extraction...\n')

  // Get a sample PDF URL from existing factset_documents
  const { data: doc, error: docError } = await supabase
    .from('factset_documents')
    .select('id, project_id, public_url, headline')
    .not('public_url', 'is', null)
    .limit(1)
    .single()

  if (docError || !doc) {
    console.error('❌ No documents found:', docError)
    return
  }

  // Get project details
  const { data: project, error: projectError } = await supabase
    .from('projects')
    .select('id, name')
    .eq('id', doc.project_id)
    .single()

  if (projectError || !project) {
    console.error('❌ No projects with documents found:', projectError)
    return
  }

  const pdfUrl = doc.public_url
  console.log(`📄 Testing with project: ${project.name || 'Unknown'}`)
  console.log(`   Document: ${doc.headline}`)
  console.log(`   PDF: ${pdfUrl}\n`)

  // Delete any existing highlights for this PDF
  console.log('🗑️  Deleting existing highlights...')
  const { error: deleteError } = await supabase
    .from('pdf_highlights')
    .delete()
    .eq('document_url', pdfUrl)

  if (deleteError) {
    console.warn('⚠️  Error deleting highlights:', deleteError)
  } else {
    console.log('✅ Old highlights deleted\n')
  }

  // Trigger fresh extraction
  console.log('🤖 Triggering fresh extraction with coordinate calculation...')
  const response = await fetch('http://localhost:3001/api/pdf/extract-highlights', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      pdfUrl,
      projectId: project.id,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    console.error('❌ Extraction failed:', errorText)
    return
  }

  const result = await response.json()

  console.log('\n✅ Extraction complete!\n')
  console.log(`📊 Results:`)
  console.log(`  - Total pages: ${result.numPages}`)
  console.log(`  - Highlights extracted: ${result.highlights?.length || 0}\n`)

  if (result.highlights && result.highlights.length > 0) {
    console.log('📌 Highlights with coordinates:\n')
    result.highlights.forEach((highlight: any, idx: number) => {
      const hasCoords = highlight.highlightAreas && highlight.highlightAreas.length > 0
      const coords = hasCoords ? highlight.highlightAreas[0] : null

      console.log(`  ${idx + 1}. ${highlight.dataType?.toUpperCase() || 'UNKNOWN'}`)
      console.log(`     Page: ${highlight.page}`)
      console.log(`     Value: ${highlight.value || 'N/A'}`)

      if (coords) {
        console.log(`     ✅ Coordinates found:`)
        console.log(`        - Page Index: ${coords.pageIndex}`)
        console.log(`        - Left: ${coords.left?.toFixed(2) || 0}%`)
        console.log(`        - Top: ${coords.top?.toFixed(2) || 0}%`)
        console.log(`        - Width: ${coords.width?.toFixed(2) || 0}%`)
        console.log(`        - Height: ${coords.height?.toFixed(2) || 0}%`)
      } else {
        console.log(`     ❌ No coordinates found`)
      }
      console.log()
    })

    // Verify in database
    const { data: savedHighlight } = await supabase
      .from('pdf_highlights')
      .select('*')
      .eq('document_url', pdfUrl)
      .single()

    if (savedHighlight) {
      console.log('✅ Verified highlights saved to database with coordinates')

      const dbHighlights = savedHighlight.highlight_data?.highlights || []
      const withCoords = dbHighlights.filter(
        (h: any) => h.highlightAreas && h.highlightAreas.length > 0
      ).length

      console.log(`   ${withCoords}/${dbHighlights.length} highlights have coordinates`)
    }
  } else {
    console.log('⚠️  No highlights extracted')
  }

  console.log('\n✅ Test complete!')
  console.log('\n💡 Next steps:')
  console.log('   1. Open the PDF viewer in the app')
  console.log('   2. Click on an extracted data item')
  console.log('   3. It should jump to the exact highlighted text on the page')
}

testHighlightCoordinates().catch(console.error)
