import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const MACASSA_PDF_URL = 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/mining-documents/5e283a0b-b2cc-4652-b83e-41341c53eb0a/Macassa/ni43101-1.pdf'
const PROJECT_ID = '2bff5781-1928-4f8f-b672-e27da94efd5a'

async function main() {
  console.log('🗑️  Deleting old highlights for Macassa...')

  const { error: deleteError } = await supabase
    .from('pdf_highlights')
    .delete()
    .eq('document_url', MACASSA_PDF_URL)

  if (deleteError) {
    console.error('Error deleting old highlights:', deleteError)
  } else {
    console.log('✅ Old highlights deleted')
  }

  console.log('\n📄 Triggering new extraction...')
  const response = await fetch('http://localhost:3000/api/pdf/extract-highlights', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      pdfUrl: MACASSA_PDF_URL,
      projectId: PROJECT_ID,
    }),
  })

  if (response.ok) {
    const data = await response.json()
    console.log('✅ Extraction complete!')
    console.log('Highlights created:', data.highlights?.highlights?.length || 0)

    if (data.highlights?.highlights) {
      data.highlights.highlights.forEach((h: any) => {
        console.log(`  - ${h.dataType}: ${h.value} (Page: ${h.page || 'NULL'})`)
      })
    }
  } else {
    console.error('❌ Extraction failed:', response.status, await response.text())
  }
}

main()
