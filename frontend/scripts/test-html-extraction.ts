import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function testHTMLExtraction() {
  console.log('🧪 Testing HTML extraction for Freeport-McMoRan project...\n')

  // Get the Freeport project
  const { data: project, error } = await supabase
    .from('projects')
    .select('*')
    .eq('name', 'Freeport-McMoRan Copper & Gold - Q3 2025 10-Q')
    .single()

  if (error || !project) {
    console.error('❌ Project not found:', error)
    return
  }

  console.log('✅ Found project:')
  console.log('   ID:', project.id)
  console.log('   Name:', project.name)
  console.log('   Company ID:', project.company_id)
  console.log('   URLs:', project.urls)

  const htmlUrl = project.urls[0]
  console.log('\n📄 HTML Document URL:', htmlUrl)

  // Test fetching the HTML
  console.log('\n🌐 Fetching HTML document...')
  const response = await fetch(htmlUrl)
  if (!response.ok) {
    console.error('❌ Failed to fetch HTML:', response.status)
    return
  }

  const htmlText = await response.text()
  console.log(`✅ HTML fetched: ${htmlText.length} characters`)

  // Check for inline XBRL tags
  const companyNameMatch = htmlText.match(/dei:EntityRegistrantName[^>]*>([^<]+)</)
  const locationMatch = htmlText.match(/dei:EntityAddressCityOrTown[^>]*>([^<]+)</)

  console.log('\n🏢 Company Information in HTML:')
  console.log('   Company Name:', companyNameMatch ? companyNameMatch[1] : 'Not found')
  console.log('   Location:', locationMatch ? locationMatch[1] : 'Not found')

  // Check for financial sections
  const hasTables = htmlText.includes('<table')
  const hasFinancialStatements = /financial statements?/i.test(htmlText)
  const hasBalanceSheet = /balance sheet/i.test(htmlText)
  const hasIncomeStatement = /income statement|statement of operations/i.test(htmlText)

  console.log('\n📊 Document Structure:')
  console.log('   Has tables:', hasTables)
  console.log('   Has financial statements:', hasFinancialStatements)
  console.log('   Has balance sheet:', hasBalanceSheet)
  console.log('   Has income statement:', hasIncomeStatement)

  console.log('\n✅ HTML document is ready for extraction!')
  console.log('\n🎯 Next steps:')
  console.log('   1. Start the dev server: npm run dev')
  console.log('   2. Navigate to the dashboard')
  console.log('   3. Find "Freeport-McMoRan Copper & Gold - Q3 2025 10-Q"')
  console.log('   4. Click on the project to open details')
  console.log('   5. Click "HTML 1" or "Technical Report" button')
  console.log('   6. Click "Extract Key Data" in the viewer')
  console.log('   7. Watch as data is extracted and highlighted!')
}

testHTMLExtraction().catch(console.error)
