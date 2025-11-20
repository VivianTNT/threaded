import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('📊 Checking PDF Highlights...\n')

  // Check Macassa highlights
  const { data: highlights, error: hlError } = await supabase
    .from('pdf_highlights')
    .select('id, project_id, data_type, page, value, quote')
    .eq('project_id', '2bff5781-1928-4f8f-b672-e27da94efd5a')
    .limit(10)

  if (hlError) {
    console.error('Error fetching highlights:', hlError)
  } else {
    console.log(`Found ${highlights?.length || 0} highlights for Macassa:`)
    highlights?.forEach((hl: any) => {
      console.log(`  - ${hl.data_type}: ${hl.value} (Page: ${hl.page || 'NULL'})`)
      console.log(`    "${hl.quote?.substring(0, 80)}..."`)
    })
  }

  console.log('\n📊 Checking Companies Table...\n')

  // Check companies table structure
  const { data: companies, error: compError } = await supabase
    .from('companies')
    .select('*')
    .limit(5)

  if (compError) {
    console.error('Error fetching companies:', compError)
  } else {
    console.log(`Found ${companies?.length || 0} companies:`)
    if (companies && companies.length > 0) {
      console.log('Columns:', Object.keys(companies[0]))
      companies.forEach((comp: any) => {
        console.log(`  - ${comp.name || comp.company_name} (${comp.ticker}): Market Cap: ${comp.market_cap || 'NULL'}`)
      })
    }
  }

  // Check if market_cap column exists
  console.log('\n📊 Checking table schema...\n')
  const { data: schema, error: schemaError } = await supabase.rpc('exec_sql' as any, {
    sql: `
      SELECT column_name, data_type
      FROM information_schema.columns
      WHERE table_name = 'companies'
      AND table_schema = 'public'
      ORDER BY ordinal_position
    `
  }).catch(() => ({ data: null, error: null }))

  if (!schema) {
    console.log('Using direct SQL query instead...')
    // Alternative approach - just try selecting specific columns
    const { data: testData, error: testError } = await supabase
      .from('companies')
      .select('id, name, ticker, market_cap, sector, headquarters')
      .limit(1)

    if (testError) {
      console.log('Columns that might be missing:', testError.message)
    } else {
      console.log('✅ All expected columns exist')
    }
  }
}

main()
