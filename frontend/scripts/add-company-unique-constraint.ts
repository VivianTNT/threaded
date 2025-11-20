import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
)

async function addConstraint() {
  console.log('Adding unique constraint to companies table...\n')

  // First, remove any duplicate entries
  const { data: duplicates, error: dupError } = await supabase
    .rpc('exec_sql', {
      sql_query: `
        DELETE FROM companies a USING companies b
        WHERE a.id > b.id
        AND a.ticker = b.ticker
        AND a.exchange = b.exchange;
      `
    })

  if (dupError) {
    console.log('Note: Could not clean duplicates (may not exist):', dupError.message)
  } else {
    console.log('✅ Cleaned duplicate entries')
  }

  // Add unique constraint
  const { error } = await supabase
    .rpc('exec_sql', {
      sql_query: `
        ALTER TABLE companies
        ADD CONSTRAINT companies_ticker_exchange_unique
        UNIQUE (ticker, exchange);
      `
    })

  if (error) {
    if (error.message.includes('already exists')) {
      console.log('✅ Unique constraint already exists')
    } else {
      console.error('❌ Error adding constraint:', error)
    }
  } else {
    console.log('✅ Successfully added unique constraint')
  }
}

addConstraint()
