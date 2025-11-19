import { createClient } from '@supabase/supabase-js'
import { readFileSync } from 'fs'
import { join } from 'path'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function applyMigration() {
  console.log('📝 Applying database migration for user upload support...\n')

  // Read the migration SQL file
  const migrationPath = join(__dirname, '..', 'supabase', 'migrations', '20250124_add_user_upload_support.sql')
  const migrationSQL = readFileSync(migrationPath, 'utf-8')

  console.log('Migration SQL to execute:')
  console.log('─'.repeat(60))
  console.log(migrationSQL)
  console.log('─'.repeat(60))
  console.log()

  // Split into individual statements and execute them one by one
  const statements = migrationSQL
    .split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0 && !s.startsWith('--'))

  console.log(`Found ${statements.length} SQL statements to execute\n`)

  for (let i = 0; i < statements.length; i++) {
    const statement = statements[i] + ';'
    console.log(`Executing statement ${i + 1}/${statements.length}...`)
    console.log(statement.substring(0, 100) + '...')

    try {
      const { error } = await supabase.rpc('exec_sql', { sql: statement })

      if (error) {
        console.error(`❌ Error executing statement ${i + 1}:`, error.message)

        // Try alternative approach using REST API
        console.log('   Trying alternative approach...')
        const response = await fetch(`${supabaseUrl}/rest/v1/rpc/exec_sql`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'apikey': supabaseServiceKey,
            'Authorization': `Bearer ${supabaseServiceKey}`
          },
          body: JSON.stringify({ sql: statement })
        })

        if (!response.ok) {
          const errorText = await response.text()
          console.error(`   ❌ Alternative approach also failed:`, errorText)
        } else {
          console.log(`   ✅ Success via alternative approach`)
        }
      } else {
        console.log(`   ✅ Success`)
      }
    } catch (err: any) {
      console.error(`   ❌ Exception:`, err.message)
    }

    console.log()
  }

  console.log('\n🎉 Migration application complete!')
  console.log('   Run check-migration-status.ts to verify\n')
}

applyMigration().catch(console.error)
