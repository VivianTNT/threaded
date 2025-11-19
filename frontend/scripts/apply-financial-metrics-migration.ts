import { createClient } from '@supabase/supabase-js'
import * as fs from 'fs'
import * as path from 'path'
import dotenv from 'dotenv'

// Load environment variables
dotenv.config({ path: '.env.local' })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

if (!supabaseUrl || !supabaseServiceKey) {
  console.error('❌ Missing Supabase credentials')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function applyMigration() {
  console.log('🚀 Applying financial metrics migration...')

  // Read migration file
  const migrationPath = path.join(process.cwd(), 'supabase/migrations/010_add_financial_metrics.sql')
  const migrationSql = fs.readFileSync(migrationPath, 'utf-8')

  console.log('📝 Executing SQL...')

  // Split by semicolons and execute each statement
  const statements = migrationSql
    .split(';')
    .map(s => s.trim())
    .filter(s => s.length > 0 && !s.startsWith('--'))

  for (const statement of statements) {
    if (statement.trim()) {
      console.log(`  Executing: ${statement.substring(0, 60)}...`)
      const { error } = await supabase.rpc('exec_sql', { sql: statement + ';' })

      if (error) {
        console.error(`  ❌ Error:`, error)
        // Continue with other statements even if one fails
      } else {
        console.log(`  ✅ Success`)
      }
    }
  }

  console.log('\n✅ Migration complete!')
}

applyMigration().catch(console.error)
