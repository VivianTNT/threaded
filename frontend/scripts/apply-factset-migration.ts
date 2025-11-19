#!/usr/bin/env node

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'
import fs from 'fs'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function runMigration() {
  console.log('📊 Applying FactSet documents migration...\n')

  const migrationPath = path.join(__dirname, '..', 'supabase', 'migrations', '20251020_create_factset_documents.sql')
  const sql = fs.readFileSync(migrationPath, 'utf-8')

  const { error } = await supabase.rpc('exec_sql', { sql_query: sql }).catch(async () => {
    // Fallback: execute statements one by one
    const statements = sql
      .split(';')
      .map(s => s.trim())
      .filter(s => s && !s.startsWith('--'))

    for (const statement of statements) {
      const { error: stmtError } = await supabase.rpc('exec_sql', { sql_query: statement + ';' })
      if (stmtError) {
        console.error(`Error:`, stmtError.message)
      }
    }

    return { error: null }
  })

  if (error) {
    console.error('❌ Migration failed:', error.message)
    process.exit(1)
  }

  console.log('✅ Migration completed successfully!')
}

runMigration()
