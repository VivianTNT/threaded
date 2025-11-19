#!/usr/bin/env node

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function applyMigration() {
  console.log('🔧 Applying unique constraint to pdf_highlights table...\n')

  try {
    // Execute SQL directly using Postgres connection string
    const { Pool } = await import('pg')
    const pool = new Pool({
      connectionString: `postgresql://postgres.dfxauievbyqwcynwtvib:${process.env.SUPABASE_SERVICE_ROLE_KEY!.split('.')[2]}@aws-0-us-west-1.pooler.supabase.com:6543/postgres`
    })

    // Drop existing index
    await pool.query('DROP INDEX IF EXISTS idx_pdf_highlights_document_url;')
    console.log('✅ Dropped existing index')

    // Add unique constraint
    await pool.query(`
      ALTER TABLE pdf_highlights
      DROP CONSTRAINT IF EXISTS pdf_highlights_document_url_key;
    `)

    await pool.query(`
      ALTER TABLE pdf_highlights
      ADD CONSTRAINT pdf_highlights_document_url_key UNIQUE (document_url);
    `)

    console.log('✅ Unique constraint added successfully!')
    console.log('   Table pdf_highlights now has UNIQUE constraint on document_url')

    await pool.end()
  } catch (error: any) {
    console.error('❌ Error:', error.message)
    process.exit(1)
  }
}

applyMigration().catch(console.error)
