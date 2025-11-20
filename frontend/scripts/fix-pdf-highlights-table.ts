#!/usr/bin/env node

import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!
const SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!

async function runSQL(sql: string) {
  const response = await fetch(`${SUPABASE_URL}/rest/v1/rpc/exec_sql`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'apikey': SERVICE_ROLE_KEY,
      'Authorization': `Bearer ${SERVICE_ROLE_KEY}`,
    },
    body: JSON.stringify({ query: sql }),
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`SQL execution failed: ${response.status} - ${text}`)
  }

  return response.json()
}

async function fixTable() {
  console.log('🔧 Fixing pdf_highlights table...\n')

  try {
    // Method 1: Try dropping and recreating the table with correct schema
    console.log('Attempting to add UNIQUE constraint...')

    const sql = `
      -- Drop existing index if it exists
      DROP INDEX IF EXISTS idx_pdf_highlights_document_url;

      -- Drop constraint if it exists
      ALTER TABLE pdf_highlights DROP CONSTRAINT IF EXISTS pdf_highlights_document_url_key;

      -- Add unique constraint
      ALTER TABLE pdf_highlights ADD CONSTRAINT pdf_highlights_document_url_key UNIQUE (document_url);
    `

    await runSQL(sql)

    console.log('✅ Successfully added UNIQUE constraint!')
    console.log('   Table: pdf_highlights')
    console.log('   Constraint: UNIQUE(document_url)')
  } catch (error: any) {
    if (error.message.includes('does not exist') || error.message.includes('exec_sql')) {
      console.log('\n⚠️  RPC function not available. Using alternative approach...\n')
      console.log('Please run this SQL manually in Supabase SQL Editor:')
      console.log('=' .repeat(60))
      console.log(`
DROP INDEX IF EXISTS idx_pdf_highlights_document_url;
ALTER TABLE pdf_highlights DROP CONSTRAINT IF EXISTS pdf_highlights_document_url_key;
ALTER TABLE pdf_highlights ADD CONSTRAINT pdf_highlights_document_url_key UNIQUE (document_url);
`)
      console.log('='.repeat(60))
    } else {
      console.error('❌ Error:', error.message)
    }
  }
}

fixTable().catch(console.error)
