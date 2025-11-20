#!/usr/bin/env node

/**
 * Setup FactSet Storage and Database Tables
 *
 * Creates:
 * 1. Supabase storage bucket for PDFs
 * 2. factset_documents table for metadata
 * 3. Updates projects table with document columns
 */

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function setupStorage() {
  console.log('📦 Setting up Supabase storage bucket...\n')

  // Check if bucket exists
  const { data: buckets, error: listError } = await supabase.storage.listBuckets()

  if (listError) {
    console.error('❌ Error listing buckets:', listError.message)
    return false
  }

  const bucketExists = buckets?.some(b => b.name === 'factset-documents')

  if (!bucketExists) {
    // Create bucket
    const { error: createError } = await supabase.storage.createBucket('factset-documents', {
      public: true,
      fileSizeLimit: 52428800 // 50MB
    })

    if (createError) {
      console.error('❌ Error creating bucket:', createError.message)
      return false
    }

    console.log('✅ Created bucket: factset-documents')
  } else {
    console.log('✅ Bucket already exists: factset-documents')
  }

  return true
}

async function setupTables() {
  console.log('\n📊 Setting up database tables...\n')

  // Create factset_documents table
  const createTableSQL = `
    CREATE TABLE IF NOT EXISTS factset_documents (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
      document_id TEXT UNIQUE NOT NULL,
      project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
      headline TEXT NOT NULL,
      filing_date TIMESTAMP WITH TIME ZONE NOT NULL,
      form_type TEXT,
      storage_path TEXT NOT NULL,
      public_url TEXT NOT NULL,
      file_size INTEGER,
      created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_factset_documents_project_id ON factset_documents(project_id);
    CREATE INDEX IF NOT EXISTS idx_factset_documents_filing_date ON factset_documents(filing_date);
  `

  // Execute via raw query (note: this requires admin access)
  // Since we can't execute raw SQL directly, we'll check if table exists another way

  // Check if table exists by trying to query it
  const { error: tableCheck } = await supabase
    .from('factset_documents')
    .select('id')
    .limit(1)

  if (tableCheck) {
    console.log('⚠️  Table factset_documents needs to be created manually')
    console.log('\nRun this SQL in your Supabase SQL Editor:')
    console.log('━'.repeat(70))
    console.log(createTableSQL)
    console.log('━'.repeat(70))
    return false
  }

  console.log('✅ Table factset_documents exists')
  return true
}

async function main() {
  console.log('='.repeat(70))
  console.log('FACTSET STORAGE SETUP')
  console.log('='.repeat(70))

  const storageOk = await setupStorage()
  const tablesOk = await setupTables()

  console.log('\n' + '='.repeat(70))
  console.log('SETUP COMPLETE')
  console.log('='.repeat(70))

  if (storageOk && tablesOk) {
    console.log('\n✅ All setup complete! Ready to upload documents.')
  } else {
    console.log('\n⚠️  Some manual steps required. See messages above.')
  }
}

main().catch(console.error)
