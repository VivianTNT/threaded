import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

async function runMigration() {
  console.log('🚀 Running database migration directly...\n')

  const supabase = createClient(supabaseUrl, supabaseServiceKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  })

  // Execute raw SQL using the REST API query endpoint
  const executeSql = async (sql: string) => {
    const response = await fetch(`${supabaseUrl}/rest/v1/rpc/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'apikey': supabaseServiceKey,
        'Authorization': `Bearer ${supabaseServiceKey}`,
      },
      body: JSON.stringify({ query: sql })
    })

    return { ok: response.ok, status: response.status, text: await response.text() }
  }

  // Full migration SQL
  const migrationSql = `
-- Add columns to support user-uploaded private documents
ALTER TABLE projects
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id),
ADD COLUMN IF NOT EXISTS is_private BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS document_storage_path TEXT;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id) WHERE is_private = TRUE;
CREATE INDEX IF NOT EXISTS idx_projects_is_private ON projects(is_private);

-- Enable RLS
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Public projects are viewable by everyone" ON projects;
DROP POLICY IF EXISTS "Users can view their own private projects" ON projects;
DROP POLICY IF EXISTS "Users can insert their own private projects" ON projects;
DROP POLICY IF EXISTS "Users can update their own private projects" ON projects;
DROP POLICY IF EXISTS "Users can delete their own private projects" ON projects;

-- Create policies
CREATE POLICY "Public projects are viewable by everyone"
ON projects FOR SELECT
USING (is_private = FALSE OR is_private IS NULL);

CREATE POLICY "Users can view their own private projects"
ON projects FOR SELECT
USING (is_private = TRUE AND user_id = auth.uid());

CREATE POLICY "Users can insert their own private projects"
ON projects FOR INSERT
WITH CHECK (is_private = TRUE AND user_id = auth.uid());

CREATE POLICY "Users can update their own private projects"
ON projects FOR UPDATE
USING (is_private = TRUE AND user_id = auth.uid());

CREATE POLICY "Users can delete their own private projects"
ON projects FOR DELETE
USING (is_private = TRUE AND user_id = auth.uid());

-- Storage bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('user-documents', 'user-documents', false)
ON CONFLICT (id) DO NOTHING;

-- Drop existing storage policies if they exist
DROP POLICY IF EXISTS "Users can upload their own documents" ON storage.objects;
DROP POLICY IF EXISTS "Users can view their own documents" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete their own documents" ON storage.objects;

-- Storage policies
CREATE POLICY "Users can upload their own documents"
ON storage.objects FOR INSERT
WITH CHECK (
  bucket_id = 'user-documents' AND
  auth.role() = 'authenticated' AND
  (storage.foldername(name))[1] = auth.uid()::text
);

CREATE POLICY "Users can view their own documents"
ON storage.objects FOR SELECT
USING (
  bucket_id = 'user-documents' AND
  auth.role() = 'authenticated' AND
  (storage.foldername(name))[1] = auth.uid()::text
);

CREATE POLICY "Users can delete their own documents"
ON storage.objects FOR DELETE
USING (
  bucket_id = 'user-documents' AND
  auth.role() = 'authenticated' AND
  (storage.foldername(name))[1] = auth.uid()::text
);
`

  console.log('Executing migration SQL via REST API...')
  console.log('─'.repeat(60))

  const result = await executeSql(migrationSql)

  if (!result.ok) {
    console.log(`❌ Status ${result.status}: ${result.text}\n`)
    console.log('The REST API endpoint may not support raw SQL execution.')
    console.log('Please run the migration manually in Supabase Dashboard SQL Editor.\n')
    console.log('SQL file location: /supabase/migrations/20250124_add_user_upload_support.sql\n')
    return
  }

  console.log('✅ Migration SQL executed successfully!\n')

  // Verify the migration
  console.log('Verifying migration...\n')

  const { data: testData, error: testError } = await supabase
    .from('projects')
    .select('user_id, is_private, uploaded_at, document_storage_path')
    .limit(1)

  if (testError) {
    console.log('❌ Verification failed:', testError.message)
  } else {
    console.log('✅ Migration columns exist in projects table')
  }

  const { data: buckets } = await supabase.storage.listBuckets()
  const userDocsBucket = buckets?.find(b => b.name === 'user-documents')

  if (userDocsBucket) {
    console.log('✅ Storage bucket "user-documents" exists')
  } else {
    console.log('❌ Storage bucket "user-documents" not found')
  }

  console.log('\n🎉 Migration complete!\n')
}

runMigration().catch(console.error)
