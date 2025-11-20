import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

async function applyMigration() {
  console.log('📝 Applying user upload migration...\n')

  // Use fetch to directly execute SQL
  const executeSQL = async (sql: string, description: string) => {
    console.log(`Executing: ${description}`)

    const response = await fetch(`${supabaseUrl}/rest/v1/rpc/exec`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'apikey': supabaseServiceKey,
        'Authorization': `Bearer ${supabaseServiceKey}`,
        'Prefer': 'return=representation'
      },
      body: JSON.stringify({ query: sql })
    })

    if (!response.ok) {
      const error = await response.text()
      console.log(`   ⚠️  ${description} - ${error.substring(0, 100)}`)
    } else {
      console.log(`   ✅ ${description}`)
    }
  }

  // Execute migrations step by step
  try {
    // Step 1: Add columns to projects table
    await executeSQL(`
      ALTER TABLE projects
      ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES auth.users(id),
      ADD COLUMN IF NOT EXISTS is_private BOOLEAN DEFAULT FALSE,
      ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      ADD COLUMN IF NOT EXISTS document_storage_path TEXT;
    `, 'Add user upload columns to projects table')

    // Step 2: Create indexes
    await executeSQL(`
      CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id) WHERE is_private = TRUE;
    `, 'Create index on user_id')

    await executeSQL(`
      CREATE INDEX IF NOT EXISTS idx_projects_is_private ON projects(is_private);
    `, 'Create index on is_private')

    // Step 3: Enable RLS
    await executeSQL(`
      ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
    `, 'Enable RLS on projects table')

    // Step 4: Create RLS policies
    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Public projects are viewable by everyone"
      ON projects FOR SELECT
      USING (is_private = FALSE OR is_private IS NULL);
    `, 'Create public projects policy')

    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Users can view their own private projects"
      ON projects FOR SELECT
      USING (is_private = TRUE AND user_id = auth.uid());
    `, 'Create private projects view policy')

    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Users can insert their own private projects"
      ON projects FOR INSERT
      WITH CHECK (is_private = TRUE AND user_id = auth.uid());
    `, 'Create private projects insert policy')

    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Users can update their own private projects"
      ON projects FOR UPDATE
      USING (is_private = TRUE AND user_id = auth.uid());
    `, 'Create private projects update policy')

    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Users can delete their own private projects"
      ON projects FOR DELETE
      USING (is_private = TRUE AND user_id = auth.uid());
    `, 'Create private projects delete policy')

    // Step 5: Create storage bucket
    console.log('\nCreating storage bucket...')
    const supabase = createClient(supabaseUrl, supabaseServiceKey)

    const { data: existingBucket } = await supabase
      .storage
      .getBucket('user-documents')

    if (!existingBucket) {
      const { error } = await supabase
        .storage
        .createBucket('user-documents', {
          public: false,
          fileSizeLimit: 52428800 // 50MB
        })

      if (error) {
        console.log(`   ⚠️  Storage bucket creation - ${error.message}`)
      } else {
        console.log(`   ✅ Created storage bucket`)
      }
    } else {
      console.log(`   ✅ Storage bucket already exists`)
    }

    // Step 6: Storage RLS policies
    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Users can upload their own documents"
      ON storage.objects FOR INSERT
      WITH CHECK (
        bucket_id = 'user-documents' AND
        auth.role() = 'authenticated' AND
        (storage.foldername(name))[1] = auth.uid()::text
      );
    `, 'Create storage upload policy')

    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Users can view their own documents"
      ON storage.objects FOR SELECT
      USING (
        bucket_id = 'user-documents' AND
        auth.role() = 'authenticated' AND
        (storage.foldername(name))[1] = auth.uid()::text
      );
    `, 'Create storage view policy')

    await executeSQL(`
      CREATE POLICY IF NOT EXISTS "Users can delete their own documents"
      ON storage.objects FOR DELETE
      USING (
        bucket_id = 'user-documents' AND
        auth.role() = 'authenticated' AND
        (storage.foldername(name))[1] = auth.uid()::text
      );
    `, 'Create storage delete policy')

    console.log('\n🎉 Migration complete!')
    console.log('   Run check-migration-status.ts to verify\n')

  } catch (error: any) {
    console.error('\n❌ Migration failed:', error.message)
  }
}

applyMigration().catch(console.error)
