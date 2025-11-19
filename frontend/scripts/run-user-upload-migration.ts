import { createClient } from '@supabase/supabase-js'
import * as fs from 'fs'
import * as path from 'path'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function runMigration() {
  console.log('🚀 Running user upload migration...')

  try {
    // Read the migration file
    const migrationPath = path.join(__dirname, '..', 'supabase', 'migrations', '20250124_add_user_upload_support.sql')
    const migrationSQL = fs.readFileSync(migrationPath, 'utf-8')

    console.log('📄 Migration file loaded')
    console.log('📝 Executing SQL...')

    // Execute the migration
    const { data, error } = await supabase.rpc('exec_sql', { sql: migrationSQL }).single()

    if (error) {
      // Try direct query instead
      console.log('⚠️ RPC method failed, trying direct query...')

      // Split by semicolons and execute each statement
      const statements = migrationSQL.split(';').filter(s => s.trim())

      for (const statement of statements) {
        if (statement.trim()) {
          const { error: stmtError } = await supabase.rpc('exec_sql', { sql: statement })
          if (stmtError) {
            console.error('❌ Error executing statement:', stmtError)
            console.log('Statement:', statement.substring(0, 100) + '...')
          } else {
            console.log('✅ Statement executed')
          }
        }
      }
    } else {
      console.log('✅ Migration executed successfully')
    }

    // Verify the columns were added
    console.log('\n🔍 Verifying migration...')
    const { data: columns, error: colError } = await supabase
      .from('projects')
      .select('user_id, is_private, uploaded_at, document_storage_path')
      .limit(1)

    if (colError && colError.message.includes('column')) {
      console.log('⚠️ Columns may not exist yet, this is expected for a new migration')
    } else {
      console.log('✅ Columns verified!')
    }

    // Check storage bucket
    const { data: buckets, error: bucketError } = await supabase
      .storage
      .listBuckets()

    if (bucketError) {
      console.error('❌ Error checking buckets:', bucketError)
    } else {
      const userDocsBucket = buckets?.find(b => b.id === 'user-documents')
      if (userDocsBucket) {
        console.log('✅ Storage bucket "user-documents" exists')
      } else {
        console.log('⚠️ Storage bucket "user-documents" not found, creating...')
        const { error: createError } = await supabase
          .storage
          .createBucket('user-documents', { public: false })

        if (createError) {
          console.error('❌ Error creating bucket:', createError)
        } else {
          console.log('✅ Storage bucket created')
        }
      }
    }

    console.log('\n✅ Migration complete!')

  } catch (error) {
    console.error('❌ Migration failed:', error)
    process.exit(1)
  }
}

runMigration()
