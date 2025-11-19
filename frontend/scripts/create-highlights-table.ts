import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function createHighlightsTable() {
  console.log('Creating pdf_highlights table...')

  const { error } = await supabase.rpc('exec_sql', {
    sql: `
      CREATE TABLE IF NOT EXISTS pdf_highlights (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_url TEXT NOT NULL,
        project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
        highlight_data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      );

      CREATE INDEX IF NOT EXISTS idx_pdf_highlights_document_url ON pdf_highlights(document_url);
      CREATE INDEX IF NOT EXISTS idx_pdf_highlights_project_id ON pdf_highlights(project_id);
    `
  })

  if (error) {
    console.error('Error creating table:', error)

    // Try direct query instead
    const { error: createError } = await supabase.from('pdf_highlights').select('id').limit(1)

    if (createError && createError.message.includes('does not exist')) {
      console.log('Table does not exist, attempting direct creation...')
      console.log('Please run this SQL manually in Supabase SQL Editor:')
      console.log(`
CREATE TABLE IF NOT EXISTS pdf_highlights (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_url TEXT NOT NULL,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  highlight_data JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_highlights_document_url ON pdf_highlights(document_url);
CREATE INDEX IF NOT EXISTS idx_pdf_highlights_project_id ON pdf_highlights(project_id);
      `)
    }
  } else {
    console.log('✅ Table created successfully!')
  }
}

createHighlightsTable()
