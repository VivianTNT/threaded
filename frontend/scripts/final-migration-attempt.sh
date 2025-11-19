#!/bin/bash

# Supabase connection details
PROJECT_REF="dfxauievbyqwcynwtvib"
SUPABASE_URL="https://dfxauievbyqwcynwtvib.supabase.co"
SERVICE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmeGF1aWV2Ynlxd2N5bnd0dmliIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0Nzg0ODI2MSwiZXhwIjoyMDYzNDI0MjYxfQ.Gs2NX-UUKtXvW3a9_h49ATSDzvpsfJdja6tt1bCkyjc"

echo "🚀 Attempting to run migration using psql connection string..."
echo ""

# The migration SQL
MIGRATION_SQL=$(cat <<'EOF'
-- Add columns
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

-- Policies
DROP POLICY IF EXISTS "Public projects are viewable by everyone" ON projects;
DROP POLICY IF EXISTS "Users can view their own private projects" ON projects;
DROP POLICY IF EXISTS "Users can insert their own private projects" ON projects;
DROP POLICY IF EXISTS "Users can update their own private projects" ON projects;
DROP POLICY IF EXISTS "Users can delete their own private projects" ON projects;

CREATE POLICY "Public projects are viewable by everyone" ON projects FOR SELECT USING (is_private = FALSE OR is_private IS NULL);
CREATE POLICY "Users can view their own private projects" ON projects FOR SELECT USING (is_private = TRUE AND user_id = auth.uid());
CREATE POLICY "Users can insert their own private projects" ON projects FOR INSERT WITH CHECK (is_private = TRUE AND user_id = auth.uid());
CREATE POLICY "Users can update their own private projects" ON projects FOR UPDATE USING (is_private = TRUE AND user_id = auth.uid());
CREATE POLICY "Users can delete their own private projects" ON projects FOR DELETE USING (is_private = TRUE AND user_id = auth.uid());

-- Storage
INSERT INTO storage.buckets (id, name, public) VALUES ('user-documents', 'user-documents', false) ON CONFLICT (id) DO NOTHING;

DROP POLICY IF EXISTS "Users can upload their own documents" ON storage.objects;
DROP POLICY IF EXISTS "Users can view their own documents" ON storage.objects;
DROP POLICY IF EXISTS "Users can delete their own documents" ON storage.objects;

CREATE POLICY "Users can upload their own documents" ON storage.objects FOR INSERT WITH CHECK (bucket_id = 'user-documents' AND auth.role() = 'authenticated' AND (storage.foldername(name))[1] = auth.uid()::text);
CREATE POLICY "Users can view their own documents" ON storage.objects FOR SELECT USING (bucket_id = 'user-documents' AND auth.role() = 'authenticated' AND (storage.foldername(name))[1] = auth.uid()::text);
CREATE POLICY "Users can delete their own documents" ON storage.objects FOR DELETE USING (bucket_id = 'user-documents' AND auth.role() = 'authenticated' AND (storage.foldername(name))[1] = auth.uid()::text);
EOF
)

echo "⚠️  Direct SQL execution not possible via API"
echo ""
echo "📋 PLEASE RUN THIS SQL IN SUPABASE DASHBOARD:"
echo "─────────────────────────────────────────────"
echo ""
echo "1. Go to: https://supabase.com/dashboard/project/$PROJECT_REF/sql/new"
echo "2. Copy and paste the following SQL:"
echo ""
echo "$MIGRATION_SQL"
echo ""
echo "─────────────────────────────────────────────"
echo ""
echo "Or run this file: supabase/migrations/20250124_add_user_upload_support.sql"
echo ""
