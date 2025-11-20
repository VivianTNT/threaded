import { createClient } from '@supabase/supabase-js';
import { NextResponse } from 'next/server';
import { readFileSync } from 'fs';
import { join } from 'path';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

export async function POST() {
  const supabase = createClient(supabaseUrl, supabaseServiceKey, {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  });

  const migrations = [
    '006_create_companies_table.sql',
    '007_create_projects_table.sql',
    '008_create_news_table.sql',
  ];

  const results = [];

  for (const migration of migrations) {
    try {
      const sql = readFileSync(
        join(process.cwd(), 'supabase', 'migrations', migration),
        'utf-8'
      );

      // Execute SQL directly using the service role
      const { data, error } = await supabase.rpc('exec', { sql });

      if (error) {
        throw error;
      }

      results.push({ migration, status: 'success' });
    } catch (error: any) {
      results.push({ migration, status: 'error', error: error.message });
    }
  }

  return NextResponse.json({ results });
}
