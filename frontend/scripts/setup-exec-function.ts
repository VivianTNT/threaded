/**
 * This script creates a PostgreSQL function that allows executing SQL
 * via the Supabase REST API using the service role key.
 */

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!;

const createFunctionSQL = `
CREATE OR REPLACE FUNCTION exec_sql(query_text text)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
  result jsonb;
BEGIN
  EXECUTE query_text;
  RETURN jsonb_build_object('success', true);
EXCEPTION WHEN OTHERS THEN
  RETURN jsonb_build_object('success', false, 'error', SQLERRM);
END;
$$;
`.trim();

async function setupFunction() {
  console.log('Setting up exec_sql function via REST API...\n');

  // Try to create the function using direct SQL execution through HTTP
  const response = await fetch(`${SUPABASE_URL}/rest/v1/rpc/exec_sql`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'apikey': SERVICE_KEY,
      'Authorization': `Bearer ${SERVICE_KEY}`,
      'Prefer': 'return=representation',
    },
    body: JSON.stringify({
      query_text: createFunctionSQL
    }),
  });

  if (response.ok) {
    const data = await response.json();
    console.log('✅ Function created successfully:', data);
  } else {
    const error = await response.text();
    console.log('Response status:', response.status);
    console.log('Response:', error);
    console.log('\n❌ Could not create function via REST API');
    console.log('\nYou need to create this function manually in the Supabase SQL Editor:');
    console.log('\n```sql');
    console.log(createFunctionSQL);
    console.log('```\n');
  }
}

setup Function();
