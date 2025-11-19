import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkWatchlistTable() {
  const { data, error, count } = await supabase
    .from('user_project_watchlist')
    .select('*', { count: 'exact' })
    .limit(10);

  console.log('user_project_watchlist table:');
  console.log('  Count:', count);
  console.log('  Error:', error?.message || 'none');
  console.log('  Sample data:', data);
}

checkWatchlistTable();
