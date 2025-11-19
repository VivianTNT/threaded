import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkUserWatchlist() {
  // Get total watchlist rows
  const { count: totalRows } = await supabase
    .from('user_watchlist')
    .select('*', { count: 'exact', head: true });

  console.log('Total user_watchlist rows:', totalRows);

  // Sample some rows to see structure
  const { data: sampleRows } = await supabase
    .from('user_watchlist')
    .select('*')
    .limit(10);

  console.log('\nSample user_watchlist rows:');
  sampleRows?.forEach(row => {
    console.log('  Row:', JSON.stringify(row));
  });

  // Count by user
  const { data: allRows } = await supabase.from('user_watchlist').select('user_id, project_id, company_id, news_id');

  if (allRows) {
    const userCounts: Record<string, number> = {};
    allRows.forEach(row => {
      const uid = (row as any).user_id;
      if (uid) {
        userCounts[uid] = (userCounts[uid] || 0) + 1;
      }
    });

    console.log('\nWatchlist counts by user:');
    Object.entries(userCounts).forEach(([uid, count]) => {
      console.log(`  User ${uid}: ${count} items`);
    });

    console.log(`\nTotal unique users with watchlists: ${Object.keys(userCounts).length}`);
  }
}

checkUserWatchlist();
