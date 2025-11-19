import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkWatchlistData() {
  // Get total usr table rows
  const { count: totalUsrRows } = await supabase
    .from('usr')
    .select('*', { count: 'exact', head: true });

  console.log('Total usr table rows (all users):', totalUsrRows);

  // Sample some rows to see structure
  const { data: sampleRows } = await supabase
    .from('usr')
    .select('*')
    .limit(10);

  console.log('\nSample usr rows:');
  sampleRows?.forEach(row => {
    console.log('  Row:', JSON.stringify(row));
  });

  // Count by user
  const { data: allRows } = await supabase.from('usr').select('uid');
  const countsByUser = allRows?.reduce((acc: any, row: any) => {
    acc[row.uid] = (acc[row.uid] || 0) + 1;
    return acc;
  }, {});

  console.log('\nWatchlist counts by user:');
  const userIds = Object.keys(countsByUser || {});
  userIds.forEach(uid => {
    console.log(`  User ${uid.substring(0, 8)}...: ${countsByUser[uid]} items`);
  });

  console.log(`\nTotal unique users: ${userIds.length}`);
}

checkWatchlistData();
