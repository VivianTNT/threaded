import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkWatchlistByUser() {
  // Check if any watchlisted projects have a user_id
  const { data: projectsWithUserId } = await supabase
    .from('projects')
    .select('id, name, watchlist, user_id')
    .eq('watchlist', true)
    .not('user_id', 'is', null)
    .limit(10);

  console.log('Watchlisted projects WITH user_id:', projectsWithUserId?.length || 0);
  if (projectsWithUserId && projectsWithUserId.length > 0) {
    console.log('Sample:', projectsWithUserId);
  }

  // Check companies
  const { data: companiesWithUserId } = await supabase
    .from('companies')
    .select('id, name, watchlist, user_id')
    .eq('watchlist', true)
    .not('user_id', 'is', null)
    .limit(10);

  console.log('\nWatchlisted companies WITH user_id:', companiesWithUserId?.length || 0);
  if (companiesWithUserId && companiesWithUserId.length > 0) {
    console.log('Sample:', companiesWithUserId);
  }

  // Check news
  const { data: newsWithUserId } = await supabase
    .from('news')
    .select('id, headline, watchlist, user_id')
    .eq('watchlist', true)
    .not('user_id', 'is', null)
    .limit(10);

  console.log('\nWatchlisted news WITH user_id:', newsWithUserId?.length || 0);
  if (newsWithUserId && newsWithUserId.length > 0) {
    console.log('Sample:', newsWithUserId);
  }

  // If all are null, the watchlist system is global, not per-user
  const totalWithUserId = (projectsWithUserId?.length || 0) + (companiesWithUserId?.length || 0) + (newsWithUserId?.length || 0);

  console.log('\n=== RESULT ===');
  if (totalWithUserId === 0) {
    console.log('❌ Watchlist system is GLOBAL (all user_id fields are null)');
    console.log('⚠️  The current code will return 0 watchlisted items for all users');
    console.log('💡 Solution: Either populate user_id fields OR remove the user_id filter');
  } else {
    console.log('✅ Watchlist system is PER-USER');
    console.log(`   Found ${totalWithUserId} items with user_id set`);
  }
}

checkWatchlistByUser();
