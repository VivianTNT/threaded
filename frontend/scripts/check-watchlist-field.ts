import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkWatchlistField() {
  // Check projects table for watchlist field
  const { count: projectsWithWatchlist } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })
    .eq('watchlist', true);

  const { count: totalProjects } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('Projects:');
  console.log('  Total:', totalProjects);
  console.log('  With watchlist=true:', projectsWithWatchlist);

  // Check companies table
  const { count: companiesWithWatchlist } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true })
    .eq('watchlist', true);

  const { count: totalCompanies } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true });

  console.log('\nCompanies:');
  console.log('  Total:', totalCompanies);
  console.log('  With watchlist=true:', companiesWithWatchlist);

  // Check news table
  const { count: newsWithWatchlist } = await supabase
    .from('news')
    .select('*', { count: 'exact', head: true })
    .eq('watchlist', true);

  const { count: totalNews } = await supabase
    .from('news')
    .select('*', { count: 'exact', head: true });

  console.log('\nNews:');
  console.log('  Total:', totalNews);
  console.log('  With watchlist=true:', newsWithWatchlist);

  const totalWatchlisted = (projectsWithWatchlist || 0) + (companiesWithWatchlist || 0) + (newsWithWatchlist || 0);
  console.log('\n=== TOTAL WATCHLISTED:', totalWatchlisted, '===');

  // Sample one watchlisted project to see structure
  const { data: sampleProject } = await supabase
    .from('projects')
    .select('id, name, watchlist, user_id')
    .eq('watchlist', true)
    .limit(1)
    .single();

  console.log('\nSample watchlisted project:');
  console.log(sampleProject);
}

checkWatchlistField();
