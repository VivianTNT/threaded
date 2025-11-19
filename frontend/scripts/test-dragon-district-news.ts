import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function checkDragonDistrict() {
  // Find the Dragon District project
  const { data: project } = await supabase
    .from('projects')
    .select('id, name, company_id, commodities, location, country')
    .ilike('name', '%Dragon District%')
    .limit(1)
    .single();

  console.log('=== DRAGON DISTRICT PROJECT ===');
  console.log('Project:', project);

  if (!project) {
    console.log('Project not found');
    return;
  }

  // Check for recent news (last 90 days)
  const ninetyDaysAgo = new Date();
  ninetyDaysAgo.setDate(ninetyDaysAgo.getDate() - 90);

  const { data: recentNews, count } = await supabase
    .from('news')
    .select('title, published_at, commodities, source', { count: 'exact' })
    .gte('published_at', ninetyDaysAgo.toISOString())
    .limit(5);

  console.log('\n=== RECENT NEWS (last 90 days) ===');
  console.log('Total news items:', count);
  console.log('Sample:', recentNews?.map(n => ({
    title: n.title?.substring(0, 60) + '...',
    source: n.source,
    date: new Date(n.published_at || '').toLocaleDateString()
  })));

  // Check if any news mentions Turkey, British Columbia, or relevant commodities
  const projectCommodities = project.commodities || [];
  console.log('\nProject commodities:', projectCommodities);

  // Check for commodity-related news
  if (projectCommodities.length > 0) {
    const { data: commodityNews, count: commCount } = await supabase
      .from('news')
      .select('title, commodities, source', { count: 'exact' })
      .gte('published_at', ninetyDaysAgo.toISOString())
      .overlaps('commodities', projectCommodities)
      .limit(5);

    console.log('\n=== COMMODITY-RELATED NEWS ===');
    console.log('Matching news:', commCount);
    console.log('Sample:', commodityNews?.map(n => ({
      title: n.title?.substring(0, 60) + '...',
      commodities: n.commodities,
      source: n.source
    })));
  }

  // Now test the relevance scoring algorithm
  console.log('\n=== TESTING RELEVANCE ALGORITHM ===');
  const { extractNewsContext } = await import('../lib/ai/document-context-extractor');
  const newsContext = await extractNewsContext(project.id, 10);

  console.log('News items returned:', newsContext.newsItems.length);
  newsContext.newsItems.forEach((item, idx) => {
    console.log(`\n${idx + 1}. ${item.title?.substring(0, 60)}...`);
    console.log('   Source:', item.source);
    console.log('   Relevance Score:', item.relevanceScore);
    console.log('   Published:', new Date(item.publishedAt).toLocaleDateString());
  });
}

checkDragonDistrict();
