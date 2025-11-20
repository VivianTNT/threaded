import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseServiceKey) {
  throw new Error('Missing Supabase environment variables');
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false
  }
});

async function checkNewsDates() {
  console.log('📅 Checking news article dates...\n');

  try {
    // Get all news sorted by published date
    const { data: allNews } = await supabase
      .from('news')
      .select('title, published_at, created_at, source')
      .order('published_at', { ascending: false })
      .limit(20);

    if (!allNews || allNews.length === 0) {
      console.log('No news articles found');
      return;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    console.log(`📰 Latest news articles (by published_at):\n`);

    allNews.forEach((item, index) => {
      const publishedDate = item.published_at ? new Date(item.published_at) : null;
      const createdDate = item.created_at ? new Date(item.created_at) : null;

      let dateLabel = '';
      if (publishedDate) {
        publishedDate.setHours(0, 0, 0, 0);
        if (publishedDate.getTime() === today.getTime()) {
          dateLabel = '🟢 TODAY';
        } else if (publishedDate.getTime() === yesterday.getTime()) {
          dateLabel = '🟡 YESTERDAY';
        } else if (publishedDate > yesterday) {
          dateLabel = '🟡 RECENT';
        } else {
          const daysAgo = Math.floor((today.getTime() - publishedDate.getTime()) / (1000 * 60 * 60 * 24));
          dateLabel = `🔴 ${daysAgo} days ago`;
        }
      }

      console.log(`${index + 1}. ${dateLabel}`);
      console.log(`   Title: ${item.title?.substring(0, 70)}...`);
      console.log(`   Source: ${item.source}`);
      console.log(`   Published: ${publishedDate?.toLocaleDateString() || 'N/A'}`);
      console.log(`   Created in DB: ${createdDate?.toLocaleString() || 'N/A'}`);
      console.log('');
    });

    // Get date statistics
    const { data: stats } = await supabase
      .from('news')
      .select('published_at');

    if (stats) {
      const dates = stats
        .map(s => s.published_at ? new Date(s.published_at) : null)
        .filter((d): d is Date => d !== null);

      const todayCount = dates.filter(d => {
        d.setHours(0, 0, 0, 0);
        return d.getTime() === today.getTime();
      }).length;

      const thisWeekStart = new Date(today);
      thisWeekStart.setDate(today.getDate() - 7);

      const thisWeekCount = dates.filter(d => d >= thisWeekStart).length;

      console.log('\n📊 Date Statistics:');
      console.log(`   Today: ${todayCount} articles`);
      console.log(`   This week: ${thisWeekCount} articles`);
      console.log(`   Total: ${stats.length} articles`);
    }

    // Check what today's date is
    console.log(`\n🗓️  Today's date: ${today.toLocaleDateString()}`);
    console.log(`   Time now: ${new Date().toLocaleString()}`);

  } catch (error) {
    console.error('❌ Error:', error);
    throw error;
  }
}

checkNewsDates()
  .then(() => {
    console.log('\n✅ Check complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Failed:', error);
    process.exit(1);
  });
