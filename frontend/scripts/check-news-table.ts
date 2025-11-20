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

async function checkNewsTable() {
  console.log('🔍 Checking news table...\n');

  try {
    // Get all news entries
    const { data, error, count } = await supabase
      .from('news')
      .select('*', { count: 'exact' });

    if (error) {
      console.error('❌ Error fetching news:', error);
      throw error;
    }

    console.log(`📊 Total entries: ${count}`);
    console.log(`📊 Data entries: ${data?.length || 0}\n`);

    if (data && data.length > 0) {
      console.log('📰 Sample entries:\n');
      data.slice(0, 5).forEach((item, index) => {
        console.log(`${index + 1}. ${item.title?.substring(0, 80)}...`);
        console.log(`   Source: ${item.source || 'N/A'}`);
        console.log(`   URLs: ${item.urls?.length || 0} link(s)`);
        if (item.urls && item.urls.length > 0) {
          console.log(`   First URL: ${item.urls[0]?.substring(0, 100)}`);
        }
        console.log(`   Published: ${item.published_at || 'N/A'}`);
        console.log('');
      });

      // Check for potential issues
      console.log('\n🔍 Checking for issues:\n');

      let imageUrlCount = 0;
      let teaserTitleCount = 0;
      let shortTitleCount = 0;
      let longTitleCount = 0;

      data.forEach((item) => {
        // Check for image URLs
        if (item.urls && Array.isArray(item.urls)) {
          const hasImageUrl = item.urls.some((url: string) =>
            /\.(jpg|jpeg|png|gif|webp|svg|bmp|ico)(\?|$)/i.test(url)
          );
          if (hasImageUrl) {
            imageUrlCount++;
            console.log(`  ⚠️  Image URL: ${item.title?.substring(0, 50)}...`);
          }
        }

        // Check for teaser titles
        if (/^(image|photo|video|teaser|thumbnail|icon)[\s:]/i.test(item.title || '')) {
          teaserTitleCount++;
          console.log(`  ⚠️  Teaser title: ${item.title?.substring(0, 50)}...`);
        }

        // Check title length
        const length = item.title?.length || 0;
        if (length < 15) {
          shortTitleCount++;
          console.log(`  ⚠️  Short title (${length} chars): ${item.title}`);
        } else if (length > 250) {
          longTitleCount++;
          console.log(`  ⚠️  Long title (${length} chars): ${item.title?.substring(0, 50)}...`);
        }
      });

      console.log('\n📊 Issue Summary:');
      console.log(`  - Image URLs: ${imageUrlCount}`);
      console.log(`  - Teaser titles: ${teaserTitleCount}`);
      console.log(`  - Short titles (<15 chars): ${shortTitleCount}`);
      console.log(`  - Long titles (>250 chars): ${longTitleCount}`);
    } else {
      console.log('ℹ️  Table is empty');
    }

  } catch (error) {
    console.error('❌ Error:', error);
    throw error;
  }
}

// Run the check
checkNewsTable()
  .then(() => {
    console.log('\n✅ Check complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Failed:', error);
    process.exit(1);
  });
