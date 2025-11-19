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

async function cleanupNews() {
  console.log('🧹 Starting news cleanup...\n');

  try {
    // Get initial count
    const { count: initialCount } = await supabase
      .from('news')
      .select('*', { count: 'exact', head: true });

    console.log(`📊 Total entries before cleanup: ${initialCount}\n`);

    // Step 1: Remove entries with image URLs
    console.log('🖼️  Removing entries with image URLs...');
    const { data: newsWithImageUrls } = await supabase
      .from('news')
      .select('id, urls, title');

    const imageUrlIds: string[] = [];
    newsWithImageUrls?.forEach((item) => {
      if (item.urls && Array.isArray(item.urls)) {
        const hasImageUrl = item.urls.some((url: string) =>
          /\.(jpg|jpeg|png|gif|webp|svg|bmp|ico)(\?|$)/i.test(url)
        );
        if (hasImageUrl) {
          imageUrlIds.push(item.id);
          console.log(`  - Removing: ${item.title?.substring(0, 50)}... (image URL)`);
        }
      }
    });

    if (imageUrlIds.length > 0) {
      const { error } = await supabase
        .from('news')
        .delete()
        .in('id', imageUrlIds);

      if (error) throw error;
      console.log(`  ✅ Removed ${imageUrlIds.length} entries with image URLs\n`);
    } else {
      console.log(`  ✅ No entries with image URLs found\n`);
    }

    // Step 2: Remove entries with teaser/image titles
    console.log('📝 Removing entries with teaser/image titles...');
    const { data: allNews } = await supabase
      .from('news')
      .select('id, title');

    const teaserTitleIds: string[] = [];
    allNews?.forEach((item) => {
      if (/^(image|photo|video|teaser|thumbnail|icon)[\s:]/i.test(item.title || '')) {
        teaserTitleIds.push(item.id);
        console.log(`  - Removing: ${item.title?.substring(0, 50)}...`);
      }
    });

    if (teaserTitleIds.length > 0) {
      const { error } = await supabase
        .from('news')
        .delete()
        .in('id', teaserTitleIds);

      if (error) throw error;
      console.log(`  ✅ Removed ${teaserTitleIds.length} entries with teaser titles\n`);
    } else {
      console.log(`  ✅ No entries with teaser titles found\n`);
    }

    // Step 3: Remove entries with invalid title lengths
    console.log('📏 Removing entries with invalid title lengths...');
    const { data: remainingNews } = await supabase
      .from('news')
      .select('id, title');

    const invalidLengthIds: string[] = [];
    remainingNews?.forEach((item) => {
      const length = item.title?.length || 0;
      if (length < 15 || length > 250) {
        invalidLengthIds.push(item.id);
        console.log(`  - Removing: ${item.title?.substring(0, 50)}... (length: ${length})`);
      }
    });

    if (invalidLengthIds.length > 0) {
      const { error } = await supabase
        .from('news')
        .delete()
        .in('id', invalidLengthIds);

      if (error) throw error;
      console.log(`  ✅ Removed ${invalidLengthIds.length} entries with invalid title lengths\n`);
    } else {
      console.log(`  ✅ No entries with invalid title lengths found\n`);
    }

    // Step 4: Remove entries with empty/null titles or URLs
    console.log('❌ Removing entries with empty titles or URLs...');
    const { data: newsToCheck } = await supabase
      .from('news')
      .select('id, title, urls');

    const emptyDataIds: string[] = [];
    newsToCheck?.forEach((item) => {
      if (!item.title || item.title.trim() === '' ||
          !item.urls || !Array.isArray(item.urls) || item.urls.length === 0) {
        emptyDataIds.push(item.id);
        console.log(`  - Removing: ${item.title || '(no title)'}`);
      }
    });

    if (emptyDataIds.length > 0) {
      const { error } = await supabase
        .from('news')
        .delete()
        .in('id', emptyDataIds);

      if (error) throw error;
      console.log(`  ✅ Removed ${emptyDataIds.length} entries with empty data\n`);
    } else {
      console.log(`  ✅ No entries with empty data found\n`);
    }

    // Step 5: Remove duplicates based on URL
    console.log('🔗 Removing URL duplicates...');
    const { data: allValidNews } = await supabase
      .from('news')
      .select('id, urls, created_at')
      .order('created_at', { ascending: true });

    const seenUrls = new Set<string>();
    const urlDuplicateIds: string[] = [];

    allValidNews?.forEach((item) => {
      if (item.urls && Array.isArray(item.urls)) {
        const hasSeenUrl = item.urls.some((url: string) => seenUrls.has(url));
        if (hasSeenUrl) {
          urlDuplicateIds.push(item.id);
          console.log(`  - Removing duplicate URL: ${item.urls[0]?.substring(0, 50)}...`);
        } else {
          item.urls.forEach((url: string) => seenUrls.add(url));
        }
      }
    });

    if (urlDuplicateIds.length > 0) {
      const { error } = await supabase
        .from('news')
        .delete()
        .in('id', urlDuplicateIds);

      if (error) throw error;
      console.log(`  ✅ Removed ${urlDuplicateIds.length} URL duplicates\n`);
    } else {
      console.log(`  ✅ No URL duplicates found\n`);
    }

    // Step 6: Remove duplicates based on exact title match
    console.log('📰 Removing title duplicates...');
    const { data: finalNews } = await supabase
      .from('news')
      .select('id, title, created_at')
      .order('created_at', { ascending: true });

    const seenTitles = new Set<string>();
    const titleDuplicateIds: string[] = [];

    finalNews?.forEach((item) => {
      if (seenTitles.has(item.title)) {
        titleDuplicateIds.push(item.id);
        console.log(`  - Removing duplicate title: ${item.title?.substring(0, 50)}...`);
      } else {
        seenTitles.add(item.title);
      }
    });

    if (titleDuplicateIds.length > 0) {
      const { error } = await supabase
        .from('news')
        .delete()
        .in('id', titleDuplicateIds);

      if (error) throw error;
      console.log(`  ✅ Removed ${titleDuplicateIds.length} title duplicates\n`);
    } else {
      console.log(`  ✅ No title duplicates found\n`);
    }

    // Get final count and stats
    const { count: finalCount } = await supabase
      .from('news')
      .select('*', { count: 'exact', head: true });

    const { data: statsData } = await supabase
      .from('news')
      .select('title');

    const titleLengths = statsData?.map(item => item.title?.length || 0) || [];
    const minLength = Math.min(...titleLengths);
    const maxLength = Math.max(...titleLengths);

    console.log('\n✅ Cleanup completed!');
    console.log('═══════════════════════════════════════');
    console.log(`📊 Total entries before: ${initialCount}`);
    console.log(`📊 Total entries after:  ${finalCount}`);
    console.log(`🗑️  Entries removed:      ${(initialCount || 0) - (finalCount || 0)}`);
    console.log(`📏 Title length range:   ${minLength} - ${maxLength} characters`);
    console.log('═══════════════════════════════════════\n');

  } catch (error) {
    console.error('❌ Error during cleanup:', error);
    throw error;
  }
}

// Run the cleanup
cleanupNews()
  .then(() => {
    console.log('✅ All done!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('❌ Failed:', error);
    process.exit(1);
  });
