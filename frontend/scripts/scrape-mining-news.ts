import { createClient } from '@supabase/supabase-js';
import Firecrawl from '@mendable/firecrawl-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const firecrawlKey = process.env.FIRECRAWL_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);
const firecrawl = new Firecrawl({ apiKey: firecrawlKey });

const NEWS_SOURCES = [
  { url: 'https://www.mining.com/latest-news/', name: 'Mining.com', pages: 5 },
  { url: 'https://www.northernminer.com/', name: 'Northern Miner', pages: 5 },
  { url: 'https://www.kitco.com/news/category/commodities/', name: 'Kitco News', pages: 5 },
  { url: 'https://www.mining-technology.com/news/', name: 'Mining Technology', pages: 3 },
  { url: 'https://www.miningweekly.com/', name: 'Mining Weekly', pages: 5 },
  { url: 'https://www.australianmining.com.au/news/', name: 'Australian Mining', pages: 3 },
  { url: 'https://www.canadianminingjournal.com/', name: 'Canadian Mining Journal', pages: 3 },
  { url: 'https://www.mining.com/web/category/copper/', name: 'Mining.com Copper', pages: 2 },
  { url: 'https://www.mining.com/web/category/gold/', name: 'Mining.com Gold', pages: 2 },
  { url: 'https://www.mining.com/web/category/lithium/', name: 'Mining.com Lithium', pages: 2 },
  { url: 'https://www.mining.com/web/category/nickel/', name: 'Mining.com Nickel', pages: 2 },
];

let newsAdded = 0;

async function scrapeNewsFromSource(source: typeof NEWS_SOURCES[0]) {
  console.log(`\n📰 Scraping ${source.name}...`);

  for (let page = 1; page <= source.pages; page++) {
    try {
      const pageUrl = source.url + (page > 1 ? `page/${page}/` : '');

      const result = await firecrawl.scrapeUrl(pageUrl, {
        formats: ['extract'],
        extract: {
          schema: {
            type: 'object',
            properties: {
              articles: {
                type: 'array',
                description: 'List of news articles on this page',
                items: {
                  type: 'object',
                  properties: {
                    title: { type: 'string', description: 'Article headline/title' },
                    url: { type: 'string', description: 'Full URL to the article' },
                    published_date: { type: 'string', description: 'Publication date in ISO format or readable format' },
                    summary: { type: 'string', description: 'Brief summary or excerpt of the article (2-3 sentences)' },
                    commodities: {
                      type: 'array',
                      items: { type: 'string' },
                      description: 'Commodities mentioned (Gold, Copper, Lithium, Nickel, Rare Earths, etc.)'
                    },
                    sentiment: {
                      type: 'string',
                      enum: ['Positive', 'Negative', 'Neutral'],
                      description: 'Overall sentiment of the article'
                    },
                  }
                }
              }
            }
          }
        }
      });

      const articles = result.extract?.articles || [];
      console.log(`  Page ${page}: Found ${articles.length} articles`);

      for (const article of articles) {
        if (!article.title || !article.url) continue;

        // Check if article already exists
        const { data: existing } = await supabase
          .from('news')
          .select('id')
          .eq('title', article.title)
          .single();

        if (existing) {
          console.log(`    ⏭️  Exists: ${article.title.substring(0, 50)}...`);
          continue;
        }

        // Parse published date
        let publishedAt = null;
        if (article.published_date) {
          const date = new Date(article.published_date);
          if (!isNaN(date.getTime())) {
            publishedAt = date.toISOString();
          }
        }
        // Default to today if no date found
        if (!publishedAt) {
          publishedAt = new Date().toISOString();
        }

        const { error } = await supabase
          .from('news')
          .insert({
            title: article.title,
            urls: [article.url],
            source: source.name,
            published_at: publishedAt,
            summary: article.summary || article.title,
            commodities: article.commodities && article.commodities.length > 0 ? article.commodities : null,
            sentiment: article.sentiment || 'Neutral',
            project_ids: [],
          });

        if (error) {
          console.error(`    ❌ Error adding article:`, error.message);
        } else {
          newsAdded++;
          console.log(`    ✓ ${article.title.substring(0, 60)}...`);
        }
      }

      // Delay between pages
      await new Promise(resolve => setTimeout(resolve, 2000));

    } catch (error: any) {
      console.error(`  ❌ Error scraping page ${page} of ${source.name}:`, error.message);
    }
  }
}

async function main() {
  console.log(`🚀 MINING NEWS SCRAPER`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);
  console.log(`📰 News sources: ${NEWS_SOURCES.length}`);
  console.log(`🎯 Goal: 200+ news articles\n`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);

  for (const source of NEWS_SOURCES) {
    await scrapeNewsFromSource(source);

    // Check progress
    const { count: totalNews } = await supabase.from('news').select('*', { count: 'exact', head: true });
    console.log(`\n📊 Total news articles in database: ${totalNews}`);

    if (totalNews && totalNews >= 200) {
      console.log(`\n🎉 TARGET REACHED: ${totalNews} news articles!`);
      break;
    }
  }

  console.log(`\n═══════════════════════════════════════════════════════════════`);
  console.log(`✅ NEWS SCRAPING COMPLETE!`);
  console.log(`═══════════════════════════════════════════════════════════════`);
  console.log(`📊 Articles added in this session: ${newsAdded}`);

  const { count: finalNews } = await supabase.from('news').select('*', { count: 'exact', head: true });
  console.log(`📊 Total news articles: ${finalNews}`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);
}

main();
