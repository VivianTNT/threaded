import { NextRequest } from 'next/server';
import { FirecrawlCombinedScraper } from '@/lib/news-scraper/firecrawl-combined';
import { createClient } from '@supabase/supabase-js';

// Server-Sent Events for real-time updates
export async function POST(request: NextRequest) {
  const encoder = new TextEncoder();
  
  // Create a readable stream for SSE
  const stream = new ReadableStream({
    async start(controller) {
      const sendEvent = (data: any) => {
        const message = `data: ${JSON.stringify(data)}\n\n`;
        controller.enqueue(encoder.encode(message));
      };

      try {
        // Get parameters from request
        const body = await request.json().catch(() => ({}));
        const { maxResults = 25 } = body;

        // Check if Firecrawl API key is configured
        const firecrawlApiKey = process.env.FIRECRAWL_API_KEY;
        if (!firecrawlApiKey || firecrawlApiKey === 'your_firecrawl_api_key_here') {
          sendEvent({
            type: 'error',
            error: 'News scraping is not configured. Firecrawl API key is missing.',
            message: 'To enable news scraping, please add FIRECRAWL_API_KEY to your .env.local file. You can get a free API key at https://www.firecrawl.dev/',
            stage: 'error'
          });
          return;
        }

        // Send initial status
        sendEvent({
          type: 'status',
          stage: 'initializing',
          description: 'Starting targeted scrape of Mining.com, Kitco News, Mining Weekly, Northern Miner, and Mining Journal',
          progress: 5
        });

        // Initialize scraper
        const scraper = new FirecrawlCombinedScraper();
        
        // Initialize Supabase
        const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
        const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
        
        if (!supabaseUrl || !supabaseKey) {
          throw new Error('Supabase configuration missing');
        }
        
        const supabase = createClient(supabaseUrl, supabaseKey);
        
        // Scrape with detailed progress updates
        const result = await scraper.scrapeAll(maxResults, (update) => {
          if (update.type === 'status') {
            // Send detailed status updates
            sendEvent({
              type: 'status',
              stage: update.stage,
              description: update.description,
              progress: update.progress || (10 + (update.articlesFound || 0) * 3),
              articlesFound: update.articlesFound || 0,
              sitesProcessed: update.sitesProcessed,
              totalSites: update.totalSites
            });
          } else if (update.type === 'article_saved') {
            // Send article saved notification
            sendEvent({
              type: 'article_saved',
              source: update.source,
              headline: update.headline,
              totalSaved: update.totalSaved
            });
          } else if (update.type === 'complete') {
            // Send completion status
            sendEvent({
              type: 'status',
              stage: 'completed',
              description: `Successfully scraped ${update.totalScraped} sources and saved ${update.totalSaved} articles`,
              progress: 100,
              totalArticles: update.totalSaved,
              totalScraped: update.totalScraped
            });
          }
        });

        const { articles, totalScraped, totalSaved } = result;

        // Save to database
        sendEvent({
          type: 'status',
          stage: 'saving',
          description: `Saving ${articles.length} articles to database with extracted metadata`,
          progress: 85,
          articlesFound: articles.length
        });

        // Clear existing news (optional - or merge with existing)
        await supabase
          .from('unified_news')
          .delete()
          .neq('id', 0); // Delete all rows

        // Insert new articles
        let savedCount = 0;
        for (const article of articles) {
          try {
            const { error } = await supabase
              .from('unified_news')
              .insert({
                headline: article.headline,
                summary: article.summary,
                url: article.url,
                source_name: article.source_name,
                published_date: article.published_date,
                primary_commodity: article.primary_commodity,
                commodities: article.secondary_commodities ? 
                  [article.primary_commodity, ...article.secondary_commodities].filter(Boolean) :
                  article.primary_commodity ? [article.primary_commodity] : null,
                company: article.company,
                symbol: article.symbol,
                countries: article.countries,
                regions: article.regions,
                project_names: article.project_names,
                topics: article.topics,
                tags: article.tags,
                sentiment_score: article.sentiment_score,
                relevance_score: article.relevance_score,
                source_type: 'news',
                scraper_source: 'firecrawl_combined'
              });
            
            if (error) {
              console.error('Error saving article:', error);
            } else {
              savedCount++;
              // Send progress update every 5 articles
              if (savedCount % 5 === 0) {
                sendEvent({
                  type: 'status',
                  stage: 'saving',
                  description: `Saved ${savedCount} of ${articles.length} articles to database`,
                  progress: 85 + (savedCount / articles.length * 10),
                  articlesFound: savedCount
                });
              }
            }
          } catch (error) {
            console.error('Error saving article:', error);
          }
        }

        // Send final completion status
        sendEvent({
          type: 'status',
          stage: 'completed',
          description: `News refresh complete! Scraped ${totalScraped} sources and saved ${savedCount} articles`,
          progress: 100,
          totalArticles: savedCount,
          totalScraped: totalScraped
        });

        // Send completion event
        sendEvent({ 
          type: 'complete', 
          totalScraped,
          totalSaved: savedCount,
          message: `Successfully refreshed news feed with ${savedCount} articles from ${totalScraped} sources`
        });

      } catch (error) {
        console.error('News refresh error:', error);
        sendEvent({ 
          type: 'error', 
          message: error instanceof Error ? error.message : 'An error occurred during news refresh'
        });
      } finally {
        // Close the stream
        controller.close();
      }
    }
  });

  // Return the stream as a response
  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    },
  });
}