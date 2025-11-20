import FirecrawlApp from '@mendable/firecrawl-js';
import { supabase } from '@/lib/supabase/client-server';

interface ScrapedArticle {
  title: string;
  url: string;
  source: string;
  published_at?: string;
  summary?: string;
  commodities?: string[];
  sentiment?: 'Positive' | 'Negative' | 'Neutral';
}

interface ExtractedArticle {
  title: string;
  url: string;
  summary?: string;
  published_date?: string;
}

export class FirecrawlNewsScraper {
  private firecrawl: FirecrawlApp;

  private newsSources = [
    { name: 'Mining.com', url: 'https://www.mining.com/latest-news/' },
    { name: 'Northern Miner', url: 'https://www.northernminer.com/' },
    { name: 'Kitco News', url: 'https://www.kitco.com/news/category/commodities/' },
  ];

  constructor() {
    const apiKey = process.env.FIRECRAWL_API_KEY;
    if (!apiKey) {
      throw new Error('FIRECRAWL_API_KEY is not configured');
    }
    this.firecrawl = new FirecrawlApp({ apiKey });
  }

  async scrapeSource(sourceUrl: string, sourceName: string, maxArticles: number = 10): Promise<ScrapedArticle[]> {
    try {
      console.log(`📰 Scraping ${sourceName}...`);

      // Use Extract endpoint for structured data extraction
      const extractResult = await this.firecrawl.scrapeUrl(sourceUrl, {
        formats: ['extract'],
        extract: {
          schema: {
            type: 'object',
            properties: {
              articles: {
                type: 'array',
                description: 'List of news articles found on the page',
                items: {
                  type: 'object',
                  properties: {
                    title: {
                      type: 'string',
                      description: 'The headline or title of the news article (NOT image captions or teaser text)'
                    },
                    url: {
                      type: 'string',
                      description: 'The full article URL (must be http/https link to article page, NOT images or media files)'
                    },
                    summary: {
                      type: 'string',
                      description: 'A brief summary or excerpt of the article content'
                    },
                    published_date: {
                      type: 'string',
                      description: 'The publication date of the article'
                    }
                  },
                  required: ['title', 'url']
                }
              }
            }
          }
        }
      });

      if (!extractResult.success || !extractResult.extract?.articles) {
        console.log(`  ⚠️  No articles extracted from ${sourceName}, trying fallback...`);
        return await this.scrapeFallback(sourceUrl, sourceName, maxArticles);
      }

      const extractedArticles: ExtractedArticle[] = extractResult.extract.articles;
      console.log(`  📑 Extracted ${extractedArticles.length} raw articles from ${sourceName}`);

      // Transform and filter extracted articles
      const articles: ScrapedArticle[] = extractedArticles
        .filter(article => this.isValidArticle(article))
        .slice(0, maxArticles)
        .map(article => ({
          title: this.cleanTitle(article.title),
          url: article.url,
          source: sourceName,
          published_at: article.published_date || new Date().toISOString(),
          summary: article.summary?.substring(0, 500),
          commodities: this.extractCommodities(article.title + ' ' + (article.summary || '')),
          sentiment: this.analyzeSentiment(article.title + ' ' + (article.summary || '')),
        }));

      console.log(`  ✓ Processed ${articles.length} valid articles from ${sourceName}`);
      return articles;

    } catch (error) {
      console.error(`  ❌ Error scraping ${sourceName}:`, error);
      return [];
    }
  }

  private isValidArticle(article: ExtractedArticle): boolean {
    if (!article.url || !article.title) return false;

    const url = article.url.toLowerCase();
    const title = article.title.toLowerCase();

    // Filter out image URLs
    if (url.match(/\.(jpg|jpeg|png|gif|webp|svg|bmp|ico)(\?|$)/i)) {
      console.log(`  ⏭️  Skipping image URL: ${url}`);
      return false;
    }

    // Filter out teaser/image titles
    if (title.match(/^(image|photo|video|teaser|thumbnail|icon)[\s:]/i)) {
      console.log(`  ⏭️  Skipping teaser: ${title.substring(0, 50)}`);
      return false;
    }

    // Title must be meaningful length
    if (article.title.length < 15 || article.title.length > 250) {
      console.log(`  ⏭️  Skipping invalid title length: ${article.title.length} chars`);
      return false;
    }

    // URL must start with http
    if (!url.startsWith('http')) {
      console.log(`  ⏭️  Skipping invalid URL: ${url}`);
      return false;
    }

    return true;
  }

  private async scrapeFallback(sourceUrl: string, sourceName: string, maxArticles: number): Promise<ScrapedArticle[]> {
    try {
      console.log(`  🔄 Using markdown fallback for ${sourceName}...`);

      const result = await this.firecrawl.scrapeUrl(sourceUrl, {
        formats: ['markdown'],
        onlyMainContent: true,
      });

      if (!result.success || !result.markdown) {
        return [];
      }

      const articles = this.parseMarkdownArticles(result.markdown, sourceName, sourceUrl);
      console.log(`  ✓ Fallback found ${articles.length} articles`);
      return articles.slice(0, maxArticles);
    } catch (error) {
      console.error(`  ❌ Fallback failed for ${sourceName}:`, error);
      return [];
    }
  }

  private parseMarkdownArticles(markdown: string, sourceName: string, baseUrl: string): ScrapedArticle[] {
    const articles: ScrapedArticle[] = [];
    const lines = markdown.split('\n');

    for (const line of lines) {
      const linkMatches = line.match(/\[([^\]]+)\]\(([^)]+)\)/g);

      if (linkMatches) {
        for (const match of linkMatches) {
          const parts = match.match(/\[([^\]]+)\]\(([^)]+)\)/);
          if (!parts) continue;

          const title = parts[1].trim();
          const url = parts[2].trim();

          // Apply same validation as extract
          if (title.length < 15 || title.length > 250) continue;
          if (url.match(/\.(jpg|jpeg|png|gif|webp|svg|bmp)(\?|$)/i)) continue;
          if (title.match(/^(image|photo|video|teaser)[\s:]/i)) continue;
          if (!url.startsWith('http') && !url.startsWith('/')) continue;

          const fullUrl = this.resolveUrl(url, baseUrl);

          articles.push({
            title: this.cleanTitle(title),
            url: fullUrl,
            source: sourceName,
            published_at: new Date().toISOString(),
            commodities: this.extractCommodities(title),
            sentiment: this.analyzeSentiment(title),
          });
        }
      }
    }

    return articles;
  }

  private cleanTitle(title: string): string {
    return title
      .replace(/^(Image|Photo|Video|Teaser|Read more|Click here)[:\s]*/i, '')
      .replace(/\s+\|\s+.*$/, '')
      .replace(/\s+-\s+[^-]+$/, '')
      .replace(/\s*\[.*?\]\s*/g, '') // Remove bracketed text like [teaser image]
      .trim();
  }

  private resolveUrl(url: string, baseUrl: string): string {
    try {
      if (url.startsWith('http')) {
        return url;
      }
      const base = new URL(baseUrl);
      return new URL(url, base.origin).toString();
    } catch {
      return url;
    }
  }

  private extractCommodities(text: string): string[] {
    const commodities = [
      'copper', 'gold', 'silver', 'lithium', 'nickel', 'cobalt',
      'iron ore', 'coal', 'zinc', 'lead', 'platinum', 'palladium',
      'rare earths', 'uranium', 'aluminum', 'tin', 'manganese',
      'graphite', 'vanadium', 'molybdenum', 'tungsten'
    ];

    const found: string[] = [];
    const lowerText = text.toLowerCase();

    for (const commodity of commodities) {
      if (lowerText.includes(commodity)) {
        found.push(commodity.charAt(0).toUpperCase() + commodity.slice(1));
      }
    }

    return found.length > 0 ? found : [];
  }

  private analyzeSentiment(text: string): 'Positive' | 'Negative' | 'Neutral' {
    const lowerText = text.toLowerCase();

    const positiveWords = [
      'growth', 'increase', 'profit', 'success', 'discover', 'breakthrough',
      'record', 'strong', 'surge', 'rally', 'gains', 'boost', 'rise', 'soar'
    ];
    const negativeWords = [
      'decline', 'loss', 'fail', 'drop', 'concern', 'risk', 'delay',
      'weak', 'fall', 'slump', 'plunge', 'threat', 'warning', 'cut'
    ];

    const positiveCount = positiveWords.filter(word => lowerText.includes(word)).length;
    const negativeCount = negativeWords.filter(word => lowerText.includes(word)).length;

    if (positiveCount > negativeCount) return 'Positive';
    if (negativeCount > positiveCount) return 'Negative';
    return 'Neutral';
  }

  async saveToDatabase(articles: ScrapedArticle[]): Promise<number> {
    let saved = 0;

    for (const article of articles) {
      try {
        // Check if article already exists by URL
        const { data: existing } = await supabase
          .from('news')
          .select('id')
          .contains('urls', [article.url])
          .single();

        if (existing) {
          console.log(`  ⏭️  Article already exists: ${article.title.substring(0, 50)}...`);
          continue;
        }

        // Also check by title to avoid duplicates
        const { data: existingTitle } = await supabase
          .from('news')
          .select('id')
          .eq('title', article.title)
          .single();

        if (existingTitle) {
          console.log(`  ⏭️  Duplicate title: ${article.title.substring(0, 50)}...`);
          continue;
        }

        // Insert new article
        const { error } = await supabase
          .from('news')
          .insert({
            title: article.title,
            urls: [article.url],
            source: article.source,
            published_at: article.published_at,
            summary: article.summary,
            commodities: article.commodities,
            sentiment: article.sentiment,
            watchlist: false,
          });

        if (error) {
          console.error(`  ❌ Error saving article:`, error);
        } else {
          console.log(`  ✓ Saved: ${article.title.substring(0, 50)}...`);
          saved++;
        }
      } catch (error) {
        console.error(`  ❌ Error processing article:`, error);
      }
    }

    return saved;
  }

  async scrapeAllSources(maxPerSource: number = 10) {
    console.log(`🚀 Starting news scrape from ${this.newsSources.length} sources...`);

    let totalScraped = 0;
    let totalSaved = 0;
    const sourceResults = [];

    for (const source of this.newsSources) {
      const articles = await this.scrapeSource(source.url, source.name, maxPerSource);
      const saved = await this.saveToDatabase(articles);

      totalScraped += articles.length;
      totalSaved += saved;

      sourceResults.push({
        name: source.name,
        scraped: articles.length,
        saved: saved,
      });

      // Add delay between sources to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    return {
      totalScraped,
      totalSaved,
      sources: sourceResults,
    };
  }
}
