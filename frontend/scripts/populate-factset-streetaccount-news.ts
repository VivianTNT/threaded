#!/usr/bin/env tsx

/**
 * Populate Supabase with FactSet StreetAccount News
 * Fetches thousands of mining news articles and enriches with AI
 */

import { createClient } from '@supabase/supabase-js';
import { ApiClient, HeadlinesApi } from '@factset/sdk-streetaccountnews';
import OpenAI from 'openai';
import { config } from 'dotenv';
import path from 'path';

config({ path: path.join(__dirname, '..', '.env.local') });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const openaiKey = process.env.OPENAI_API_KEY!;

if (!supabaseUrl || !supabaseKey || !openaiKey) {
  console.error('Missing required environment variables');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);
const openai = new OpenAI({ apiKey: openaiKey });

// FactSet StreetAccount credentials
const FACTSET_USERNAME = 'LITHOS-2220379';
const FACTSET_PASSWORD = 'YaCeW9yKju608Q4rWUt8yxn0hb2AGQQmpyLde8IG';

interface FactSetHeadline {
  storyTime?: string;
  headlines?: string;
  id?: string;
  primarySymbols?: string[];
  symbols?: string[];
  subjects?: string[];
  storyBody?: string;
  url?: string;
}

async function fetchHeadlines(options: {
  predefinedRange?: string;
  sectors?: string[];
  regions?: string[];
  limit: number;
  offset: number;
}): Promise<FactSetHeadline[]> {
  const { predefinedRange, sectors, regions, limit, offset } = options;

  // Initialize API client
  const apiClient = ApiClient.instance;
  const factSetApiKey = apiClient.authentications['FactSetApiKey'];
  factSetApiKey.username = FACTSET_USERNAME;
  factSetApiKey.password = FACTSET_PASSWORD;

  const headlinesApi = new HeadlinesApi();

  const requestBody: any = {
    data: {},
    meta: {
      pagination: {
        limit: Math.min(limit, 500), // API max
        offset: offset
      }
    }
  };

  if (predefinedRange) {
    requestBody.data.predefinedRange = predefinedRange;
  }

  if (sectors && sectors.length > 0) {
    requestBody.data.sectors = sectors;
  }

  if (regions && regions.length > 0) {
    requestBody.data.regions = regions;
  }

  try {
    const response = await headlinesApi.getStreetAccountHeadlines(requestBody);
    return response && response.data ? response.data : [];
  } catch (error: any) {
    console.error(`Error fetching headlines:`, error.message);
    return [];
  }
}

async function enrichWithAI(headline: FactSetHeadline): Promise<{
  summary: string;
  commodities: string[];
  sentiment: string;
}> {
  const prompt = `Analyze this financial news headline and content:

Headline: ${headline.headlines || 'N/A'}
Tickers: ${headline.symbols?.join(', ') || 'N/A'}
Content: ${headline.storyBody?.replace(/<[^>]*>/g, '').substring(0, 1000) || 'N/A'}

Extract:
1. A concise 1-2 sentence summary
2. All relevant mining/commodities mentioned (Gold, Copper, Lithium, Silver, Nickel, Iron Ore, etc.)
3. Sentiment (Positive, Negative, or Neutral) - consider impact on mining industry

Return as JSON:
{
  "summary": "...",
  "commodities": ["..."],
  "sentiment": "Positive|Negative|Neutral"
}`;

  try {
    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: 'You are a mining industry analyst. Extract structured information from news articles about mining, metals, and commodities.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      response_format: { type: 'json_object' },
      temperature: 0.3,
      max_tokens: 500
    });

    const content = response.choices[0]?.message?.content;
    if (!content) {
      throw new Error('No response from OpenAI');
    }

    const analysis = JSON.parse(content);
    return {
      summary: analysis.summary || headline.headlines || '',
      commodities: analysis.commodities || [],
      sentiment: analysis.sentiment || 'Neutral'
    };
  } catch (error: any) {
    console.error(`AI enrichment error:`, error.message);
    return {
      summary: headline.headlines || '',
      commodities: [],
      sentiment: 'Neutral'
    };
  }
}

async function saveToSupabase(headline: FactSetHeadline, enrichment: any): Promise<boolean> {
  try {
    // Check if already exists
    const { data: existing } = await supabase
      .from('news')
      .select('id')
      .eq('title', headline.headlines || '')
      .eq('published_at', headline.storyTime || '')
      .single();

    if (existing) {
      return false; // Skip duplicate
    }

    // Insert new news item
    const { error } = await supabase
      .from('news')
      .insert({
        title: headline.headlines || 'No title',
        urls: headline.url ? [headline.url] : null,
        source: 'FactSet StreetAccount',
        published_at: headline.storyTime || new Date().toISOString(),
        summary: enrichment.summary,
        commodities: enrichment.commodities.length > 0 ? enrichment.commodities : null,
        project_ids: null,
        sentiment: enrichment.sentiment,
        watchlist: false
      });

    if (error) {
      console.error(`Error saving to Supabase:`, error.message);
      return false;
    }

    return true;
  } catch (error: any) {
    console.error(`Error in saveToSupabase:`, error.message);
    return false;
  }
}

async function main() {
  console.log('=== FactSet StreetAccount News Population ===\n');

  // Fetch multiple date ranges to get thousands of articles
  const dateRanges = [
    'today',
    'twoDays',
    'oneWeek',
    'oneMonth',
    'threeMonths'
  ];

  let totalFetched = 0;
  let totalEnriched = 0;
  let totalSaved = 0;
  let totalSkipped = 0;

  for (const range of dateRanges) {
    console.log(`\n--- Fetching headlines for range: ${range} ---`);

    let offset = 0;
    let hasMore = true;
    let batchCount = 0;

    while (hasMore && batchCount < 10) { // Limit to 10 batches per range (5000 max)
      console.log(`  Batch ${batchCount + 1}: Fetching 500 headlines (offset: ${offset})...`);

      const headlines = await fetchHeadlines({
        predefinedRange: range,
        limit: 500,
        offset: offset
      });

      totalFetched += headlines.length;

      if (headlines.length === 0) {
        hasMore = false;
        break;
      }

      // Process in smaller batches for AI enrichment
      const aiBatchSize = 10;
      for (let i = 0; i < headlines.length; i += aiBatchSize) {
        const batch = headlines.slice(i, i + aiBatchSize);

        const enrichedBatch = await Promise.all(
          batch.map(async (headline) => {
            const enrichment = await enrichWithAI(headline);
            totalEnriched++;
            return { headline, enrichment };
          })
        );

        // Save batch to Supabase
        for (const { headline, enrichment } of enrichedBatch) {
          const saved = await saveToSupabase(headline, enrichment);
          if (saved) {
            totalSaved++;
          } else {
            totalSkipped++;
          }
        }

        console.log(`    Processed ${Math.min(i + aiBatchSize, headlines.length)}/${headlines.length} headlines (Saved: ${totalSaved}, Skipped: ${totalSkipped})`);

        // Rate limit: Wait 1 second between AI batches
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      offset += headlines.length;
      batchCount++;

      if (headlines.length < 500) {
        hasMore = false; // Last batch
      }

      // Wait between batches to avoid rate limits
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }

  // Get final count from database
  const { count } = await supabase
    .from('news')
    .select('*', { count: 'exact', head: true });

  console.log('\n=== Population Complete ===');
  console.log(`Total fetched: ${totalFetched}`);
  console.log(`Total enriched: ${totalEnriched}`);
  console.log(`Total saved: ${totalSaved}`);
  console.log(`Total skipped (duplicates): ${totalSkipped}`);
  console.log(`Total in database: ${count || 0}`);
}

main().catch(console.error);
