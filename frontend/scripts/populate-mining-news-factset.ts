#!/usr/bin/env tsx

/**
 * Populate Supabase with Mining-Specific FactSet StreetAccount News
 * Focuses on mining, metals, and commodities news only
 */

import { createClient } from '@supabase/supabase-js';
import { ApiClient, HeadlinesApi, FiltersApi } from '@factset/sdk-streetaccountnews';
import OpenAI from 'openai';
import { config } from 'dotenv';
import path from 'path';

config({ path: path.join(__dirname, '..', '.env.local') });

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const openaiKey = process.env.OPENAI_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseKey);
const openai = new OpenAI({ apiKey: openaiKey });

// FactSet credentials
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

async function fetchMiningHeadlines(limit: number): Promise<FactSetHeadline[]> {
  const apiClient = ApiClient.instance;
  const factSetApiKey = apiClient.authentications['FactSetApiKey'];
  factSetApiKey.username = FACTSET_USERNAME;
  factSetApiKey.password = FACTSET_PASSWORD;

  const headlinesApi = new HeadlinesApi();

  const requestBody: any = {
    data: {
      predefinedRange: 'oneMonth' // Get last month of news
    },
    meta: {
      pagination: {
        limit: Math.min(limit, 500),
        offset: 0
      }
    }
  };

  const response = await headlinesApi.getStreetAccountHeadlines(requestBody);
  return response && response.data ? response.data : [];
}

async function isMiningRelated(headline: FactSetHeadline): Promise<boolean> {
  const text = `${headline.headlines || ''} ${headline.storyBody?.replace(/<[^>]*>/g, '').substring(0, 500) || ''}`.toLowerCase();

  const miningKeywords = [
    'mining', 'mine', 'mines', 'mineral', 'minerals',
    'gold', 'silver', 'copper', 'lithium', 'nickel', 'cobalt', 'zinc', 'iron ore',
    'platinum', 'palladium', 'rare earth', 'molybdenum', 'manganese',
    'exploration', 'drill', 'drilling', 'ore', 'deposit', 'resource', 'reserve',
    'metallurgical', 'smelter', 'refinery', 'concentrate'
  ];

  return miningKeywords.some(keyword => text.includes(keyword));
}

async function enrichWithAI(headline: FactSetHeadline): Promise<{
  summary: string;
  commodities: string[];
  sentiment: string;
}> {
  const cleanBody = headline.storyBody?.replace(/<[^>]*>/g, '').substring(0, 1500) || '';

  const prompt = `Analyze this mining industry news:

Headline: ${headline.headlines}
Content: ${cleanBody}

Extract:
1. A clear 1-2 sentence summary
2. All mining commodities mentioned (Gold, Copper, Lithium, Silver, Nickel, Iron Ore, Cobalt, Zinc, Platinum, Rare Earth, etc.)
3. Sentiment (Positive, Negative, or Neutral) for the mining industry

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
        { role: 'system', content: 'You are a mining industry analyst. Extract structured information from mining news.' },
        { role: 'user', content: prompt }
      ],
      response_format: { type: 'json_object' },
      temperature: 0.3,
      max_tokens: 500
    });

    const content = response.choices[0]?.message?.content;
    if (!content) throw new Error('No AI response');

    const analysis = JSON.parse(content);
    return {
      summary: analysis.summary || headline.headlines || '',
      commodities: analysis.commodities || [],
      sentiment: analysis.sentiment || 'Neutral'
    };
  } catch (error: any) {
    return {
      summary: headline.headlines || '',
      commodities: [],
      sentiment: 'Neutral'
    };
  }
}

async function saveToSupabase(headline: FactSetHeadline, enrichment: any): Promise<boolean> {
  try {
    if (!headline.headlines || !headline.storyTime) {
      console.log(`  ⚠ Skipping - missing required fields`);
      return false;
    }

    // Check duplicate
    const { data: existing } = await supabase
      .from('news')
      .select('id')
      .eq('title', headline.headlines)
      .single();

    if (existing) {
      return false;
    }

    const { error } = await supabase
      .from('news')
      .insert({
        title: headline.headlines,
        urls: headline.url ? [headline.url] : null,
        source: 'FactSet StreetAccount',
        published_at: headline.storyTime,
        summary: enrichment.summary,
        commodities: enrichment.commodities.length > 0 ? enrichment.commodities : null,
        project_ids: null,
        sentiment: enrichment.sentiment,
        watchlist: false
      });

    if (error) {
      console.log(`  ✗ Error: ${error.message}`);
      return false;
    }

    console.log(`  ✓ Saved: ${headline.headlines.substring(0, 80)}...`);
    return true;
  } catch (error: any) {
    return false;
  }
}

async function main() {
  console.log('=== Mining News Population from FactSet StreetAccount ===\n');

  let totalFetched = 0;
  let totalMiningRelated = 0;
  let totalSaved = 0;

  // Fetch in multiple batches
  const batches = 5; // Fetch 2500 headlines total
  for (let batch = 0; batch < batches; batch++) {
    console.log(`\nBatch ${batch + 1}/${batches}: Fetching 500 headlines...`);

    const headlines = await fetchMiningHeadlines(500);
    totalFetched += headlines.length;

    console.log(`  Fetched ${headlines.length} headlines`);
    console.log(`  Filtering for mining-related news...`);

    // Filter for mining-related news
    const miningHeadlines: FactSetHeadline[] = [];
    for (const headline of headlines) {
      if (headline.headlines && await isMiningRelated(headline)) {
        miningHeadlines.push(headline);
      }
    }

    totalMiningRelated += miningHeadlines.length;
    console.log(`  Found ${miningHeadlines.length} mining-related headlines`);

    // Enrich and save
    for (const headline of miningHeadlines) {
      console.log(`\n  Processing: ${headline.headlines?.substring(0, 80)}...`);

      const enrichment = await enrichWithAI(headline);
      const saved = await saveToSupabase(headline, enrichment);

      if (saved) {
        totalSaved++;
        console.log(`    Commodities: ${enrichment.commodities.join(', ') || 'None'}`);
        console.log(`    Sentiment: ${enrichment.sentiment}`);
      }

      // Rate limit
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    console.log(`\n  Progress: ${totalSaved} saved so far`);

    // Wait between batches
    if (batch < batches - 1) {
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
  }

  console.log('\n=== Complete ===');
  console.log(`Total fetched: ${totalFetched}`);
  console.log(`Mining-related: ${totalMiningRelated}`);
  console.log(`Saved to database: ${totalSaved}`);
}

main().catch(console.error);
