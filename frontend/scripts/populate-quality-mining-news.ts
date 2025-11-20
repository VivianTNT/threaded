#!/usr/bin/env tsx

/**
 * Populate High-Quality Mining News from FactSet StreetAccount
 * - Filters out low-quality summaries/recaps
 * - Downloads PDFs to news bucket
 * - Extracts actual news source
 * - AI enrichment for commodities and sentiment
 */

import { createClient } from '@supabase/supabase-js';
import { ApiClient, HeadlinesApi } from '@factset/sdk-streetaccountnews';
import OpenAI from 'openai';
import { config } from 'dotenv';
import path from 'path';
import fetch from 'node-fetch';

config({ path: path.join(__dirname, '..', '.env.local') });

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

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

// Low-quality patterns to filter out
const LOW_QUALITY_PATTERNS = [
  /StreetAccount Top Stories/i,
  /StreetAccount Summary/i,
  /StreetAccount.*Reads/i,
  /StreetAccount Crypto/i,
  /StreetAccount Scorecard/i,
  /Market Synopsis/i,
  /market recap/i,
  /Trading higher\/lower/i,
  /as of \d+:\d+ ET/i
];

// Mining-related keywords
const MINING_KEYWORDS = [
  'mining', 'mine', 'mines', 'mineral', 'minerals',
  'gold', 'silver', 'copper', 'lithium', 'nickel', 'cobalt', 'zinc', 'iron ore',
  'platinum', 'palladium', 'rare earth', 'molybdenum', 'manganese', 'uranium',
  'exploration', 'drill', 'drilling', 'ore', 'deposit', 'resource', 'reserve',
  'metallurgical', 'smelter', 'refinery', 'concentrate', 'mining company'
];

function isLowQuality(headline: string): boolean {
  return LOW_QUALITY_PATTERNS.some(pattern => pattern.test(headline));
}

function isMiningRelated(headline: FactSetHeadline): boolean {
  const text = `${headline.headlines || ''} ${headline.storyBody?.replace(/<[^>]*>/g, '').substring(0, 500) || ''}`.toLowerCase();
  return MINING_KEYWORDS.some(keyword => text.includes(keyword));
}

function extractSource(headline: string): string {
  // Try to extract source from patterns like "Company - Source" or "Title - Source (ticker)"
  const patterns = [
    / - ([A-Z][a-zA-Z\s&]+)(?:\s+\(|$)/,  // "Title - Bloomberg (ticker)"
    / - ([A-Z][a-zA-Z\s&]+)$/,             // "Title - Reuters"
    /\(([A-Z][a-zA-Z\s&]+)\)$/,            // "Title (The Information)"
  ];

  for (const pattern of patterns) {
    const match = headline.match(pattern);
    if (match && match[1]) {
      const source = match[1].trim();
      // Filter out common non-source patterns
      if (!source.match(/^\$?\d+/) && source.length > 2 && source.length < 50) {
        return source;
      }
    }
  }

  return 'FactSet StreetAccount';
}

async function fetchHeadlines(predefinedRange: string, limit: number): Promise<FactSetHeadline[]> {
  const apiClient = ApiClient.instance;
  const factSetApiKey = apiClient.authentications['FactSetApiKey'];
  factSetApiKey.username = FACTSET_USERNAME;
  factSetApiKey.password = FACTSET_PASSWORD;

  const headlinesApi = new HeadlinesApi();

  const requestBody: any = {
    data: { predefinedRange },
    meta: {
      pagination: { limit: Math.min(limit, 500), offset: 0 }
    }
  };

  const response = await headlinesApi.getStreetAccountHeadlines(requestBody);
  return response && response.data ? response.data : [];
}

async function downloadPDF(url: string, storyId: string): Promise<string | null> {
  try {
    console.log(`    Downloading PDF from FactSet...`);

    const authHeader = 'Basic ' + Buffer.from(`${FACTSET_USERNAME}:${FACTSET_PASSWORD}`).toString('base64');

    const response = await fetch(url, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf, text/html'
      }
    });

    if (!response.ok) {
      console.log(`      ✗ Download failed: ${response.status}`);
      return null;
    }

    const buffer = await response.buffer();
    const contentType = response.headers.get('content-type') || 'text/html';
    const extension = contentType.includes('pdf') ? 'pdf' : 'html';

    // Upload to Supabase storage
    const storagePath = `factset/${storyId}.${extension}`;

    // Force text/html to application/pdf for FactSet HTML content
    const uploadContentType = contentType.includes('html') ? 'application/pdf' : contentType;

    const { data, error } = await supabase.storage
      .from('news')
      .upload(storagePath, buffer, {
        contentType: uploadContentType,
        upsert: true
      });

    if (error) {
      console.log(`      ✗ Upload failed: ${error.message}`);
      return null;
    }

    // Get public URL
    const { data: publicUrlData } = supabase.storage
      .from('news')
      .getPublicUrl(storagePath);

    console.log(`      ✓ PDF saved to storage`);
    return publicUrlData.publicUrl;
  } catch (error: any) {
    console.log(`      ✗ Error: ${error.message}`);
    return null;
  }
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
1. A concise 1-2 sentence summary (focus on the key news, not the stock price)
2. All mining commodities mentioned (Gold, Copper, Lithium, Silver, Nickel, Iron Ore, Cobalt, Zinc, Platinum, Rare Earth, Uranium, Molybdenum, Manganese, etc.)
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

async function saveToSupabase(
  headline: FactSetHeadline,
  enrichment: any,
  source: string,
  pdfUrl: string | null
): Promise<boolean> {
  try {
    if (!headline.headlines || !headline.storyTime) {
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

    const urls = [];
    if (pdfUrl) urls.push(pdfUrl);
    if (headline.url) urls.push(headline.url);

    const { error } = await supabase
      .from('news')
      .insert({
        title: headline.headlines,
        urls: urls.length > 0 ? urls : null,
        source: source,
        published_at: headline.storyTime,
        summary: enrichment.summary,
        commodities: enrichment.commodities.length > 0 ? enrichment.commodities : null,
        project_ids: null,
        sentiment: enrichment.sentiment,
        watchlist: false
      });

    if (error) {
      console.log(`      ✗ Save error: ${error.message}`);
      return false;
    }

    return true;
  } catch (error: any) {
    return false;
  }
}

async function main() {
  console.log('=== High-Quality Mining News Population ===\n');

  const dateRanges = ['today', 'twoDays', 'oneWeek', 'oneMonth'];
  let totalFetched = 0;
  let totalFiltered = 0;
  let totalMining = 0;
  let totalSaved = 0;

  for (const range of dateRanges) {
    console.log(`\n--- Fetching ${range} headlines ---`);

    const headlines = await fetchHeadlines(range, 500);
    totalFetched += headlines.length;
    console.log(`  Fetched: ${headlines.length}`);

    // Filter out low-quality headlines
    const qualityHeadlines = headlines.filter(h =>
      h.headlines && !isLowQuality(h.headlines)
    );
    totalFiltered += headlines.length - qualityHeadlines.length;
    console.log(`  After quality filter: ${qualityHeadlines.length}`);

    // Filter for mining-related
    const miningHeadlines = qualityHeadlines.filter(isMiningRelated);
    totalMining += miningHeadlines.length;
    console.log(`  Mining-related: ${miningHeadlines.length}\n`);

    // Process each headline
    for (const headline of miningHeadlines.slice(0, 100)) { // Limit to 100 per range
      const title = headline.headlines!.substring(0, 100);
      console.log(`  Processing: ${title}...`);

      // Extract source
      const source = extractSource(headline.headlines!);
      console.log(`    Source: ${source}`);

      // Download PDF
      const pdfUrl = headline.url ? await downloadPDF(headline.url, headline.id || 'unknown') : null;

      // AI enrichment
      const enrichment = await enrichWithAI(headline);
      console.log(`    Commodities: ${enrichment.commodities.join(', ') || 'None'}`);
      console.log(`    Sentiment: ${enrichment.sentiment}`);

      // Save
      const saved = await saveToSupabase(headline, enrichment, source, pdfUrl);
      if (saved) {
        totalSaved++;
        console.log(`    ✓ Saved (#${totalSaved})\n`);
      } else {
        console.log(`    ⚠ Skipped (duplicate or error)\n`);
      }

      // Rate limit
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }

  console.log('\n=== Complete ===');
  console.log(`Total fetched: ${totalFetched}`);
  console.log(`Filtered out (low quality): ${totalFiltered}`);
  console.log(`Mining-related: ${totalMining}`);
  console.log(`Saved to database: ${totalSaved}`);
}

main().catch(console.error);
