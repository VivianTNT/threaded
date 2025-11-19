#!/usr/bin/env tsx

/**
 * Populate thousands of mining news articles from FactSet StreetAccount
 * This script fetches historical news in batches and enriches with AI
 */

import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;

if (!supabaseUrl || !supabaseKey) {
  console.error('Missing Supabase credentials');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

interface BatchResult {
  fetched: number;
  enriched: number;
  saved: number;
}

async function fetchNewsBatch(options: {
  limit: number;
  startDate?: string;
  endDate?: string;
}): Promise<BatchResult> {
  try {
    const response = await fetch('http://localhost:3000/api/factset-news', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        limit: options.limit,
        sectors: ['Materials', 'Metals & Mining', 'Energy', 'Gold', 'Silver', 'Copper'],
        regions: ['North America', 'South America', 'Australia', 'Africa', 'Europe', 'Asia'],
        startDate: options.startDate,
        endDate: options.endDate
      })
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API error: ${error}`);
    }

    const result = await response.json();

    return {
      fetched: result.fetched || 0,
      enriched: result.enriched || 0,
      saved: result.saved || 0
    };
  } catch (error: any) {
    console.error('Error fetching batch:', error.message);
    return { fetched: 0, enriched: 0, saved: 0 };
  }
}

async function getDateRanges(daysBack: number, batchDays: number): Promise<Array<{ startDate: string; endDate: string }>> {
  const ranges: Array<{ startDate: string; endDate: string }> = [];
  const today = new Date();

  for (let i = 0; i < daysBack; i += batchDays) {
    const endDate = new Date(today);
    endDate.setDate(today.getDate() - i);

    const startDate = new Date(endDate);
    startDate.setDate(endDate.getDate() - batchDays);

    ranges.push({
      startDate: startDate.toISOString().split('T')[0],
      endDate: endDate.toISOString().split('T')[0]
    });
  }

  return ranges;
}

async function main() {
  console.log('=== FactSet StreetAccount News Population ===\n');

  // Target: Fetch news from the last 365 days (1 year)
  // Process in 30-day batches to avoid API limits
  const daysBack = 365;
  const batchDays = 30;
  const articlesPerBatch = 100;

  console.log(`Configuration:`);
  console.log(`- Days back: ${daysBack}`);
  console.log(`- Batch size: ${batchDays} days`);
  console.log(`- Articles per batch: ${articlesPerBatch}`);
  console.log('');

  const dateRanges = await getDateRanges(daysBack, batchDays);
  console.log(`Processing ${dateRanges.length} date ranges...\n`);

  let totalFetched = 0;
  let totalEnriched = 0;
  let totalSaved = 0;

  for (let i = 0; i < dateRanges.length; i++) {
    const range = dateRanges[i];
    console.log(`\n[${i + 1}/${dateRanges.length}] Fetching news from ${range.startDate} to ${range.endDate}...`);

    const result = await fetchNewsBatch({
      limit: articlesPerBatch,
      startDate: range.startDate,
      endDate: range.endDate
    });

    totalFetched += result.fetched;
    totalEnriched += result.enriched;
    totalSaved += result.saved;

    console.log(`  ✓ Fetched: ${result.fetched}`);
    console.log(`  ✓ Enriched: ${result.enriched}`);
    console.log(`  ✓ Saved: ${result.saved}`);
    console.log(`  → Total saved so far: ${totalSaved}`);

    // Rate limiting: Wait 2 seconds between batches
    if (i < dateRanges.length - 1) {
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
  console.log(`Total in database: ${count || 0}`);
}

main().catch(console.error);
