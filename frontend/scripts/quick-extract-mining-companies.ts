/**
 * Quick Mining Company Extraction Script
 *
 * Efficiently extracts mining companies using targeted Firecrawl searches
 * Run with: npx tsx scripts/quick-extract-mining-companies.ts
 */

import FirecrawlApp from '@mendable/firecrawl-js';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

interface MiningCompany {
  name: string;
  ticker?: string;
  exchange?: string;
  commodities?: string[];
  marketCap?: string;
  source: string;
}

const firecrawlApiKey = process.env.FIRECRAWL_API_KEY;
if (!firecrawlApiKey) {
  console.error('❌ FIRECRAWL_API_KEY not found');
  process.exit(1);
}

const firecrawl = new FirecrawlApp({ apiKey: firecrawlApiKey });

const extractCommodities = (text: string): string[] => {
  const commodities: Record<string, string> = {
    'gold': 'Gold', 'silver': 'Silver', 'copper': 'Copper',
    'lithium': 'Lithium', 'nickel': 'Nickel', 'cobalt': 'Cobalt',
    'zinc': 'Zinc', 'iron': 'Iron', 'coal': 'Coal',
    'uranium': 'Uranium', 'rare earth': 'REE', 'graphite': 'Graphite'
  };

  const lower = text.toLowerCase();
  return Object.entries(commodities)
    .filter(([key]) => lower.includes(key))
    .map(([, value]) => value);
};

async function searchCompanies(query: string, limit: number = 3): Promise<MiningCompany[]> {
  console.log(`\n🔎 ${query}`);

  try {
    const result = await firecrawl.search(query, {
      limit,
      scrapeOptions: { formats: ['markdown'], timeout: 20000 }
    });

    if (!result.success || !result.data) {
      console.log('   ⚠️  No results');
      return [];
    }

    const companies: MiningCompany[] = [];

    for (const item of result.data) {
      const title = item.title || '';
      const content = item.markdown || '';
      const url = item.url || '';

      // Extract company names from titles and content
      const companyPatterns = [
        /([A-Z][A-Za-z\s&\.-]{3,50})\s+(?:Mining|Minerals|Resources|Metals|Gold|Silver|Copper|Lithium)/gi,
        /([A-Z][A-Z\s&\.]{3,50})\s+\(([A-Z]{2,5})\)/g,
        /^([A-Z][A-Za-z\s&\.-]{5,50})$/gm
      ];

      for (const pattern of companyPatterns) {
        const matches = content.matchAll(pattern);
        for (const match of matches) {
          const name = match[1]?.trim();
          if (name && name.length >= 5 && name.length <= 50) {
            companies.push({
              name,
              ticker: match[2]?.trim(),
              commodities: extractCommodities(content),
              source: query
            });
          }
        }
      }

      // Also check title for company names
      const titleMatch = title.match(/([A-Z][A-Za-z\s&\.-]{5,50})(?:\s+(?:Mining|Ltd|Inc|Corp|PLC|Limited))/i);
      if (titleMatch) {
        companies.push({
          name: titleMatch[1].trim(),
          commodities: extractCommodities(content + ' ' + title),
          source: query
        });
      }
    }

    // Deduplicate
    const unique = Array.from(
      new Map(companies.map(c => [c.name.toLowerCase(), c])).values()
    );

    console.log(`   ✓ Found ${unique.length} companies`);
    return unique;

  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`);
    return [];
  }
}

async function scrapeMiningExchangePage(url: string, name: string): Promise<MiningCompany[]> {
  console.log(`\n🌐 Scraping ${name}`);

  try {
    const result = await firecrawl.scrapeUrl(url, {
      formats: ['markdown'],
      timeout: 30000,
      waitFor: 1500
    });

    if (!result.success || !result.markdown) {
      console.log('   ⚠️  Scrape failed');
      return [];
    }

    console.log(`   ✓ Downloaded ${Math.round(result.markdown.length / 1024)}KB`);

    const companies: MiningCompany[] = [];
    const lines = result.markdown.split('\n');

    for (const line of lines) {
      // ASX format: Company Name (TICKER)
      const asxMatch = line.match(/\|\s*([A-Z][A-Za-z\s&\.-]{5,50})\s*\|\s*([A-Z]{2,5})\s*\|/);
      if (asxMatch) {
        companies.push({
          name: asxMatch[1].trim(),
          ticker: asxMatch[2].trim(),
          exchange: 'ASX',
          source: name
        });
      }

      // Generic: TICKER - Company Name
      const genericMatch = line.match(/([A-Z]{2,5})\s*[-–]\s*([A-Z][A-Za-z\s&\.-]{5,50})/);
      if (genericMatch) {
        companies.push({
          name: genericMatch[2].trim(),
          ticker: genericMatch[1].trim(),
          source: name
        });
      }
    }

    const unique = Array.from(
      new Map(companies.map(c => [c.name.toLowerCase(), c])).values()
    );

    console.log(`   ✓ Extracted ${unique.length} companies`);
    return unique;

  } catch (error: any) {
    console.log(`   ❌ Error: ${error.message}`);
    return [];
  }
}

async function main() {
  console.log('🚀 MINING COMPANY EXTRACTION');
  console.log('='.repeat(60));

  const allCompanies: MiningCompany[] = [];

  // Quick targeted searches
  const searches = [
    'ASX listed gold mining companies Australia',
    'TSX mining companies Canada list',
    'largest copper mining companies world',
    'lithium mining companies stock market'
  ];

  for (const search of searches) {
    const companies = await searchCompanies(search, 3);
    allCompanies.push(...companies);
    await new Promise(r => setTimeout(r, 1500)); // Rate limit
  }

  // Try one direct scrape of a mining company database
  const dbCompanies = await scrapeMiningExchangePage(
    'https://www.mining.com/markets/',
    'Mining.com Markets'
  );
  allCompanies.push(...dbCompanies);

  // Deduplicate all companies
  const unique = Array.from(
    new Map(allCompanies.map(c => [c.name.toLowerCase(), c])).values()
  );

  console.log('\n' + '='.repeat(60));
  console.log(`📊 FOUND ${unique.length} UNIQUE MINING COMPANIES`);
  console.log('='.repeat(60));

  // Display companies
  unique.forEach((company, i) => {
    console.log(`\n${(i + 1).toString().padStart(3)}. ${company.name}`);
    if (company.ticker) console.log(`     Ticker: ${company.ticker}`);
    if (company.exchange) console.log(`     Exchange: ${company.exchange}`);
    if (company.commodities?.length) {
      console.log(`     Commodities: ${company.commodities.join(', ')}`);
    }
    console.log(`     Source: ${company.source}`);
  });

  console.log('\n' + '='.repeat(60));
  console.log('✅ COMPLETE');

  return unique;
}

main()
  .then(() => process.exit(0))
  .catch(err => {
    console.error('❌ Fatal:', err);
    process.exit(1);
  });
