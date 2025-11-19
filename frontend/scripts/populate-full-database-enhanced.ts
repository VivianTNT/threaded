import { createClient } from '@supabase/supabase-js';
import FirecrawlApp from '@mendable/firecrawl-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const firecrawlApiKey = process.env.FIRECRAWL_API_KEY!;

if (!supabaseUrl || !supabaseServiceKey || !firecrawlApiKey) {
  throw new Error('Missing environment variables');
}

const supabase = createClient(supabaseUrl, supabaseServiceKey, {
  auth: {
    autoRefreshToken: false,
    persistSession: false
  }
});

const firecrawl = new FirecrawlApp({ apiKey: firecrawlApiKey });

let stats = {
  companiesAdded: 0,
  projectsAdded: 0,
  newsAdded: 0
};

// Comprehensive sources with detailed mining company and project data
const dataSources = [
  // Detailed company listings with market caps
  { url: 'https://www.nasdaq.com/articles/5-biggest-asx-gold-mining-companies-2025', type: 'companies', focus: 'Gold' },
  { url: 'https://www.nasdaq.com/articles/asx-copper-mining-stocks-5-biggest-companies-2024', type: 'companies', focus: 'Copper' },
  { url: 'https://investingnews.com/daily/resource-investing/battery-metals-investing/lithium-investing/top-lithium-producers/', type: 'companies', focus: 'Lithium' },
  { url: 'https://investingnews.com/daily/resource-investing/battery-metals-investing/nickel-investing/top-nickel-producers/', type: 'companies', focus: 'Nickel' },
  { url: 'https://investingnews.com/daily/resource-investing/critical-metals-investing/rare-earth-investing/rare-earth-producing-countries/', type: 'companies', focus: 'Rare Earths' },
  { url: 'https://investingnews.com/top-graphite-miners-asx', type: 'companies', focus: 'Graphite' },
  { url: 'https://investingnews.com/daily/resource-investing/battery-metals-investing/cobalt-investing/top-cobalt-stocks/', type: 'companies', focus: 'Cobalt' },

  // Project listings with resource/reserve estimates
  { url: 'https://www.mining.com/featured-article/ranked-top-10-copper-mining-projects-in-the-world/', type: 'projects', focus: 'Copper' },
  { url: 'https://www.mining.com/featured-article/ranked-worlds-top-10-copper-mining-projects-2022/', type: 'projects', focus: 'Copper' },
  { url: 'https://www.mining.com/lithium-projects-key-to-the-race-to-secure-strategic-materials-report/', type: 'projects', focus: 'Lithium' },
  { url: 'https://www.mining.com/web/us-adds-10-more-mining-projects-to-fast-track-permitting-list/', type: 'projects', focus: 'Various' },
  { url: 'https://www.mining.com/web/trump-to-fast-track-permitting-for-10-mining-projects-across-us/', type: 'projects', focus: 'Various' },

  // News sources
  { url: 'https://www.mining.com/latest-news/', type: 'news', focus: 'Mining' },
  { url: 'https://www.northernminer.com/', type: 'news', focus: 'Mining' },
  { url: 'https://www.kitco.com/news/category/commodities/', type: 'news', focus: 'Commodities' },
];

async function scrapeWithRetry(url: string, schema: any, maxRetries: number = 2): Promise<any> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const result = await firecrawl.scrapeUrl(url, {
        formats: ['extract'],
        extract: { schema }
      });

      if (result.success && result.extract) {
        return result.extract;
      }

      if (attempt < maxRetries) {
        console.log(`  ⚠️  Attempt ${attempt} failed, retrying...`);
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    } catch (error: any) {
      if (attempt === maxRetries) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
  return null;
}

async function scrapeCompanies(url: string, focus: string): Promise<void> {
  console.log(`\n📊 Scraping companies from: ${url.substring(0, 80)}... (${focus})`);

  const schema = {
    type: 'object',
    properties: {
      companies: {
        type: 'array',
        description: 'Extract ALL mining companies with complete details including market cap, ticker, projects',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string', description: 'Full company name' },
            ticker: { type: 'string', description: 'Stock ticker symbol' },
            exchange: { type: 'string', description: 'Stock exchange (ASX, TSX, NYSE, LSE, etc.)' },
            country: { type: 'string', description: 'Country where headquartered' },
            market_cap: { type: 'string', description: 'Market capitalization value with unit (e.g., "$5.2 billion", "A$1.3 billion")' },
            description: { type: 'string', description: 'Company description and operations' },
            website: { type: 'string', description: 'Company website URL' },
            projects: {
              type: 'array',
              description: 'Major mining projects operated by this company',
              items: {
                type: 'object',
                properties: {
                  name: { type: 'string' },
                  location: { type: 'string' },
                  commodity: { type: 'string' },
                  stage: { type: 'string' },
                  status: { type: 'string' },
                  resources: { type: 'string', description: 'Resource estimate with grade and tonnage' },
                  reserves: { type: 'string', description: 'Reserve estimate with grade and tonnage' },
                  production: { type: 'string', description: 'Annual production if mentioned' },
                  ownership: { type: 'string' }
                }
              }
            }
          },
          required: ['name']
        }
      }
    }
  };

  try {
    const data = await scrapeWithRetry(url, schema);
    if (!data?.companies || data.companies.length === 0) {
      console.log('  ⚠️  No companies extracted');
      return;
    }

    console.log(`  📑 Extracted ${data.companies.length} companies`);

    for (const comp of data.companies) {
      if (!comp.name) continue;

      // Check if company already exists
      const { data: existing } = await supabase
        .from('companies')
        .select('id')
        .ilike('name', comp.name)
        .limit(1);

      let companyId: string;

      if (existing && existing.length > 0) {
        companyId = existing[0].id;
        console.log(`  ⏭️  Company exists: ${comp.name}`);
      } else {
        // Parse market cap
        let marketCap = null;
        if (comp.market_cap) {
          const match = comp.market_cap.match(/([\d,.]+)\s*(billion|million|B|M|bn|mn)/i);
          if (match) {
            const value = parseFloat(match[1].replace(/,/g, ''));
            const unit = match[2].toLowerCase();
            marketCap = unit.startsWith('b') ? value * 1000000000 : value * 1000000;
          }
        }

        const { data: newCompany, error } = await supabase
          .from('companies')
          .insert({
            name: comp.name,
            ticker: comp.ticker || null,
            exchange: comp.exchange || null,
            country: comp.country || null,
            website: comp.website || null,
            description: comp.description || `${comp.name} is a ${focus} mining company.`,
            market_cap: marketCap,
            urls: [url],
            watchlist: false
          })
          .select()
          .single();

        if (error || !newCompany) {
          console.log(`  ❌ Error adding company: ${comp.name}`);
          continue;
        }

        companyId = newCompany.id;
        stats.companiesAdded++;
        console.log(`  ✓ Added: ${comp.name}${marketCap ? ` (${comp.market_cap})` : ''}`);
      }

      // Add projects for this company
      if (comp.projects && Array.isArray(comp.projects)) {
        for (const proj of comp.projects) {
          if (!proj.name) continue;

          const commodities = proj.commodity
            ? proj.commodity.split(/[,/&]/).map((c: string) => c.trim()).filter((c: string) => c.length > 1)
            : [focus];

          const ownership = proj.ownership ? parseFloat(proj.ownership.replace('%', '')) : null;

          const { error: projError } = await supabase
            .from('projects')
            .insert({
              company_id: companyId,
              name: proj.name.includes('Project') || proj.name.includes('Mine') ? proj.name : `${proj.name} Project`,
              location: proj.location || null,
              stage: proj.stage || 'Unknown',
              commodities,
              resource_estimate: proj.resources || null,
              reserve_estimate: proj.reserves || null,
              ownership_percentage: ownership,
              status: proj.status || 'Active',
              description: `${proj.name} is a ${commodities.join('/')} project${proj.location ? ` in ${proj.location}` : ''}.`,
              urls: [url],
              watchlist: false
            });

          if (!projError) {
            stats.projectsAdded++;
            console.log(`    ✓ Project: ${proj.name}${proj.resources ? ` (${proj.resources})` : ''}`);
          }
        }
      }
    }
  } catch (error: any) {
    console.error(`  ❌ Error: ${error.message}`);
  }
}

async function scrapeProjects(url: string, focus: string): Promise<void> {
  console.log(`\n🏗️  Scraping projects from: ${url.substring(0, 80)}... (${focus})`);

  const schema = {
    type: 'object',
    properties: {
      projects: {
        type: 'array',
        description: 'Extract ALL mining projects with complete resource/reserve estimates and company details',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string', description: 'Project or mine name' },
            company: { type: 'string', description: 'Operating company name' },
            location: { type: 'string', description: 'Country and region' },
            commodity: { type: 'string', description: 'Commodities produced or targeted' },
            stage: { type: 'string', description: 'Development stage (exploration, development, production, etc.)' },
            status: { type: 'string', description: 'Current operational status' },
            resources: { type: 'string', description: 'Resource estimate with tonnage and grade (e.g., "500 Mt @ 0.5% Cu")' },
            reserves: { type: 'string', description: 'Reserve estimate with tonnage and grade' },
            production: { type: 'string', description: 'Annual production if mentioned' },
            ownership: { type: 'string', description: 'Company ownership percentage' },
            description: { type: 'string', description: 'Project details and description' }
          },
          required: ['name']
        }
      }
    }
  };

  try {
    const data = await scrapeWithRetry(url, schema);
    if (!data?.projects || data.projects.length === 0) {
      console.log('  ⚠️  No projects extracted');
      return;
    }

    console.log(`  📑 Extracted ${data.projects.length} projects`);

    for (const proj of data.projects) {
      if (!proj.name) continue;

      // Find or create company
      let companyId = null;
      if (proj.company) {
        const { data: existingCompany } = await supabase
          .from('companies')
          .select('id')
          .ilike('name', proj.company)
          .limit(1);

        if (existingCompany && existingCompany.length > 0) {
          companyId = existingCompany[0].id;
        } else {
          const { data: newCompany } = await supabase
            .from('companies')
            .insert({
              name: proj.company,
              description: `${proj.company} operates ${focus} mining projects.`,
              urls: [url],
              watchlist: false
            })
            .select()
            .single();

          if (newCompany) {
            companyId = newCompany.id;
            stats.companiesAdded++;
          }
        }
      }

      const commodities = proj.commodity
        ? proj.commodity.split(/[,/&]/).map((c: string) => c.trim()).filter((c: string) => c.length > 1)
        : [focus];

      const ownership = proj.ownership ? parseFloat(proj.ownership.replace('%', '')) : null;

      const { error } = await supabase
        .from('projects')
        .insert({
          company_id: companyId,
          name: proj.name.includes('Project') || proj.name.includes('Mine') ? proj.name : `${proj.name} Project`,
          location: proj.location || null,
          stage: proj.stage || 'Unknown',
          commodities,
          resource_estimate: proj.resources || null,
          reserve_estimate: proj.reserves || null,
          ownership_percentage: ownership,
          status: proj.status || 'Active',
          description: proj.description || `${proj.name} is a ${commodities.join('/')} project${proj.location ? ` in ${proj.location}` : ''}.`,
          urls: [url],
          watchlist: false
        });

      if (!error) {
        stats.projectsAdded++;
        console.log(`  ✓ Added: ${proj.name}${proj.resources ? ` (${proj.resources})` : ''}`);
      }
    }
  } catch (error: any) {
    console.error(`  ❌ Error: ${error.message}`);
  }
}

async function scrapeNews(url: string): Promise<void> {
  console.log(`\n📰 Scraping news from: ${url.substring(0, 80)}...`);

  const schema = {
    type: 'object',
    properties: {
      articles: {
        type: 'array',
        description: 'Extract latest 30-50 mining news articles',
        items: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            url: { type: 'string' },
            source: { type: 'string' },
            published_date: { type: 'string' },
            summary: { type: 'string' },
            commodities: { type: 'string' },
            companies: { type: 'string' },
            sentiment: { type: 'string', description: 'Positive, Negative, or Neutral' }
          },
          required: ['title']
        }
      }
    }
  };

  try {
    const data = await scrapeWithRetry(url, schema);
    if (!data?.articles || data.articles.length === 0) {
      console.log('  ⚠️  No articles extracted');
      return;
    }

    console.log(`  📑 Extracted ${data.articles.length} articles`);

    for (const article of data.articles) {
      if (!article.title || article.title.length < 15) continue;

      // Check for duplicates
      const { data: existing } = await supabase
        .from('news')
        .select('id')
        .eq('title', article.title)
        .limit(1);

      if (existing && existing.length > 0) continue;

      const commodities = article.commodities
        ? article.commodities.split(/[,/&]/).map((c: string) => c.trim()).filter((c: string) => c.length > 1)
        : [];

      const { error } = await supabase
        .from('news')
        .insert({
          title: article.title,
          urls: article.url ? [article.url] : [url],
          source: article.source || new URL(url).hostname.replace('www.', ''),
          published_at: article.published_date ? new Date(article.published_date).toISOString() : new Date().toISOString(),
          summary: article.summary || null,
          commodities,
          project_ids: [],
          sentiment: article.sentiment || 'Neutral',
          watchlist: false
        });

      if (!error) {
        stats.newsAdded++;
        console.log(`  ✓ ${article.title.substring(0, 60)}...`);
      }
    }
  } catch (error: any) {
    console.error(`  ❌ Error: ${error.message}`);
  }
}

async function main() {
  console.log('🚀 ENHANCED COMPREHENSIVE DATABASE POPULATION');
  console.log('═══════════════════════════════════════════════════════════════\n');
  console.log('Target: Populate with REAL data including:');
  console.log('  • Companies with market caps');
  console.log('  • Projects with resource/reserve estimates');
  console.log('  • News articles with commodities and sentiment\n');
  console.log('═══════════════════════════════════════════════════════════════\n');

  for (const source of dataSources) {
    if (source.type === 'companies') {
      await scrapeCompanies(source.url, source.focus);
    } else if (source.type === 'projects') {
      await scrapeProjects(source.url, source.focus);
    } else if (source.type === 'news') {
      await scrapeNews(source.url);
    }

    await new Promise(resolve => setTimeout(resolve, 4000));
  }

  const { count: companiesCount } = await supabase.from('companies').select('*', { count: 'exact', head: true });
  const { count: projectsCount } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  const { count: newsCount } = await supabase.from('news').select('*', { count: 'exact', head: true });

  console.log('\n\n═══════════════════════════════════════════════════════════════');
  console.log('✅ SCRAPING COMPLETE!');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`📊 Companies: ${companiesCount} (added: ${stats.companiesAdded})`);
  console.log(`📊 Projects: ${projectsCount} (added: ${stats.projectsAdded})`);
  console.log(`📊 News: ${newsCount} (added: ${stats.newsAdded})`);
  console.log('═══════════════════════════════════════════════════════════════\n');
}

main()
  .then(() => {
    console.log('🎉 Done!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('💥 Failed:', error);
    process.exit(1);
  });
