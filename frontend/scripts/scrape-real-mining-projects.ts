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

interface MiningProject {
  company_id?: string | null;
  name: string;
  location: string;
  stage: string;
  commodities: string[];
  resource_estimate?: string | null;
  reserve_estimate?: string | null;
  ownership_percentage?: number | null;
  status: string;
  description?: string | null;
  urls: string[];
  watchlist: boolean;
}

// URLs to scrape actual mining project data from
const sourcesToScrape = [
  {
    url: 'https://www.mining.com/featured-article/ranked-top-10-copper-mining-projects-in-the-world/',
    commodity: 'Copper',
    description: 'Top 10 copper projects worldwide'
  },
  {
    url: 'https://www.mining.com/lithium-projects-key-to-the-race-to-secure-strategic-materials-report/',
    commodity: 'Lithium',
    description: 'Key lithium mining projects'
  },
  {
    url: 'https://www.mining.com/featured-article/ranked-top-10-gold-mining-companies-of-2025/',
    commodity: 'Gold',
    description: 'Top gold mining companies and projects'
  },
  {
    url: 'https://www.mining.com/web/us-adds-10-more-mining-projects-to-fast-track-permitting-list/',
    commodity: 'Various',
    description: 'US fast-track mining projects'
  },
];

async function scrapeProjectsFromSource(sourceUrl: string, primaryCommodity: string): Promise<MiningProject[]> {
  console.log(`\n📰 Scraping: ${sourceUrl}`);
  console.log(`   Commodity focus: ${primaryCommodity}`);

  try {
    const result = await firecrawl.scrapeUrl(sourceUrl, {
      formats: ['extract'],
      extract: {
        schema: {
          type: 'object',
          properties: {
            projects: {
              type: 'array',
              description: 'List of mining projects mentioned in the article',
              items: {
                type: 'object',
                properties: {
                  name: {
                    type: 'string',
                    description: 'Name of the mining project (e.g., "Resolution Copper", "Oyu Tolgoi", "Escondida")'
                  },
                  company: {
                    type: 'string',
                    description: 'Company or companies operating the project'
                  },
                  location: {
                    type: 'string',
                    description: 'Geographic location (country, state, or region)'
                  },
                  commodity: {
                    type: 'string',
                    description: 'Primary commodity or commodities (copper, lithium, gold, etc.)'
                  },
                  stage: {
                    type: 'string',
                    description: 'Development stage (exploration, development, production, etc.)'
                  },
                  status: {
                    type: 'string',
                    description: 'Current status (active, on hold, under construction, etc.)'
                  },
                  resources: {
                    type: 'string',
                    description: 'Resource estimate if mentioned (e.g., "100 Mt @ 0.5% copper")'
                  },
                  reserves: {
                    type: 'string',
                    description: 'Reserve estimate if mentioned'
                  },
                  ownership: {
                    type: 'string',
                    description: 'Ownership percentage if mentioned'
                  },
                  description: {
                    type: 'string',
                    description: 'Brief description of the project'
                  }
                },
                required: ['name']
              }
            }
          }
        }
      }
    });

    if (!result.success || !result.extract?.projects) {
      console.log(`  ⚠️  No projects extracted from this source`);
      return [];
    }

    const extractedProjects: any[] = result.extract.projects;
    console.log(`  📑 Extracted ${extractedProjects.length} projects`);

    const projects: MiningProject[] = extractedProjects
      .filter(p => p.name && p.name.length > 3)
      .map(p => ({
        name: p.name.includes('Project') ? p.name : `${p.name} Project`,
        location: p.location || 'Unknown',
        stage: p.stage || 'Unknown',
        commodities: p.commodity ? p.commodity.split(/[,/]/).map((c: string) => c.trim()) : [primaryCommodity],
        resource_estimate: p.resources || null,
        reserve_estimate: p.reserves || null,
        ownership_percentage: p.ownership ? parseFloat(p.ownership) : null,
        status: p.status || 'Active',
        description: p.description || `${p.name} is a ${primaryCommodity} mining project located in ${p.location || 'an undisclosed location'}.`,
        urls: [sourceUrl],
        watchlist: false,
        company_id: null
      }));

    console.log(`  ✓ Processed ${projects.length} valid projects`);
    return projects;

  } catch (error) {
    console.error(`  ❌ Error scraping ${sourceUrl}:`, error);
    return [];
  }
}

async function scrapeAllProjects(): Promise<void> {
  console.log('🚀 Starting to scrape actual mining projects from the web...\n');
  console.log('═══════════════════════════════════════════════════════════════\n');

  const allProjects: MiningProject[] = [];

  // Scrape from each source
  for (const source of sourcesToScrape) {
    const projects = await scrapeProjectsFromSource(source.url, source.commodity);
    allProjects.push(...projects);

    // Delay between sources to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log(`\n📊 Total projects scraped: ${allProjects.length}`);

  if (allProjects.length === 0) {
    console.log('\n⚠️  No projects were scraped. This may be due to:');
    console.log('   - Website structure changes');
    console.log('   - Firecrawl API rate limits');
    console.log('   - Network issues');
    console.log('\nTry running the script again or check the source URLs manually.');
    return;
  }

  // Upload to Supabase in batches
  console.log('\n💾 Uploading projects to Supabase...\n');

  const BATCH_SIZE = 50;
  let uploaded = 0;

  for (let i = 0; i < allProjects.length; i += BATCH_SIZE) {
    const batch = allProjects.slice(i, i + BATCH_SIZE);
    const batchNum = Math.floor(i / BATCH_SIZE) + 1;

    console.log(`📦 Uploading batch ${batchNum} (${batch.length} projects)...`);

    const { data, error } = await supabase
      .from('projects')
      .insert(batch)
      .select();

    if (error) {
      console.error(`❌ Error uploading batch ${batchNum}:`, error);
      continue;
    }

    uploaded += data?.length || 0;
    console.log(`✅ Batch ${batchNum} uploaded (${data?.length} projects)\n`);
  }

  // Get final count
  const { count: finalCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('\n✅ Upload complete!');
  console.log('═══════════════════════════════════════');
  console.log(`📊 Projects in database: ${finalCount}`);
  console.log(`📈 Projects uploaded: ${uploaded}`);
  console.log('═══════════════════════════════════════\n');
}

scrapeAllProjects()
  .then(() => {
    console.log('🎉 Scraping complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('💥 Failed:', error);
    process.exit(1);
  });
