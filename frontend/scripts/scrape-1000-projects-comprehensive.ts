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

// Comprehensive list of mining industry sources to scrape
const sourcesToScrape = [
  // Copper projects
  'https://www.mining.com/featured-article/ranked-top-10-copper-mining-projects-in-the-world/',
  'https://www.mining.com/featured-article/ranked-worlds-top-10-copper-mining-projects-2022/',

  // Lithium projects
  'https://www.mining.com/lithium-projects-key-to-the-race-to-secure-strategic-materials-report/',
  'https://investingnews.com/daily/resource-investing/battery-metals-investing/lithium-investing/top-lithium-producers/',

  // Gold projects
  'https://www.mining.com/featured-article/ranked-top-10-gold-mining-companies-of-2025/',
  'https://www.northernminer.com/top-stories/list-top-10-gold-mining-companies-of-2024/',

  // Mixed/Various critical minerals
  'https://www.mining.com/web/us-adds-10-more-mining-projects-to-fast-track-permitting-list/',
  'https://www.mining.com/web/trump-to-fast-track-permitting-for-10-mining-projects-across-us/',

  // Nickel/Battery metals
  'https://investingnews.com/daily/resource-investing/battery-metals-investing/nickel-investing/top-nickel-producers/',

  // Cobalt
  'https://investingnews.com/daily/resource-investing/battery-metals-investing/cobalt-investing/top-cobalt-stocks/',

  // Rare earth elements
  'https://investingnews.com/daily/resource-investing/critical-metals-investing/rare-earth-investing/rare-earth-producing-countries/',

  // Graphite
  'https://investingnews.com/daily/resource-investing/battery-metals-investing/graphite-investing/graphite-producers/',

  // General mining companies with projects
  'https://www.mining.com/top-50-biggest-mining-companies/',
  'https://miningdigital.com/top10/top-10-worlds-biggest-mining-companies',

  // Regional focuses
  'https://www.mining.com/web/australia-home-to-worlds-most-exciting-lithium-projects-study/',
  'https://www.northernminer.com/top-stories/critical-minerals-projects-gain-momentum-in-canada/',
];

let allProjects: MiningProject[] = [];
let processedUrls = new Set<string>();

async function scrapeProjectsFromUrl(url: string, attempt: number = 1): Promise<MiningProject[]> {
  const maxAttempts = 2;

  if (processedUrls.has(url)) {
    console.log(`  ⏭️  Skipping (already processed)`);
    return [];
  }

  console.log(`\n📰 [${allProjects.length} projects] Scraping: ${url.substring(0, 80)}...`);

  try {
    const result = await firecrawl.scrapeUrl(url, {
      formats: ['extract'],
      extract: {
        schema: {
          type: 'object',
          properties: {
            projects: {
              type: 'array',
              description: 'Extract ALL mining projects, mines, developments, and exploration projects mentioned in the article. Include company names, project names, locations, and any operational details.',
              items: {
                type: 'object',
                properties: {
                  name: {
                    type: 'string',
                    description: 'Name of the mining project, mine, or development'
                  },
                  company: {
                    type: 'string',
                    description: 'Operating company or companies'
                  },
                  location: {
                    type: 'string',
                    description: 'Country, state/province, or region'
                  },
                  commodity: {
                    type: 'string',
                    description: 'Commodities produced or targeted (e.g., copper, gold, lithium)'
                  },
                  stage: {
                    type: 'string',
                    description: 'Development stage: exploration, development, production, construction, feasibility, etc.'
                  },
                  status: {
                    type: 'string',
                    description: 'Current status: active, on hold, under development, operating, etc.'
                  },
                  resources: {
                    type: 'string',
                    description: 'Resource estimate if mentioned'
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
                    description: 'Any additional details about the project'
                  }
                },
                required: ['name']
              }
            }
          }
        }
      }
    });

    if (!result.success || !result.extract?.projects || result.extract.projects.length === 0) {
      if (attempt < maxAttempts) {
        console.log(`  ⚠️  No projects found, retrying (attempt ${attempt + 1})...`);
        await new Promise(resolve => setTimeout(resolve, 3000));
        return scrapeProjectsFromUrl(url, attempt + 1);
      }
      console.log(`  ⚠️  No projects extracted after ${maxAttempts} attempts`);
      processedUrls.add(url);
      return [];
    }

    const extractedProjects: any[] = result.extract.projects;
    console.log(`  📑 Extracted ${extractedProjects.length} raw projects`);

    const projects: MiningProject[] = extractedProjects
      .filter(p => p.name && p.name.length > 2 && !p.name.match(/^(the|a|an|inc|corp|ltd)$/i))
      .map(p => {
        const name = p.name.includes('Project') || p.name.includes('Mine') ? p.name : `${p.name} Project`;
        const commodities = p.commodity
          ? p.commodity.split(/[,/&]/).map((c: string) => c.trim()).filter((c: string) => c.length > 1)
          : ['Unknown'];

        return {
          name,
          location: p.location || 'Unknown',
          stage: p.stage || (p.status?.toLowerCase().includes('production') || p.status?.toLowerCase().includes('operating') ? 'Production' : 'Unknown'),
          commodities,
          resource_estimate: p.resources || null,
          reserve_estimate: p.reserves || null,
          ownership_percentage: p.ownership ? parseFloat(p.ownership.replace('%', '')) : null,
          status: p.status || 'Active',
          description: p.description || `${name} is a ${commodities.join('/')} mining project ${p.location ? `located in ${p.location}` : ''}.`.trim(),
          urls: [url],
          watchlist: false,
          company_id: null
        };
      });

    // Deduplicate within this batch
    const uniqueProjects = projects.filter((p, index, self) =>
      index === self.findIndex((t) => t.name.toLowerCase() === p.name.toLowerCase())
    );

    console.log(`  ✓ Processed ${uniqueProjects.length} unique valid projects`);
    processedUrls.add(url);
    return uniqueProjects;

  } catch (error: any) {
    console.error(`  ❌ Error: ${error.message || error}`);
    if (attempt < maxAttempts) {
      console.log(`  🔄 Retrying (attempt ${attempt + 1})...`);
      await new Promise(resolve => setTimeout(resolve, 5000));
      return scrapeProjectsFromUrl(url, attempt + 1);
    }
    processedUrls.add(url);
    return [];
  }
}

async function uploadBatch(projects: MiningProject[], batchNumber: number): Promise<number> {
  console.log(`\n💾 Uploading batch ${batchNumber} (${projects.length} projects)...`);

  const { data, error } = await supabase
    .from('projects')
    .insert(projects)
    .select();

  if (error) {
    console.error(`❌ Error uploading batch ${batchNumber}:`, error.message);
    return 0;
  }

  const uploaded = data?.length || 0;
  console.log(`✅ Batch ${batchNumber} uploaded successfully (${uploaded} projects)`);
  return uploaded;
}

async function scrapeAndUpload(): Promise<void> {
  console.log('🚀 Starting comprehensive scraping of actual mining projects...\n');
  console.log(`📋 Sources to scrape: ${sourcesToScrape.length}`);
  console.log('═══════════════════════════════════════════════════════════════\n');

  let totalUploaded = 0;
  let batchNumber = 0;
  const BATCH_SIZE = 100;

  for (const url of sourcesToScrape) {
    const projects = await scrapeProjectsFromUrl(url);

    if (projects.length > 0) {
      // Deduplicate against all collected projects
      const newProjects = projects.filter(p =>
        !allProjects.some(existing => existing.name.toLowerCase() === p.name.toLowerCase())
      );

      allProjects.push(...newProjects);
      console.log(`  📊 Total unique projects collected: ${allProjects.length}`);

      // Upload in batches of 100
      if (allProjects.length >= BATCH_SIZE) {
        const batch = allProjects.splice(0, BATCH_SIZE);
        batchNumber++;
        const uploaded = await uploadBatch(batch, batchNumber);
        totalUploaded += uploaded;
      }
    }

    // Delay between sources
    await new Promise(resolve => setTimeout(resolve, 3000));
  }

  // Upload remaining projects
  while (allProjects.length > 0) {
    const batch = allProjects.splice(0, BATCH_SIZE);
    batchNumber++;
    const uploaded = await uploadBatch(batch, batchNumber);
    totalUploaded += uploaded;
  }

  // Final count
  const { count: finalCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('\n═══════════════════════════════════════════════════════════════');
  console.log('✅ Scraping and upload complete!');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`📊 Sources scraped: ${processedUrls.size}/${sourcesToScrape.length}`);
  console.log(`📈 Projects uploaded: ${totalUploaded}`);
  console.log(`📊 Total in database: ${finalCount}`);
  console.log('═══════════════════════════════════════════════════════════════\n');
}

scrapeAndUpload()
  .then(() => {
    console.log('🎉 Complete!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('💥 Failed:', error);
    process.exit(1);
  });
