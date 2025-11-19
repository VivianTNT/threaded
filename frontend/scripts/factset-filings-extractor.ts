import { createClient } from '@supabase/supabase-js';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const factsetUsername = process.env.FACTSET_USERNAME!;
const factsetApiKey = process.env.FACTSET_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

// Major mining companies with correct tickers
const MINING_COMPANIES = [
  // Major US-listed
  { ticker: 'NEM', name: 'Newmont Corporation', website: 'https://www.newmont.com', exchange: 'NYSE' },
  { ticker: 'GOLD', name: 'Barrick Gold', website: 'https://www.barrick.com', exchange: 'NYSE' },
  { ticker: 'FCX', name: 'Freeport-McMoRan', website: 'https://www.fcx.com', exchange: 'NYSE' },
  { ticker: 'BHP', name: 'BHP Group', website: 'https://www.bhp.com', exchange: 'NYSE' },
  { ticker: 'RIO', name: 'Rio Tinto', website: 'https://www.riotinto.com', exchange: 'NYSE' },
  { ticker: 'VALE', name: 'Vale S.A.', website: 'https://www.vale.com', exchange: 'NYSE' },
  { ticker: 'SCCO', name: 'Southern Copper', website: 'https://www.southerncoppercorp.com', exchange: 'NYSE' },
  { ticker: 'AEM', name: 'Agnico Eagle Mines', website: 'https://www.agnicoeagle.com', exchange: 'NYSE' },
  { ticker: 'KGC', name: 'Kinross Gold', website: 'https://www.kinross.com', exchange: 'NYSE' },
  { ticker: 'FNV', name: 'Franco-Nevada', website: 'https://www.franco-nevada.com', exchange: 'NYSE' },
  { ticker: 'WPM', name: 'Wheaton Precious Metals', website: 'https://www.wheatonpm.com', exchange: 'NYSE' },
  { ticker: 'RGLD', name: 'Royal Gold', website: 'https://www.royalgold.com', exchange: 'NASDAQ' },
  { ticker: 'PAAS', name: 'Pan American Silver', website: 'https://www.panamericansilver.com', exchange: 'NYSE' },
  { ticker: 'HL', name: 'Hecla Mining', website: 'https://www.hecla-mining.com', exchange: 'NYSE' },
  { ticker: 'CDE', name: 'Coeur Mining', website: 'https://www.coeur.com', exchange: 'NYSE' },
  { ticker: 'AG', name: 'First Majestic Silver', website: 'https://www.firstmajestic.com', exchange: 'NYSE' },
  { ticker: 'EGO', name: 'Eldorado Gold', website: 'https://www.eldoradogold.com', exchange: 'NYSE' },
  { ticker: 'IAG', name: 'IAMGOLD', website: 'https://www.iamgold.com', exchange: 'NYSE' },
  { ticker: 'NGD', name: 'New Gold', website: 'https://www.newgold.com', exchange: 'NYSE' },
  { ticker: 'BTG', name: 'B2Gold', website: 'https://www.b2gold.com', exchange: 'NYSE' },
  { ticker: 'SSRM', name: 'SSR Mining', website: 'https://www.ssrmining.com', exchange: 'NASDAQ' },
  { ticker: 'AU', name: 'AngloGold Ashanti', website: 'https://www.anglogoldashanti.com', exchange: 'NYSE' },
  { ticker: 'HMY', name: 'Harmony Gold', website: 'https://www.harmony.co.za', exchange: 'NYSE' },
  { ticker: 'SBSW', name: 'Sibanye Stillwater', website: 'https://www.sibanyestillwater.com', exchange: 'NYSE' },
  { ticker: 'TECK', name: 'Teck Resources', website: 'https://www.teck.com', exchange: 'NYSE' },
  { ticker: 'AA', name: 'Alcoa Corporation', website: 'https://www.alcoa.com', exchange: 'NYSE' },
  { ticker: 'MP', name: 'MP Materials', website: 'https://www.mpmaterials.com', exchange: 'NYSE' },
  { ticker: 'LAC', name: 'Lithium Americas', website: 'https://www.lithiumamericas.com', exchange: 'NYSE' },
  { ticker: 'ALB', name: 'Albemarle Corporation', website: 'https://www.albemarle.com', exchange: 'NYSE' },
  { ticker: 'SQM', name: 'Sociedad Química y Minera', website: 'https://www.sqm.com', exchange: 'NYSE' },
  { ticker: 'PLL', name: 'Piedmont Lithium', website: 'https://www.piedmontlithium.com', exchange: 'NYSE' },
  { ticker: 'CCJ', name: 'Cameco Corporation', website: 'https://www.cameco.com', exchange: 'NYSE' },
  { ticker: 'DNN', name: 'Denison Mines', website: 'https://www.denisonmines.com', exchange: 'NYSE' },
  { ticker: 'NXE', name: 'NexGen Energy', website: 'https://www.nexgenenergy.ca', exchange: 'NYSE' },
  { ticker: 'UUUU', name: 'Energy Fuels', website: 'https://www.energyfuels.com', exchange: 'NYSE' },
  { ticker: 'UEC', name: 'Uranium Energy Corp', website: 'https://www.uraniumenergy.com', exchange: 'NYSE' },
];

// Search for documents on SEDAR+ using company name
async function searchSEDAR(companyName: string): Promise<any[]> {
  try {
    const searchTerms = [
      'technical report',
      'NI 43-101',
      'feasibility study',
      'preliminary economic assessment',
      'PEA',
      'mineral resource',
      'mineral reserve'
    ];

    const results: any[] = [];

    for (const term of searchTerms) {
      try {
        // Using FactSet Filings API to search SEDAR
        const response = await axios.get(
          `https://api.factset.com/content/factset-global-filings/v1/filings`,
          {
            params: {
              sources: 'SDR', // SEDAR
              search: `"${companyName}" AND "${term}"`,
              limit: 10,
              sort: '-filingDate'
            },
            auth: {
              username: factsetUsername,
              password: factsetApiKey
            }
          }
        );

        if (response.data?.data) {
          results.push(...response.data.data);
        }
      } catch (error) {
        console.error(`SEDAR search failed for "${companyName}" + "${term}":`, error);
      }
    }

    return results;
  } catch (error) {
    console.error(`Failed to search SEDAR for ${companyName}:`, error);
    return [];
  }
}

// Search for documents on SEC EDGAR
async function searchEDGAR(ticker: string): Promise<any[]> {
  try {
    const formTypes = ['8-K', '10-K', '10-Q', '6-K', '20-F', '40-F'];
    const results: any[] = [];

    for (const formType of formTypes) {
      try {
        const response = await axios.get(
          `https://api.factset.com/content/factset-global-filings/v1/filings`,
          {
            params: {
              sources: 'EDG', // EDGAR
              tickers: ticker,
              formTypes: formType,
              limit: 5,
              sort: '-filingDate'
            },
            auth: {
              username: factsetUsername,
              password: factsetApiKey
            }
          }
        );

        if (response.data?.data) {
          // Filter for documents mentioning mining projects
          const miningDocs = response.data.data.filter((doc: any) => {
            const title = (doc.title || '').toLowerCase();
            const desc = (doc.description || '').toLowerCase();
            return title.includes('mining') || title.includes('mineral') ||
                   title.includes('resource') || title.includes('technical') ||
                   desc.includes('mining') || desc.includes('mineral');
          });
          results.push(...miningDocs);
        }
      } catch (error) {
        console.error(`EDGAR search failed for ${ticker} form ${formType}:`, error);
      }
    }

    return results;
  } catch (error) {
    console.error(`Failed to search EDGAR for ${ticker}:`, error);
    return [];
  }
}

// Download document
async function downloadDocument(documentUrl: string): Promise<Buffer | null> {
  try {
    const response = await axios.get(documentUrl, {
      responseType: 'arraybuffer',
      maxContentLength: 100 * 1024 * 1024, // 100MB limit
      timeout: 60000 // 60 second timeout
    });

    return Buffer.from(response.data);
  } catch (error) {
    console.error(`Failed to download document:`, error);
    return null;
  }
}

// Upload to Supabase storage
async function uploadToSupabase(buffer: Buffer, fileName: string): Promise<string | null> {
  try {
    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(`mining-reports/${fileName}`, buffer, {
        contentType: 'application/pdf',
        upsert: true
      });

    if (error) {
      console.error('Upload error:', error);
      return null;
    }

    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(`mining-reports/${fileName}`);

    return urlData?.publicUrl || null;
  } catch (error) {
    console.error('Upload failed:', error);
    return null;
  }
}

// Get market cap from Yahoo Finance or similar
async function getMarketCap(ticker: string): Promise<number | null> {
  try {
    // Simple approach - you might want to use a proper financial API
    const response = await axios.get(
      `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}`,
      { timeout: 5000 }
    );

    const marketCap = response.data?.chart?.result?.[0]?.meta?.marketCap;
    return marketCap || null;
  } catch (error) {
    console.error(`Failed to get market cap for ${ticker}:`, error);
    return null;
  }
}

async function processCompany(company: typeof MINING_COMPANIES[0]) {
  console.log(`\nProcessing ${company.name} (${company.ticker})...`);

  try {
    // Get market cap
    const marketCap = await getMarketCap(company.ticker);

    // Insert company
    const { data: companyData, error: companyError } = await supabase
      .from('companies')
      .insert({
        name: company.name,
        ticker: company.ticker,
        website: company.website,
        market_cap: marketCap,
        exchange: company.exchange,
        sector: 'Mining',
        subsector: 'Metals & Mining',
        created_at: new Date().toISOString()
      })
      .select()
      .single();

    if (companyError) {
      console.error(`Failed to insert company:`, companyError);
      return;
    }

    console.log(`✓ Added company: ${company.name} (Market Cap: $${marketCap ? (marketCap / 1e9).toFixed(2) + 'B' : 'N/A'})`);

    // Search for documents
    const sedarDocs = await searchSEDAR(company.name);
    const edgarDocs = await searchEDGAR(company.ticker);

    const allDocs = [...sedarDocs, ...edgarDocs];
    console.log(`Found ${allDocs.length} documents`);

    // Process documents (limit to top 3 per company for demo)
    for (const doc of allDocs.slice(0, 3)) {
      try {
        if (!doc.documentUrl) continue;

        const fileName = `${company.ticker}_${doc.accessionNo || doc.filingId}_${Date.now()}.pdf`;

        // Download document
        console.log(`  Downloading: ${doc.title || 'Document'}...`);
        const buffer = await downloadDocument(doc.documentUrl);
        if (!buffer) continue;

        // Upload to Supabase
        const storageUrl = await uploadToSupabase(buffer, fileName);
        if (!storageUrl) continue;

        // Create project entry
        const projectName = doc.title?.replace(/technical report/i, '').trim() ||
                          `${company.name} Project`;

        const { error: projectError } = await supabase
          .from('projects')
          .insert({
            name: projectName,
            company_id: companyData.id,
            location: 'TBD - Extract from document',
            commodities: ['Gold', 'Silver', 'Copper'], // Placeholder
            stage: 'development',
            document_urls: [doc.documentUrl],
            document_storage_path: storageUrl,
            technical_report_date: doc.filingDate,
            source: doc.source === 'EDG' ? 'EDGAR' : 'SEDAR',
            extraction_date: new Date().toISOString()
          });

        if (projectError) {
          console.error(`Failed to insert project:`, projectError);
        } else {
          console.log(`  ✓ Added project: ${projectName}`);
        }

        // Rate limiting
        await new Promise(resolve => setTimeout(resolve, 2000));
      } catch (error) {
        console.error(`Failed to process document:`, error);
      }
    }
  } catch (error) {
    console.error(`Failed to process company ${company.name}:`, error);
  }
}

async function main() {
  console.log('Starting mining data extraction...');
  console.log(`Processing ${MINING_COMPANIES.length} companies`);

  // Process in batches of 3
  const batchSize = 3;
  for (let i = 0; i < MINING_COMPANIES.length; i += batchSize) {
    const batch = MINING_COMPANIES.slice(i, i + batchSize);

    await Promise.all(batch.map(processCompany));

    console.log(`\nCompleted batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(MINING_COMPANIES.length / batchSize)}`);

    // Delay between batches
    if (i + batchSize < MINING_COMPANIES.length) {
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }

  // Summary
  const { count: companies } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true });

  const { count: projects } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('\n=== Extraction Complete ===');
  console.log(`Total companies: ${companies}`);
  console.log(`Total projects: ${projects}`);
}

main().catch(console.error);