/**
 * FactSet Global Filings Extractor for Mining Companies
 *
 * Extracts technical documents (100+ pages) from SEDAR and SEC/EDGAR
 * for major mining companies and uploads to Supabase storage.
 *
 * Based on FactSet Global Filings API V2
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

// FactSet Credentials
const FACTSET_CONFIG = {
  username: 'LITHOS-2220379',
  apiKey: 'f69RjKOALrE2921T9x8PWTr4chAPmkhcWBEpZdOI',
  baseUrl: 'https://api.factset.com/content/global-filings/v2'
};

// Supabase
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Top 10 Mining Company Tickers
const MINING_TICKERS = [
  { ticker: 'FCX-US', name: 'Freeport-McMoRan', commodity: 'Copper/Gold' },
  { ticker: 'VALE-US', name: 'Vale', commodity: 'Iron Ore' },
  { ticker: 'RIO-GB', name: 'Rio Tinto', commodity: 'Diversified' },
  { ticker: 'BHP-US', name: 'BHP Group', commodity: 'Diversified' },
  { ticker: 'NEM-US', name: 'Newmont', commodity: 'Gold' },
  { ticker: 'TECK.B-CA', name: 'Teck Resources', commodity: 'Copper/Coal' },
  { ticker: 'ABX-CA', name: 'Barrick Gold', commodity: 'Gold' },
  { ticker: 'FM-CA', name: 'First Quantum', commodity: 'Copper' },
  { ticker: 'SCCO-US', name: 'Southern Copper', commodity: 'Copper' },
  { ticker: 'GOLD-US', name: 'Barrick Gold (US)', commodity: 'Gold' }
];

interface DocumentMetadata {
  documentId: string;
  headline: string;
  source: string;
  ticker: string;
  companyName: string;
  filingsDateTime: string;
  formTypes: string[];
  filingSize: string;
  filingsLink: string;
  estimatedPages?: number;
  actualPages?: number;
}

const auth = Buffer.from(`${FACTSET_CONFIG.username}:${FACTSET_CONFIG.apiKey}`).toString('base64');

async function factsetAPI(endpoint: string, params?: Record<string, any>) {
  const url = new URL(`${FACTSET_CONFIG.baseUrl}${endpoint}`);

  if (params) {
    Object.keys(params).forEach(key => {
      const value = params[key];
      if (Array.isArray(value)) {
        value.forEach(v => url.searchParams.append(key, v));
      } else if (value !== undefined && value !== null) {
        url.searchParams.append(key, value);
      }
    });
  }

  const response = await fetch(url.toString(), {
    headers: {
      'Authorization': `Basic ${auth}`,
      'Accept': 'application/json'
    }
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`FactSet API Error ${response.status}: ${error}`);
  }

  return response.json();
}

function estimatePagesFromSize(fileSizeStr: string): number {
  // Parse size string like "12MB", "1.5MB"
  const match = fileSizeStr.match(/([\d.]+)\s*MB/i);
  if (!match) return 0;

  const sizeMB = parseFloat(match[1]);

  // Estimation: 1 page ≈ 50KB for text-heavy PDFs
  // So 5MB ≈ 100 pages
  return Math.round(sizeMB * 20);
}

function getPDFPageCount(filepath: string): number {
  try {
    const result = execSync(`pdfinfo "${filepath}"`, { encoding: 'utf-8' });
    const match = result.match(/Pages:\s+(\d+)/);
    return match ? parseInt(match[1]) : 0;
  } catch (error) {
    console.log(`   ⚠️  Could not get page count: ${error}`);
    return 0;
  }
}

async function searchFilings(ticker: string, companyName: string) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`🔍 Searching: ${companyName} (${ticker})`);
  console.log('='.repeat(70));

  try {
    const response = await factsetAPI('/search', {
      ids: [ticker],
      sources: ['SDR', 'SDRP', 'EDG'], // SEDAR + SEDAR+ + EDGAR
      startDate: '20200101', // Last 5 years
      endDate: '20251101',
      _paginationLimit: 100,
      _paginationOffset: 0,
      _sort: ['-filingsDateTime'], // Newest first
      primaryId: true // Only primary company documents
    });

    if (!response.data || response.data.length === 0) {
      console.log(`   ℹ️  No data returned`);
      return [];
    }

    const documents: DocumentMetadata[] = [];

    for (const entry of response.data) {
      if (entry.error) {
        console.log(`   ❌ Error: ${entry.error.detail}`);
        continue;
      }

      const docs = entry.documents || [];
      console.log(`   📄 Found ${docs.length} documents`);

      for (const doc of docs) {
        const estimatedPages = estimatePagesFromSize(doc.filingSize || '0MB');

        // Filter: Only documents estimated to be 100+ pages
        if (estimatedPages < 100) {
          continue;
        }

        documents.push({
          documentId: doc.documentId,
          headline: doc.headline,
          source: doc.source,
          ticker: ticker,
          companyName: companyName,
          filingsDateTime: doc.filingsDateTime,
          formTypes: doc.formTypes || [],
          filingSize: doc.filingSize || '0MB',
          filingsLink: doc.filingsLink,
          estimatedPages: estimatedPages
        });
      }
    }

    console.log(`   ✅ Filtered to ${documents.length} documents (estimated 100+ pages)`);
    return documents;

  } catch (error: any) {
    console.log(`   ❌ Search failed: ${error.message}`);
    return [];
  }
}

async function downloadDocument(doc: DocumentMetadata, outputDir: string): Promise<string | null> {
  console.log(`\n📥 ${doc.headline.substring(0, 80)}...`);
  console.log(`   Source: ${doc.source} | Est. Pages: ${doc.estimatedPages} | Size: ${doc.filingSize}`);

  try {
    // Download with authentication
    const response = await fetch(doc.filingsLink, {
      headers: {
        'Authorization': `Basic ${auth}`
      }
    });

    if (!response.ok) {
      console.log(`   ❌ Download failed: ${response.status}`);
      return null;
    }

    const buffer = Buffer.from(await response.arrayBuffer());
    console.log(`   ✅ Downloaded ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);

    // Save temporarily
    const tempPath = path.join(outputDir, `${doc.documentId}.html`);
    fs.writeFileSync(tempPath, buffer);

    return tempPath;

  } catch (error: any) {
    console.log(`   ❌ Download error: ${error.message}`);
    return null;
  }
}

async function uploadToSupabase(
  filepath: string,
  doc: DocumentMetadata
): Promise<string | null> {
  try {
    const buffer = fs.readFileSync(filepath);
    const ext = path.extname(filepath);

    const cleanCompany = doc.companyName.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
    const year = new Date(doc.filingsDateTime).getFullYear();

    const storagePath = `factset/${doc.source.toLowerCase()}/${cleanCompany}/${year}/${doc.documentId}${ext}`;

    console.log(`   ☁️  Uploading to: ${storagePath}`);

    const { data, error } = await supabase.storage
      .from('refinitiv')
      .upload(storagePath, buffer, {
        contentType: ext === '.pdf' ? 'application/pdf' : 'text/html',
        upsert: true
      });

    if (error) {
      console.log(`   ❌ Upload error: ${error.message}`);
      return null;
    }

    const { data: urlData } = supabase.storage
      .from('refinitiv')
      .getPublicUrl(storagePath);

    console.log(`   ✅ Uploaded: ${urlData.publicUrl}`);

    return urlData.publicUrl;

  } catch (error: any) {
    console.log(`   ❌ Upload failed: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('🚀 FactSet Mining Document Extractor\n');
  console.log('📋 Target: 100+ page technical documents from SEDAR and SEC/EDGAR');
  console.log(`📊 Processing ${MINING_TICKERS.length} mining companies\n`);

  const outputDir = './downloads/factset';
  fs.mkdirSync(outputDir, { recursive: true });

  const results: Array<{
    ticker: string;
    companyName: string;
    documentsFound: number;
    documentsUploaded: number;
    urls: string[];
  }> = [];

  for (const company of MINING_TICKERS) {
    // Search for documents
    const documents = await searchFilings(company.ticker, company.name);

    if (documents.length === 0) {
      results.push({
        ticker: company.ticker,
        companyName: company.name,
        documentsFound: 0,
        documentsUploaded: 0,
        urls: []
      });
      continue;
    }

    const urls: string[] = [];

    // Download and upload each document
    for (const doc of documents.slice(0, 5)) { // Limit to 5 per company for safety
      const filepath = await downloadDocument(doc, outputDir);

      if (filepath) {
        const url = await uploadToSupabase(filepath, doc);

        if (url) {
          urls.push(url);
        }

        // Cleanup temp file
        try {
          fs.unlinkSync(filepath);
        } catch (e) {}
      }

      // Small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    results.push({
      ticker: company.ticker,
      companyName: company.name,
      documentsFound: documents.length,
      documentsUploaded: urls.length,
      urls: urls
    });

    // Delay between companies
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  // Summary
  console.log('\n' + '='.repeat(70));
  console.log('📊 EXTRACTION SUMMARY');
  console.log('='.repeat(70));

  let totalFound = 0;
  let totalUploaded = 0;

  for (const result of results) {
    totalFound += result.documentsFound;
    totalUploaded += result.documentsUploaded;

    console.log(`\n${result.companyName} (${result.ticker})`);
    console.log(`  Found: ${result.documentsFound} | Uploaded: ${result.documentsUploaded}`);

    if (result.urls.length > 0) {
      console.log(`  URLs:`);
      result.urls.forEach(url => console.log(`    ${url}`));
    }
  }

  console.log(`\n${'='.repeat(70)}`);
  console.log(`✅ Total Documents Found: ${totalFound}`);
  console.log(`✅ Total Documents Uploaded: ${totalUploaded}`);
  console.log(`📦 Storage Location: Supabase refinitiv bucket`);
  console.log('='.repeat(70));

  // Save summary to file
  fs.writeFileSync(
    './factset-extraction-summary.json',
    JSON.stringify(results, null, 2)
  );

  console.log(`\n📝 Summary saved to: factset-extraction-summary.json`);
}

if (require.main === module) {
  main().catch(console.error);
}
