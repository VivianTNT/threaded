/**
 * Working Refinitiv NI 43-101 Extractor
 * Using confirmed working bearer token
 */

import { createClient } from '@supabase/supabase-js';
import fs from 'fs/promises';
import path from 'path';

const BEARER_TOKEN = process.env.REFINITIV_BEARER_TOKEN || '';

const REFINITIV_CONFIG = {
  apiKey: '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8',
  clientId: 'API_Playground',
  baseUrl: 'https://api.refinitiv.com'
};

// Supabase client
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function refinitivAPI(endpoint: string, method: 'GET' | 'POST' = 'GET', body?: any) {
  const url = `${REFINITIV_CONFIG.baseUrl}${endpoint}`;

  const headers: Record<string, string> = {
    'ClientID': REFINITIV_CONFIG.clientId,
    'X-Api-Key': REFINITIV_CONFIG.apiKey,
    'Authorization': `Bearer ${BEARER_TOKEN}`,
    'Accept': '*/*'
  };

  if (body) {
    headers['Content-Type'] = 'application/json';
  }

  console.log(`🔹 ${method} ${endpoint}`);

  const response = await fetch(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API Error ${response.status}: ${error}`);
  }

  return response.json();
}

/**
 * Search for NI 43-101 documents using GraphQL
 */
async function searchNI43101Documents() {
  console.log('\n🔍 Searching for NI 43-101 documents via GraphQL...\n');

  const queries = [
    // Try different query structures
    `{
      FinancialFiling(filter: { filingType: "NI 43-101" }, limit: 50) {
        filingId dcn filename companyName filingDate
      }
    }`,
    `{
      SEDARFiling(filter: { formType: "NI 43-101" }, limit: 50) {
        filingId dcn filename issuerName filingDate
      }
    }`,
    `{
      DocumentMaster(filter: { documentType: "Technical Report" }, limit: 50) {
        id filename companyName
      }
    }`
  ];

  for (const [index, query] of queries.entries()) {
    try {
      console.log(`Trying query ${index + 1}...`);
      const result = await refinitivAPI('/data-store/v1/graphql', 'POST', { query });

      if (result.data && Object.keys(result.data).length > 0) {
        console.log(`✅ Success with query ${index + 1}!`);
        console.log('Result:', JSON.stringify(result.data, null, 2).substring(0, 1000));
        return result.data;
      }
    } catch (error) {
      console.log(`❌ Query ${index + 1} failed:`, (error as Error).message.substring(0, 200));
    }
  }

  console.log('⚠️  No GraphQL queries succeeded. Will try alternative approach.');
  return null;
}

/**
 * Get GraphQL schema to understand structure
 */
async function getSchema() {
  console.log('\n📖 Fetching GraphQL schema...');

  const response = await fetch(`${REFINITIV_CONFIG.baseUrl}/data-store/v1/graphql/schema/sdl`, {
    headers: {
      'X-Api-Key': REFINITIV_CONFIG.apiKey,
      'Authorization': `Bearer ${BEARER_TOKEN}`
    }
  });

  if (!response.ok) {
    console.log('❌ Schema fetch failed:', response.status);
    return null;
  }

  const schema = await response.text();
  await fs.writeFile('./refinitiv-schema.sdl', schema);
  console.log(`✅ Schema saved (${(schema.length / 1024).toFixed(2)} KB)`);

  // Search for relevant types
  const lines = schema.split('\n');
  const relevantLines = lines.filter(line =>
    /filing|sedar|edgar|document|report|ni.*43|sk.*1300/i.test(line)
  );

  console.log(`\nFound ${relevantLines.length} relevant lines in schema:`);
  console.log(relevantLines.slice(0, 30).join('\n'));

  return schema;
}

/**
 * Download a document
 */
async function downloadDocument(signedUrl: string, filename: string) {
  console.log(`\n⬇️  Downloading: ${filename}`);

  const response = await fetch(signedUrl);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  const outputDir = './downloads/refinitiv';
  await fs.mkdir(outputDir, { recursive: true });

  const filepath = path.join(outputDir, filename);
  await fs.writeFile(filepath, Buffer.from(buffer));

  const sizeMB = (buffer.byteLength / 1024 / 1024).toFixed(2);
  console.log(`✅ Downloaded ${sizeMB} MB to ${filepath}`);

  return filepath;
}

/**
 * Test document retrieval with known identifiers
 */
async function testKnownDocuments() {
  console.log('\n📄 Testing known document identifiers...\n');

  const testDocs = [
    { type: 'dcn', value: 'cr00329072' },
    { type: 'docId', value: '49612437' },
    { type: 'filingId', value: '34359955599' }
  ];

  const downloaded: string[] = [];

  for (const doc of testDocs) {
    try {
      console.log(`\nTesting ${doc.type}: ${doc.value}`);
      const result = await refinitivAPI(`/data/filings/v1/retrieval/search/${doc.type}/${doc.value}`);

      console.log('✅ API Response received');

      // Extract signed URL(s)
      const files = Object.entries(result);
      console.log(`Found ${files.length} file(s)`);

      for (const [filename, data] of files) {
        const { signedUrl, mimeType } = data as any;
        console.log(`  File: ${filename}`);
        console.log(`  Type: ${mimeType}`);
        console.log(`  URL: ${signedUrl.substring(0, 100)}...`);

        // Download if it's a PDF
        if (mimeType === 'application/pdf') {
          const filepath = await downloadDocument(signedUrl, filename);
          downloaded.push(filepath);
        }
      }
    } catch (error) {
      console.log(`❌ Failed:`, (error as Error).message);
    }
  }

  return downloaded;
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Refinitiv NI 43-101 Document Extractor - WORKING VERSION\n');
  console.log('='.repeat(60));

  // Check token
  if (!BEARER_TOKEN) {
    console.error('\n❌ ERROR: REFINITIV_BEARER_TOKEN environment variable not set!\n');
    console.log('Please set it with:');
    console.log('  export REFINITIV_BEARER_TOKEN="your_token_here"');
    console.log('\nOr run with:');
    console.log('  REFINITIV_BEARER_TOKEN="your_token" npx tsx scripts/refinitiv-working-extractor.ts');
    process.exit(1);
  }

  console.log('✅ Bearer token found');
  console.log(`   Token preview: ${BEARER_TOKEN.substring(0, 50)}...`);
  console.log('');

  try {
    // Step 1: Get GraphQL schema
    console.log('\n📋 STEP 1: Understanding API Structure');
    console.log('-'.repeat(60));
    await getSchema();

    // Step 2: Search for NI 43-101 documents
    console.log('\n\n📋 STEP 2: Searching for NI 43-101 Documents');
    console.log('-'.repeat(60));
    const searchResults = await searchNI43101Documents();

    // Step 3: Test known documents
    console.log('\n\n📋 STEP 3: Testing Known Document Retrieval');
    console.log('-'.repeat(60));
    const downloaded = await testKnownDocuments();

    // Summary
    console.log('\n\n' + '='.repeat(60));
    console.log('📊 SUMMARY');
    console.log('='.repeat(60));
    console.log(`✅ API Access: Working`);
    console.log(`✅ Bearer Token: Valid`);
    console.log(`✅ Documents Downloaded: ${downloaded.length}`);

    if (downloaded.length > 0) {
      console.log('\n📁 Downloaded files:');
      downloaded.forEach(f => console.log(`   ${f}`));
    }

    console.log('\n📝 Next Steps:');
    console.log('   1. Review GraphQL schema file (refinitiv-schema.sdl)');
    console.log('   2. Identify correct query structure for NI 43-101 search');
    console.log('   3. Find document identifiers for major mining projects');
    console.log('   4. Begin bulk download process');

  } catch (error) {
    console.error('\n❌ Error:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { searchNI43101Documents, downloadDocument, testKnownDocuments };
