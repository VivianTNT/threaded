/**
 * Test HVMI API for searching NI 43-101 documents
 */

const REFINITIV_CONFIG = {
  apiKey: '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8',
  clientId: 'API_Playground',
  baseUrl: 'https://api.refinitiv.com',
  bearerToken: process.env.REFINITIV_BEARER_TOKEN || ''
};

async function refinitivAPI(endpoint: string, method: 'GET' | 'POST' = 'GET', body?: any) {
  const url = `${REFINITIV_CONFIG.baseUrl}${endpoint}`;

  const headers: Record<string, string> = {
    'ClientID': REFINITIV_CONFIG.clientId,
    'X-Api-Key': REFINITIV_CONFIG.apiKey,
    'Authorization': `Bearer ${REFINITIV_CONFIG.bearerToken}`,
    'Accept': '*/*'
  };

  if (body) {
    headers['Content-Type'] = 'application/json';
  }

  const response = await fetch(url, { method, headers, body: body ? JSON.stringify(body) : undefined });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API Error ${response.status}: ${error}`);
  }

  return response.json();
}

async function testQuickSearch(searchText: string) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`🔍 Quick Search: "${searchText}"`);
  console.log('='.repeat(70));

  try {
    const result = await refinitivAPI(`/data/hvmi/v1/quicksearch/${encodeURIComponent(searchText)}`, 'GET');
    console.log(`✅ Success!`);
    console.log(JSON.stringify(result, null, 2));
    return result;
  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}`);
    return null;
  }
}

async function testSearch(searchParams?: any) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`🔍 Advanced Search`);
  console.log('='.repeat(70));

  try {
    // Try GET with query params first
    let endpoint = '/data/hvmi/v1/search';
    if (searchParams) {
      const params = new URLSearchParams(searchParams).toString();
      endpoint = `${endpoint}?${params}`;
    }

    const result = await refinitivAPI(endpoint, 'GET');
    console.log(`✅ Success!`);
    console.log(JSON.stringify(result, null, 2));
    return result;
  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}`);
    return null;
  }
}

async function testDocumentsEndpoint() {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📄 Documents Endpoint`);
  console.log('='.repeat(70));

  try {
    const result = await refinitivAPI('/data/hvmi/v1/documents', 'GET');
    console.log(`✅ Success!`);
    console.log(JSON.stringify(result, null, 2).substring(0, 2000));
    return result;
  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('🔬 Testing HVMI Search APIs\n');

  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('❌ REFINITIV_BEARER_TOKEN not set!\n');
    process.exit(1);
  }

  console.log(`✅ Token ready\n`);

  // Test 1: Quick search for "NI 43-101"
  await testQuickSearch('NI 43-101');

  // Test 2: Quick search for "43-101"
  await testQuickSearch('43-101');

  // Test 3: Quick search for "technical report"
  await testQuickSearch('technical report');

  // Test 4: Quick search for "mining"
  await testQuickSearch('mining');

  // Test 5: Advanced search (no params)
  await testSearch();

  // Test 6: Advanced search with document type
  await testSearch({ documentType: 'Technical Report' });

  // Test 7: Documents endpoint
  await testDocumentsEndpoint();

  console.log('\n' + '='.repeat(70));
  console.log('🏁 Testing complete');
  console.log('='.repeat(70));
}

if (require.main === module) {
  main().catch(console.error);
}
