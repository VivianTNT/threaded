/**
 * Test Discovery Search API for NI 43-101 documents
 * This API looks more promising - has CatalogItems view
 */

const REFINITIV_CONFIG = {
  apiKey: '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8',
  clientId: 'API_Playground',
  baseUrl: 'https://api.refinitiv.com',
  bearerToken: process.env.REFINITIV_BEARER_TOKEN || ''
};

async function discoverySearch(searchBody: any) {
  const url = `${REFINITIV_CONFIG.baseUrl}/discovery/search/v1/`;

  const headers: Record<string, string> = {
    'ClientID': REFINITIV_CONFIG.clientId,
    'X-Api-Key': REFINITIV_CONFIG.apiKey,
    'Authorization': `Bearer ${REFINITIV_CONFIG.bearerToken}`,
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  };

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(searchBody)
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API Error ${response.status}: ${error}`);
  }

  return response.json();
}

async function testSearch(name: string, searchBody: any) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`🔍 ${name}`);
  console.log('='.repeat(70));
  console.log('Request:', JSON.stringify(searchBody, null, 2));

  try {
    const result = await discoverySearch(searchBody);
    console.log(`\n✅ Success! Found ${result.Total || 0} results`);

    if (result.Hits && result.Hits.length > 0) {
      console.log(`\n📄 First ${Math.min(3, result.Hits.length)} results:`);
      result.Hits.slice(0, 3).forEach((hit: any, i: number) => {
        console.log(`\n${i + 1}. ${hit.DocumentTitle || hit.RIC || 'Unknown'}`);
        console.log(JSON.stringify(hit, null, 2).substring(0, 500));
      });
    }

    if (result.Warnings) {
      console.log(`\n⚠️  Warnings:`, result.Warnings);
    }

    return result;
  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('🔬 Testing Discovery Search API\n');

  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('❌ REFINITIV_BEARER_TOKEN not set!\n');
    process.exit(1);
  }

  console.log(`✅ Token ready\n`);

  // Test 1: Simple search for "NI 43-101" in all content
  await testSearch('Search: NI 43-101', {
    Query: 'NI 43-101',
    View: 'SearchAll',
    Top: 10
  });

  // Test 2: Search in CatalogItems view (documents)
  await testSearch('CatalogItems: 43-101', {
    Query: '43-101',
    View: 'CatalogItems',
    Top: 10
  });

  // Test 3: Search for technical report
  await testSearch('CatalogItems: technical report', {
    Query: 'technical report',
    View: 'CatalogItems',
    Top: 10
  });

  // Test 4: Search for mining
  await testSearch('CatalogItems: mining', {
    Query: 'mining',
    View: 'CatalogItems',
    Top: 10
  });

  // Test 5: Search with filter for Canada
  await testSearch('43-101 + Canada filter', {
    Query: '43-101',
    Filter: "Country eq 'Canada'",
    View: 'SearchAll',
    Top: 10
  });

  // Test 6: Structured search
  await testSearch('Structured: mining + technical', {
    Query: 'mining technical report',
    View: 'CatalogItems',
    Top: 10,
    Select: 'DocumentTitle,RIC,PI'
  });

  console.log('\n' + '='.repeat(70));
  console.log('🏁 Complete');
  console.log('='.repeat(70));
}

if (require.main === module) {
  main().catch(console.error);
}
