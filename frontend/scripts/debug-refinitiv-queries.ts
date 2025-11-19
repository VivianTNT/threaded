/**
 * Debug script to test Refinitiv GraphQL queries incrementally
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

async function testQuery(name: string, query: string) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`🧪 TEST: ${name}`);
  console.log('='.repeat(70));
  console.log(`Query:\n${query}\n`);

  try {
    const result = await refinitivAPI('/data-store/v1/graphql', 'POST', { query });

    if (result.errors) {
      console.log(`❌ GraphQL Errors:`, JSON.stringify(result.errors, null, 2));
      return null;
    }

    if (result.data) {
      const keys = Object.keys(result.data);
      console.log(`✅ Success! Data keys: ${keys.join(', ')}`);

      for (const key of keys) {
        const items = result.data[key];
        if (Array.isArray(items)) {
          console.log(`   ${key}: ${items.length} items`);
          if (items.length > 0) {
            console.log(`\n📄 Sample result:`);
            console.log(JSON.stringify(items[0], null, 2));
          }
        } else {
          console.log(`   ${key}:`, items);
        }
      }

      return result.data;
    }

    console.log(`⚠️  No data or errors in response:`, result);
    return null;

  } catch (error: any) {
    console.log(`❌ Request failed: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('🔬 Refinitiv GraphQL Query Debugger\n');

  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('❌ ERROR: REFINITIV_BEARER_TOKEN not set!\n');
    process.exit(1);
  }

  console.log(`✅ Token: ${REFINITIV_CONFIG.bearerToken.substring(0, 50)}...`);

  // Test 1: Simplest possible query - just get 5 filings
  await testQuery(
    'Test 1: Basic query - any 5 filings',
    `{
      FinancialFiling(limit: 5) {
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
          }
        }
      }
    }`
  );

  // Test 2: Filter by SEDAR only
  await testQuery(
    'Test 2: SEDAR filings only',
    `{
      FinancialFiling(
        filter: {
          FilingDocument_DocumentSummary_FeedName: "SEDAR"
        }
        limit: 5
      ) {
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
          }
        }
      }
    }`
  );

  // Test 3: Search for "Technical" in title
  await testQuery(
    'Test 3: Technical in title',
    `{
      FinancialFiling(
        filter: {
          FilingDocument_DocumentSummary_DocumentTitle_contains: "Technical"
        }
        limit: 5
      ) {
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
          }
        }
      }
    }`
  );

  // Test 4: Search for "43-101"
  await testQuery(
    'Test 4: Search for 43-101',
    `{
      FinancialFiling(
        filter: {
          FilingDocument_DocumentSummary_DocumentTitle_contains: "43-101"
        }
        limit: 5
      ) {
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
            PageCount
          }
        }
      }
    }`
  );

  // Test 5: SEDAR + Technical
  await testQuery(
    'Test 5: SEDAR + Technical',
    `{
      FinancialFiling(
        filter: {
          FilingDocument_DocumentSummary_FeedName: "SEDAR"
          FilingDocument_DocumentSummary_DocumentTitle_contains: "Technical"
        }
        limit: 5
      ) {
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
          }
        }
      }
    }`
  );

  // Test 6: SEDAR + 43-101
  await testQuery(
    'Test 6: SEDAR + 43-101',
    `{
      FinancialFiling(
        filter: {
          FilingDocument_DocumentSummary_FeedName: "SEDAR"
          FilingDocument_DocumentSummary_DocumentTitle_contains: "43-101"
        }
        limit: 5
      ) {
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
            PageCount
          }
        }
        FilingOrganization {
          CommonName
        }
      }
    }`
  );

  // Test 7: Try camelCase instead of underscores
  await testQuery(
    'Test 7: Try camelCase syntax',
    `{
      FinancialFiling(
        filter: {
          filingDocument: {
            documentSummary: {
              feedName: "SEDAR"
            }
          }
        }
        limit: 5
      ) {
        FilingDocument {
          FinancialFilingId
        }
      }
    }`
  );

  // Test 8: Large page count (technical reports are usually big)
  await testQuery(
    'Test 8: Large documents (>100 pages)',
    `{
      FinancialFiling(
        filter: {
          FilingDocument_DocumentSummary_PageCount_gte: 100
        }
        limit: 5
      ) {
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
            PageCount
            DocumentType
          }
        }
      }
    }`
  );

  console.log('\n' + '='.repeat(70));
  console.log('🏁 Debug session complete');
  console.log('='.repeat(70));
}

if (require.main === module) {
  main().catch(console.error);
}
