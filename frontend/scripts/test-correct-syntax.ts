/**
 * Test correct GraphQL syntax based on schema
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
  console.log(`🧪 ${name}`);
  console.log('='.repeat(70));

  try {
    const result = await refinitivAPI('/data-store/v1/graphql', 'POST', { query });

    if (result.errors) {
      console.log(`❌ Errors:`, JSON.stringify(result.errors, null, 2));
      return null;
    }

    if (result.data) {
      const keys = Object.keys(result.data);
      console.log(`✅ Success!`);

      for (const key of keys) {
        const items = result.data[key];
        if (Array.isArray(items)) {
          console.log(`   ${key}: ${items.length} items found`);
          if (items.length > 0) {
            console.log(`\n📄 Sample (first result):`);
            console.log(JSON.stringify(items[0], null, 2));
          }
        }
      }

      return result.data;
    }

  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('🔬 Testing Correct GraphQL Syntax\n');

  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('❌ REFINITIV_BEARER_TOKEN not set!\n');
    process.exit(1);
  }

  console.log(`✅ Token: ${REFINITIV_CONFIG.bearerToken.substring(0, 50)}...\n`);

  // Test 1: Correct nested structure - SEDAR only
  await testQuery(
    'Test 1: SEDAR filings (correct nested syntax)',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              FeedName: { eq: "SEDAR" }
            }
          }
        }
        limit: 3
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocId
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
            FormType
            DocumentType
          }
        }
        FilingOrganization {
          CommonName
        }
      }
    }`
  );

  // Test 2: Search for "Technical" in document title
  await testQuery(
    'Test 2: Documents with "Technical" in title',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              DocumentTitle: { contains: "Technical" }
            }
          }
        }
        limit: 3
      ) {
        ObjectId
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

  // Test 3: Search for "43-101"
  await testQuery(
    'Test 3: NI 43-101 documents',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              DocumentTitle: { contains: "43-101" }
            }
          }
        }
        limit: 5
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocId
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
            DocumentType
          }
        }
        FilingOrganization {
          CommonName
        }
      }
    }`
  );

  // Test 4: SEDAR + "43-101" combination
  await testQuery(
    'Test 4: SEDAR NI 43-101 reports',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              FeedName: { eq: "SEDAR" }
              DocumentTitle: { contains: "43-101" }
            }
          }
        }
        limit: 10
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocId
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
            DocumentType
            FormType
          }
        }
        FilingOrganization {
          CommonName
        }
      }
    }`
  );

  console.log('\n' + '='.repeat(70));
  console.log('🏁 Testing complete');
  console.log('='.repeat(70));
}

if (require.main === module) {
  main().catch(console.error);
}
