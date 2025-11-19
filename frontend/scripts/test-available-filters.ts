/**
 * Test queries using only available filter fields
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
      console.log(`❌ Errors:`, JSON.stringify(result.errors[0], null, 2));
      return null;
    }

    if (result.data) {
      for (const key of Object.keys(result.data)) {
        const items = result.data[key];
        if (Array.isArray(items)) {
          console.log(`✅ Success! Found ${items.length} documents`);
          if (items.length > 0) {
            console.log(`\n📄 First result:`);
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
  console.log('🔬 Testing Available Filter Fields\n');

  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('❌ REFINITIV_BEARER_TOKEN not set!\n');
    process.exit(1);
  }

  console.log(`✅ Token ready\n`);

  // Test 1: FeedName = SEDAR (correct syntax)
  await testQuery(
    'Test 1: SEDAR filings',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              FeedName: { EQ: "SEDAR" }
            }
          }
        }
        limit: 3
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocId
          DocumentName
          DocumentSummary {
            FeedName
            DocumentType
            FormType
            FilingDate
            HighLevelCategory
            MidLevelCategory
          }
        }
        FilingOrganization {
          CommonName
        }
      }
    }`
  );

  // Test 2: DocumentType contains "Technical"
  await testQuery(
    'Test 2: DocumentType = Technical Report',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              DocumentType: { CONT: "Technical" }
            }
          }
        }
        limit: 3
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocumentName
          DocumentSummary {
            FeedName
            DocumentType
            FilingDate
          }
        }
      }
    }`
  );

  // Test 3: FormType contains "Technical"
  await testQuery(
    'Test 3: FormType contains Technical',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              FormType: { CONT: "Technical" }
            }
          }
        }
        limit: 3
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            FeedName
            FormType
            DocumentType
          }
        }
      }
    }`
  );

  // Test 4: SEDAR + DocumentType
  await testQuery(
    'Test 4: SEDAR + Technical DocumentType',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              FeedName: { EQ: "SEDAR" }
              DocumentType: { CONT: "Technical" }
            }
          }
        }
        limit: 5
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocumentName
          DocumentSummary {
            FeedName
            DocumentType
            FormType
            FilingDate
          }
        }
        FilingOrganization {
          CommonName
        }
      }
    }`
  );

  // Test 5: HighLevelCategory or MidLevelCategory
  await testQuery(
    'Test 5: MidLevelCategory = Technical Report',
    `{
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              MidLevelCategory: { CONT: "Technical" }
            }
          }
        }
        limit: 3
      ) {
        ObjectId
        FilingDocument {
          DocumentSummary {
            HighLevelCategory
            MidLevelCategory
            DocumentType
            FeedName
          }
        }
      }
    }`
  );

  console.log('\n' + '='.repeat(70));
  console.log('🏁 Complete');
  console.log('='.repeat(70));
}

if (require.main === module) {
  main().catch(console.error);
}
