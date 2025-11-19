/**
 * Test NI 43-101 GraphQL Search
 * Using correct query structure from schema analysis
 */

const BEARER_TOKEN = process.env.REFINITIV_BEARER_TOKEN || '';
const API_KEY = '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8';

async function testGraphQLQuery(query: string, name: string) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Testing: ${name}`);
  console.log('='.repeat(60));
  console.log('\nQuery:');
  console.log(query);
  console.log('\n' + '-'.repeat(60));

  try {
    const response = await fetch('https://api.refinitiv.com/data-store/v1/graphql', {
      method: 'POST',
      headers: {
        'X-Api-Key': API_KEY,
        'Authorization': `Bearer ${BEARER_TOKEN}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ query })
    });

    console.log(`Status: ${response.status} ${response.statusText}`);

    if (!response.ok) {
      const error = await response.text();
      console.log(`❌ Failed:`, error.substring(0, 500));
      return null;
    }

    const result = await response.json();

    if (result.errors) {
      console.log('❌ GraphQL Errors:');
      result.errors.forEach((err: any) => {
        console.log(`  - ${err.message}`);
        if (err.locations) {
          console.log(`    Location: Line ${err.locations[0].line}, Column ${err.locations[0].column}`);
        }
      });
      return null;
    }

    if (result.data) {
      console.log('✅ Success!');
      console.log('\nResults:');
      console.log(JSON.stringify(result.data, null, 2));
      return result.data;
    }

    console.log('⚠️ No data or errors returned');
    return null;

  } catch (error) {
    console.log(`❌ Error: ${(error as Error).message}`);
    return null;
  }
}

async function main() {
  console.log('🔍 Testing NI 43-101 GraphQL Search Queries\n');

  if (!BEARER_TOKEN) {
    console.error('❌ REFINITIV_BEARER_TOKEN not set!');
    console.log('Usage: REFINITIV_BEARER_TOKEN="token" npx tsx scripts/test-ni43101-search.ts');
    process.exit(1);
  }

  console.log(`✅ Token found: ${BEARER_TOKEN.substring(0, 50)}...`);

  // Test 1: Simple DocumentMaster query (baseline test)
  const query1 = `
    query TestDocumentMaster {
      DocumentMaster(limit: 3) {
        ObjectId
        Document {
          DocumentId
          TitleText {
            Value
          }
        }
      }
    }
  `;
  await testGraphQLQuery(query1, 'Test 1: Basic DocumentMaster Query');

  // Test 2: FinancialFiling with SEDAR filter
  const query2 = `
    query TestSEDARFilings {
      FinancialFiling(
        filter: {
          FilingDocument_FeedName: "SEDAR"
        }
        limit: 5
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocumentName
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
            DocumentType
            FormType
          }
        }
      }
    }
  `;
  await testGraphQLQuery(query2, 'Test 2: SEDAR Filings Filter');

  // Test 3: Search by document title containing "43-101"
  const query3 = `
    query SearchNI43101ByTitle {
      FinancialFiling(
        filter: {
          FilingDocument_DocumentSummary_DocumentTitle_contains: "43-101"
        }
        limit: 10
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocId
          DocumentName
          DocumentSummary {
            DocumentTitle
            DocumentType
            FormType
            FeedName
            FilingDate
            PageCount
            HighLevelCategory
            MidLevelCategory
          }
        }
        FilingOrganization {
          CommonName
          OfficialName
        }
      }
    }
  `;
  await testGraphQLQuery(query3, 'Test 3: Search by "43-101" in Title');

  // Test 4: SEDAR + Technical Report combination
  const query4 = `
    query SEDARTechnicalReports {
      FinancialFiling(
        filter: {
          FilingDocument_FeedName: "SEDAR"
          FilingDocument_DocumentSummary_DocumentType_contains: "Technical"
        }
        limit: 10
      ) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocumentName
          DocumentSummary {
            DocumentTitle
            DocumentType
            FilingDate
            PageCount
          }
        }
        FilingOrganization {
          CommonName
        }
      }
    }
  `;
  await testGraphQLQuery(query4, 'Test 4: SEDAR Technical Reports');

  // Test 5: Just list some FinancialFilings to see structure
  const query5 = `
    query ListSomeFilings {
      FinancialFiling(limit: 3) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
            DocumentType
            FormType
          }
        }
      }
    }
  `;
  await testGraphQLQuery(query5, 'Test 5: List Sample Filings');

  console.log('\n\n' + '='.repeat(60));
  console.log('📊 TESTING COMPLETE');
  console.log('='.repeat(60));
  console.log('\nReview the results above to see which queries work.');
  console.log('Successful queries can be used to find NI 43-101 documents.');
}

main().catch(console.error);
