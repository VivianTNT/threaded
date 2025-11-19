import axios from 'axios';

const BEARER_TOKEN = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImRMdFd2Q0tCSC1NclVyWm9YMXFod2pZQ2t1eDV0V2ZSS2o4ME9vcjdUY28iLCJ0eXAiOiJhdCtqd3QifQ.eyJhdWQiOiJhZDEwMmEwN2IwYWQ0MzVlOTNlZTczMDBiYzRkMDI2NjJhMjBmNTFkIiwiZGF0YSI6IntcImNpcGhlcnRleHRcIjpcIm5EdmxkckhmOS16ZjJGV3AtNnFoUXBhemJKUDVVN1ozQ20tVDhFX3FlNS1FNldabTJQNWswVVN6MXpqZ3lycDJYeUhiMzdwVFkwT3dKYnNnclJvWDFjRHVMZGdkVVEzV3NDdDZxVXk5TTdkS1FPOGgxZzgzNkg4WnpkM1RVSzc0OUdqbE9lbVdpOXdGS09KdkpHM09sQ1doZmJsa2xuTjNpQ25fVVB3dTZheUl0N3JkRmVFcUtpUHdwOVZvY25mR2hvSzhMbkl4YVRSV1B2VnZsZU1tLWhZMGZkZW9GcFZJV3d3d3dhOFk1STlSNjgyZm45Vy1LNDBSNTYyV3RReHJENjFLT0FpVDFBX0I0ZWpENzF0S1RxZXZyRnJFWXlXTjBaenBuSEdkb1dIeTAzbnB1YzEwOUVxSTh2aFpHd0ZJdElFb2FrakVMRC01dy16c2plMDFaUGxEWkhQZHU4aXA4VFVxYUJGbjhjU2tsVlpDcFdfWUwwM3pZSUdqcW54Nnk1ZGlvWHotVFpBNnVIZHl6Z0Mwekp0bVFtOXB5RGJ5eXpWYUpFQkl6RkhDQWp0Q0o4VW9wNTlsbVZOTDJ1ZW83NnY4VUxCdGc2RUdqUVwiLFwiZW5jcnlwdGVkX2tleVwiOlwiQVFJQkFIaVNlRXB1YUtGWGx3OU1JZDVwRDdNVzdveHRPX3ljOEpucGZvcngtN2JPU3dGeEkta2ZEQmNsNl9SdnRzcFlpeWpVQUFBQWZqQjhCZ2txaGtpRzl3MEJCd2FnYnpCdEFnRUFNR2dHQ1NxR1NJYjNEUUVIQVRBZUJnbGdoa2dCWlFNRUFTNHdFUVFNMXdoUHc1bl9kUzBZelBwaEFnRVFnRHYzZWYzbFpzX1RlUFZETHB4Rk9EYmxoNDNRTldMdVRERnQyUzd5eGR5eWk0MU1zZmNHdW4wYWQ3TktCenRBMzVqbVc2czM5MGdweS1BS0RRXCIsXCJoZWFkZXJcIjp7XCJraWRcIjpcImFybjphd3M6a21zOnVzLWVhc3QtMTo4OTgwODQ5ODQ4Nzc6a2V5L21yay1jNzljNGNmNmI5NzA0NjQxYWQ4ZDkyZjYyNDRjODM1MFwifSxcIml2XCI6XCIyLTFvc2JBZEtCSi1Qa1lDXCIsXCJwcm90ZWN0ZWRcIjpcImV5SmhiR2NpT2lKQlYxTmZSVTVEWDFORVMxOUJNalUySWl3aVpXNWpJam9pUVRJMU5rZERUU0lzSW5wcGNDSTZJa1JGUmlKOVwiLFwidGFnXCI6XCJ2dThWdzBxVUNndEEyWk5uSng0cE9RXCJ9IiwiZXhwIjoxNzYxODY4NTg5LCJpYXQiOjE3NjE4Njc5ODksImlzcyI6Imh0dHBzOi8vaWRlbnRpdHkuY2lhbS5yZWZpbml0aXYuY29tL2FwaS9pZGVudGl0eS9zdHNfcHJvZCIsInJzMSI6ImQ5OTExMDEzOWVjOTk0NDQ4YmRkMDMyZDk0MGZlNDlmYTVhN2UwZTMifQ.ZunSSEHsKAvhy5O1K6zWHsX2sCbftgN0Te8Nn_kCoLXR3Di05KuL0AYeqc_RbYiJF_Ajpc5YOOs0gmkgEsffZx7-prn1W5Yypo2_z4NRHeE9TqUpTC7dzgCMsUs0SfJy6EOveXe45zvvvH-P7gV8x2RasLvKPOKotOHZrs5K04niY0rhW2QaUuS-8v2xAAsmHanw08Hyfx-MoIKIZUIh51_hL4pNBGPQNvZ6gOoKOW1YAyYeEgbe5RoTCe3ztlrB5CEktMFOepz1LO-xZFoTDAaxKyxHOjmB4EWwPegUtZ8rNuN03WWrdDuAsNEDCLPGg5Yw5nHUZYKM5t1rZ1cekQ";

async function testQuery() {
  console.log('🔍 Testing Refinitiv GraphQL API for FinancialFiling...\n');

  // Test 1: Simple query to see what's available
  const testQuery1 = `
    query TestFilings {
      FinancialFiling(limit: 5) {
        _metadata {
          totalCount
        }
        FilingOrganization {
          Names {
            Name {
              OrganizationName {
                Name
              }
            }
          }
        }
        FilingDocument {
          DocumentSummary {
            DocumentTitle
            FeedName
            FormType
            FilingDate
            PageCount
          }
        }
      }
    }
  `;

  console.log('Test 1: Fetching any 5 filings...');
  try {
    const response1 = await axios.post(
      'https://api.refinitiv.com/data-store/v1/graphql',
      { query: testQuery1 },
      {
        headers: {
          'Authorization': `Bearer ${BEARER_TOKEN}`,
          'Content-Type': 'application/json'
        }
      }
    );

    console.log('✅ Query successful!');
    console.log('Total count:', response1.data.data.FinancialFiling?.[0]?._metadata?.totalCount);
    console.log('\nSample filings:');
    response1.data.data.FinancialFiling?.slice(0, 3).forEach((filing: any, i: number) => {
      console.log(`\n${i + 1}. ${filing.FilingDocument.DocumentSummary.DocumentTitle}`);
      console.log(`   Feed: ${filing.FilingDocument.DocumentSummary.FeedName}`);
      console.log(`   Form: ${filing.FilingDocument.DocumentSummary.FormType}`);
      console.log(`   Pages: ${filing.FilingDocument.DocumentSummary.PageCount}`);
    });
  } catch (error: any) {
    console.error('❌ Test 1 failed:', error.response?.data || error.message);
  }

  // Test 2: Filter by SEDAR
  console.log('\n\n' + '='.repeat(70));
  console.log('Test 2: Filtering for SEDAR filings...');
  const testQuery2 = `
    query SEDARFilings {
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              FeedName: {EQ: "SEDAR"}
            }
          }
        },
        limit: 10
      ) {
        _metadata {
          totalCount
        }
        FilingDocument {
          DocumentSummary {
            DocumentTitle
            FormType
            PageCount
            FilingDate
          }
        }
      }
    }
  `;

  try {
    const response2 = await axios.post(
      'https://api.refinitiv.com/data-store/v1/graphql',
      { query: testQuery2 },
      {
        headers: {
          'Authorization': `Bearer ${BEARER_TOKEN}`,
          'Content-Type': 'application/json'
        }
      }
    );

    console.log('✅ Query successful!');
    console.log('SEDAR filings found:', response2.data.data.FinancialFiling?.length || 0);
    response2.data.data.FinancialFiling?.forEach((filing: any, i: number) => {
      console.log(`\n${i + 1}. ${filing.FilingDocument.DocumentSummary.DocumentTitle}`);
      console.log(`   Form: ${filing.FilingDocument.DocumentSummary.FormType}`);
      console.log(`   Pages: ${filing.FilingDocument.DocumentSummary.PageCount}`);
    });
  } catch (error: any) {
    console.error('❌ Test 2 failed:', error.response?.data || error.message);
  }

  // Test 3: Search for documents with many pages
  console.log('\n\n' + '='.repeat(70));
  console.log('Test 3: Finding documents with 100+ pages...');
  const testQuery3 = `
    query LargeDocuments {
      FinancialFiling(
        filter: {
          FilingDocument: {
            DocumentSummary: {
              PageCount: {GT: 100}
            }
          }
        },
        sort: {FilingDocument: {DocumentSummary: {PageCount: DESC}}},
        limit: 10
      ) {
        FilingDocument {
          DocumentSummary {
            DocumentTitle
            FeedName
            FormType
            PageCount
            FilingDate
          }
        }
      }
    }
  `;

  try {
    const response3 = await axios.post(
      'https://api.refinitiv.com/data-store/v1/graphql',
      { query: testQuery3 },
      {
        headers: {
          'Authorization': `Bearer ${BEARER_TOKEN}`,
          'Content-Type': 'application/json'
        }
      }
    );

    console.log('✅ Query successful!');
    console.log('Large documents found:', response3.data.data.FinancialFiling?.length || 0);
    response3.data.data.FinancialFiling?.forEach((filing: any, i: number) => {
      console.log(`\n${i + 1}. ${filing.FilingDocument.DocumentSummary.DocumentTitle.substring(0, 80)}...`);
      console.log(`   Feed: ${filing.FilingDocument.DocumentSummary.FeedName}`);
      console.log(`   Form: ${filing.FilingDocument.DocumentSummary.FormType}`);
      console.log(`   Pages: ${filing.FilingDocument.DocumentSummary.PageCount}`);
    });
  } catch (error: any) {
    console.error('❌ Test 3 failed:', error.response?.data || error.message);
  }
}

testQuery().catch(console.error);
