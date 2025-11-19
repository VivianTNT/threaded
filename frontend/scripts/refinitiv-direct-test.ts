/**
 * Refinitiv Direct API Test - Using API Key Only
 * Some endpoints may work with just the API key
 */

const REFINITIV_CONFIG = {
  apiKey: '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8',
  clientId: 'API_Playground'
};

/**
 * Test if we can access any endpoints with just API key
 */
async function testAPIKeyOnly() {
  console.log('🔑 Testing API access with API Key only (no OAuth)\n');
  console.log('='.repeat(60));

  const testEndpoints = [
    {
      name: 'GraphQL Schema (GET)',
      url: 'https://api.refinitiv.com/data-store/v1/graphql/schema/sdl',
      method: 'GET' as const,
      headers: {
        'X-Api-Key': REFINITIV_CONFIG.apiKey
      }
    },
    {
      name: 'Document by DCN',
      url: 'https://api.refinitiv.com/data/filings/v1/retrieval/search/dcn/cr00329072',
      method: 'GET' as const,
      headers: {
        'ClientID': REFINITIV_CONFIG.clientId,
        'X-Api-Key': REFINITIV_CONFIG.apiKey,
        'Accept': '*/*'
      }
    },
    {
      name: 'GraphQL Introspection',
      url: 'https://api.refinitiv.com/data-store/v1/graphql',
      method: 'POST' as const,
      headers: {
        'X-Api-Key': REFINITIV_CONFIG.apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: `
          query {
            __schema {
              types {
                name
                description
              }
            }
          }
        `
      })
    }
  ];

  for (const test of testEndpoints) {
    console.log(`\n${'-'.repeat(60)}`);
    console.log(`Test: ${test.name}`);
    console.log(`URL: ${test.url}`);

    try {
      const response = await fetch(test.url, {
        method: test.method,
        headers: test.headers,
        body: test.body
      });

      console.log(`Status: ${response.status} ${response.statusText}`);

      if (response.ok) {
        const contentType = response.headers.get('content-type');
        console.log(`✅ Success! Content-Type: ${contentType}`);

        if (contentType?.includes('json')) {
          const data = await response.json();
          const preview = JSON.stringify(data, null, 2).substring(0, 500);
          console.log(`Response preview:\n${preview}...`);
        } else {
          const text = await response.text();
          console.log(`Response length: ${text.length} characters`);
          console.log(`Preview: ${text.substring(0, 300)}...`);
        }
      } else {
        const errorText = await response.text();
        console.log(`❌ Failed`);
        console.log(`Error: ${errorText.substring(0, 300)}`);
      }

    } catch (error) {
      console.log(`❌ Error: ${(error as Error).message}`);
    }
  }
}

/**
 * Check what authentication methods are available
 */
async function checkAuthOptions() {
  console.log('\n\n' + '='.repeat(60));
  console.log('🔍 Checking Authentication Options\n');

  // Try to get OAuth discovery document
  const endpoints = [
    'https://api.refinitiv.com/auth/oauth2/v1/.well-known/openid-configuration',
    'https://api.refinitiv.com/.well-known/oauth-authorization-server',
    'https://developers.refinitiv.com/api-catalog/refinitiv-data-platform/refinitiv-data-platform-apis'
  ];

  for (const endpoint of endpoints) {
    console.log(`\nTrying: ${endpoint}`);
    try {
      const response = await fetch(endpoint);
      console.log(`Status: ${response.status}`);
      if (response.ok) {
        const text = await response.text();
        console.log(`✅ Response: ${text.substring(0, 500)}...`);
      }
    } catch (error) {
      console.log(`❌ Not accessible`);
    }
  }
}

/**
 * Test with different client IDs
 */
async function testDifferentClientIds() {
  console.log('\n\n' + '='.repeat(60));
  console.log('🔑 Testing Different Client ID Configurations\n');

  const clientIds = [
    'API_Playground',
    '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8', // Use API key as client ID
    'lithos-ai',
    'anish@lithos-ai.com'
  ];

  for (const clientId of clientIds) {
    console.log(`\n${'-'.repeat(60)}`);
    console.log(`Testing Client ID: ${clientId}`);

    try {
      const response = await fetch('https://api.refinitiv.com/data/filings/v1/retrieval/search/dcn/cr00329072', {
        headers: {
          'ClientID': clientId,
          'X-Api-Key': REFINITIV_CONFIG.apiKey,
          'Accept': '*/*'
        }
      });

      console.log(`Status: ${response.status}`);

      if (response.ok) {
        console.log(`✅ Success with ClientID: ${clientId}`);
        const data = await response.json();
        console.log(`Response: ${JSON.stringify(data).substring(0, 200)}...`);
      } else {
        const error = await response.text();
        console.log(`❌ Failed: ${error.substring(0, 150)}`);
      }

    } catch (error) {
      console.log(`❌ Error: ${(error as Error).message}`);
    }
  }
}

async function main() {
  console.log('🚀 Refinitiv API Direct Access Test\n');

  try {
    await testAPIKeyOnly();
    await testDifferentClientIds();
    await checkAuthOptions();

    console.log('\n\n' + '='.repeat(60));
    console.log('📊 ANALYSIS\n');
    console.log('Based on the Refinitiv Developer Portal screenshot you provided,');
    console.log('it appears you need to:');
    console.log('\n1. Generate an "App Key" from the Developer Portal');
    console.log('   - Go to: AppKey Generator in the portal');
    console.log('   - This may be different from the X-Api-Key');
    console.log('\n2. Use proper OAuth2 credentials');
    console.log('   - The username/password might need to be for an application account');
    console.log('   - Not your personal login credentials');
    console.log('\n3. Check the "My APIs" section');
    console.log('   - You have 250 APIs listed under "My APIs"');
    console.log('   - These may require specific subscription/entitlement');
    console.log('\n📝 Next Steps:');
    console.log('   • Generate a new App Key from the portal');
    console.log('   • Check if /data/filings/v1 is in your "My APIs" list');
    console.log('   • Look for OAuth credentials in the Developer Portal settings');
    console.log('   • Contact Refinitiv support for proper authentication setup');

  } catch (error) {
    console.error('❌ Error:', error);
  }
}

main().catch(console.error);
