#!/usr/bin/env tsx

/**
 * Test FactSet StreetAccount News API access
 * This script verifies credentials and tests basic headline retrieval
 */

import { ApiClient, HeadlinesApi, FiltersApi } from '@factset/sdk-streetaccountnews';

// FactSet StreetAccount credentials
const FACTSET_USERNAME = 'LITHOS-2220379';
const FACTSET_PASSWORD = 'YaCeW9yKju608Q4rWUt8yxn0hb2AGQQmpyLde8IG';

async function testAuthentication() {
  console.log('=== Testing FactSet StreetAccount API Access ===\n');
  console.log(`Username: ${FACTSET_USERNAME}`);
  console.log(`Password: ${FACTSET_PASSWORD.substring(0, 10)}***\n`);

  // Initialize API client
  const apiClient = ApiClient.instance;
  const factSetApiKey = apiClient.authentications['FactSetApiKey'];
  factSetApiKey.username = FACTSET_USERNAME;
  factSetApiKey.password = FACTSET_PASSWORD;

  // Test 1: Get available filters
  console.log('Test 1: Fetching available sectors...');
  try {
    const filtersApi = new FiltersApi();
    const sectorsResponse = await filtersApi.getStreetAccountFiltersSectors();

    if (sectorsResponse && sectorsResponse.data) {
      console.log(`✓ SUCCESS - Found ${Object.keys(sectorsResponse.data).length} sectors`);
      console.log(`  Sample sectors:`, Object.keys(sectorsResponse.data).slice(0, 5).join(', '));
    } else {
      console.log('⚠ Unexpected response format');
    }
  } catch (error: any) {
    console.log(`✗ FAILED - ${error.message}`);
    return false;
  }

  // Test 2: Fetch recent headlines (small batch)
  console.log('\nTest 2: Fetching recent headlines...');
  try {
    const headlinesApi = new HeadlinesApi();

    const requestBody = {
      data: {
        predefinedRange: 'today' // Get today's headlines
      },
      meta: {
        pagination: {
          limit: 10, // Just get 10 headlines for testing
          offset: 0
        }
      }
    };

    console.log(`Request body:`, JSON.stringify(requestBody, null, 2));

    const response = await headlinesApi.getStreetAccountHeadlines(requestBody);

    if (response && response.data && Array.isArray(response.data)) {
      console.log(`✓ SUCCESS - Retrieved ${response.data.length} headlines`);

      if (response.data.length > 0) {
        const firstHeadline = response.data[0];
        console.log(`\n  Sample headline (full object):`);
        console.log(JSON.stringify(firstHeadline, null, 2));
      }

      return true;
    } else {
      console.log('⚠ Unexpected response format');
      console.log('Response:', response);
      return false;
    }
  } catch (error: any) {
    console.log(`✗ FAILED - ${error.message}`);
    if (error.response) {
      console.log('Error response:', error.response);
    }
    return false;
  }
}

async function main() {
  const success = await testAuthentication();

  console.log('\n=== Test Results ===');
  if (success) {
    console.log('✓ FactSet StreetAccount API is accessible and working!');
    console.log('You can proceed with the integration.');
  } else {
    console.log('✗ FactSet StreetAccount API test failed.');
    console.log('Please check credentials and network connectivity.');
  }
}

main().catch(console.error);
