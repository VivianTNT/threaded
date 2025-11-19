/**
 * Try to browse DocumentMaster to find NI 43-101 reports
 * Use pagination or try querying without filters
 */

import { createClient } from '@supabase/supabase-js';

const REFINITIV_CONFIG = {
  apiKey: '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8',
  clientId: 'API_Playground',
  baseUrl: 'https://api.refinitiv.com',
  bearerToken: process.env.REFINITIV_BEARER_TOKEN || ''
};

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

async function graphqlQuery(query: string) {
  const url = `${REFINITIV_CONFIG.baseUrl}/data-store/v1/graphql`;

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
    body: JSON.stringify({ query })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API Error ${response.status}: ${error}`);
  }

  return response.json();
}

async function downloadDocument(dcn: string, docId: string, filingId: string) {
  console.log(`\n📥 Downloading: DCN=${dcn}, DocId=${docId}, FilingId=${filingId}`);

  // Try DCN first
  for (const { type, id } of [
    { type: 'dcn', id: dcn },
    { type: 'docId', id: docId },
    { type: 'filingId', id: filingId }
  ]) {
    if (!id) continue;

    try {
      const url = `${REFINITIV_CONFIG.baseUrl}/data/filings/v1/retrieval/search/${type}/${id}`;
      const response = await fetch(url, {
        headers: {
          'ClientID': REFINITIV_CONFIG.clientId,
          'X-Api-Key': REFINITIV_CONFIG.apiKey,
          'Authorization': `Bearer ${REFINITIV_CONFIG.bearerToken}`
        }
      });

      if (!response.ok) continue;

      const result = await response.json();
      const files = Object.entries(result);

      for (const [filename, data] of files) {
        const fileData = data as any;
        if (fileData.mimeType !== 'application/pdf') continue;

        console.log(`   📄 ${filename}`);
        const pdfResponse = await fetch(fileData.signedUrl);
        const buffer = Buffer.from(await pdfResponse.arrayBuffer());
        console.log(`   ✅ Downloaded ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);

        const storagePath = `sedar/${filingId}/${filename.replace(/[^a-zA-Z0-9._-]/g, '-')}`;
        const { error } = await supabase.storage
          .from('refinitiv')
          .upload(storagePath, buffer, { contentType: 'application/pdf', upsert: true });

        if (!error) {
          const { data: urlData } = supabase.storage.from('refinitiv').getPublicUrl(storagePath);
          console.log(`   ☁️  ${urlData.publicUrl}`);
          return urlData.publicUrl;
        }
      }
    } catch (e) {
      // Try next identifier
    }
  }

  console.log(`   ❌ Could not download`);
  return null;
}

async function main() {
  console.log('🔬 Browse DocumentMaster for NI 43-101\n');

  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('❌ TOKEN not set\n');
    process.exit(1);
  }

  // Try querying DocumentMaster - check what it returns
  console.log('📋 Attempting to query DocumentMaster...\n');

  const testQueries = [
    // Try with objectIds from known filings
    `{
      DocumentMaster(objectIds: ["4295865951", "4295886324", "5046649783"]) {
        ObjectId
        DocumentId
        DocumentTitle
        DocumentType
        IssuerName
        FilingDate
      }
    }`,
    // Try FinancialFiling with known ObjectIds
    `{
      FinancialFiling(objectIds: ["4295865951"]) {
        ObjectId
        FilingDocument {
          FinancialFilingId
          DocId
          DocumentName
          DocumentSummary {
            DocumentTitle
            FeedName
            FilingDate
          }
        }
      }
    }`
  ];

  for (const [i, query] of testQueries.entries()) {
    console.log(`\n${'='.repeat(70)}`);
    console.log(`Query ${i + 1}:`);
    console.log('='.repeat(70));

    try {
      const result = await graphqlQuery(query);

      if (result.errors) {
        console.log(`❌ Errors:`);
        result.errors.forEach((err: any) => console.log(`   ${err.message}`));
      } else if (result.data) {
        console.log(`✅ Success!`);
        console.log(JSON.stringify(result.data, null, 2));

        // If we got documents, try to download them
        const docs = result.data.DocumentMaster || result.data.FinancialFiling || [];
        for (const doc of docs) {
          const dcn = doc.DCN || '';
          const docId = doc.DocumentId || doc.FilingDocument?.DocId || '';
          const filingId = doc.FilingDocument?.FinancialFilingId || doc.ObjectId || '';

          if (docId || filingId || dcn) {
            await downloadDocument(dcn, docId, filingId);
          }
        }
      }
    } catch (error: any) {
      console.log(`❌ Failed: ${error.message}`);
    }
  }

  console.log('\n' + '='.repeat(70));
  console.log('🏁 Complete');
  console.log('='.repeat(70));
}

if (require.main === module) {
  main().catch(console.error);
}
