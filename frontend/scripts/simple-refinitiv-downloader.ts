/**
 * Simple Refinitiv document downloader using REST API
 * Downloads NI 43-101 and other mining technical documents
 */

import { createClient } from '@supabase/supabase-js';
import * as fs from 'fs';
import * as path from 'path';

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

// Known document identifiers to download (from previous successful test)
const DOCUMENT_IDS = [
  { type: 'dcn', id: 'cr00329072' },
  { type: 'docId', id: '49612437' },
  { type: 'filingId', id: '34359955599' },
];

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
    throw new Error(`API Error ${response.status}: ${await response.text()}`);
  }

  if (response.headers.get('content-type')?.includes('application/json')) {
    return response.json();
  }

  return response.arrayBuffer();
}

async function downloadDocument(docIdentifier: { type: string; id: string }) {
  console.log(`\n📥 Downloading: ${docIdentifier.type} = ${docIdentifier.id}`);

  try {
    // Get document using retrieval/search endpoint
    const result = await refinitivAPI(
      `/data/filings/v1/retrieval/search/${docIdentifier.type}/${docIdentifier.id}`,
      'GET'
    );

    console.log(`   ✅ Result received`);

    // Extract files from result
    const files = Object.entries(result);
    if (files.length === 0) {
      console.log(`   ⚠️  No files found`);
      return null;
    }

    console.log(`   📦 Found ${files.length} file(s)`);

    // Process first PDF file
    for (const [filename, data] of files) {
      const fileData = data as any;
      const { signedUrl, mimeType } = fileData;

      if (mimeType !== 'application/pdf') {
        console.log(`   ⏭️  Skipping non-PDF: ${filename} (${mimeType})`);
        continue;
      }

      console.log(`   📄 File: ${filename}`);
      console.log(`   ⬇️  Downloading from CDN...`);

      // Download from signed URL
      const response = await fetch(signedUrl);
      if (!response.ok) {
        console.log(`   ❌ Download failed: ${response.status}`);
        continue;
      }

      const arrayBuffer = await response.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      console.log(`   ✅ Downloaded ${(buffer.length / 1024 / 1024).toFixed(2)} MB`);

      // Upload to Supabase
      const cleanFilename = filename.replace(/[^a-zA-Z0-9._-]/g, '-');
      const source = docIdentifier.type.toLowerCase();
      const storagePath = `${source}/${docIdentifier.id}/${cleanFilename}`;

      console.log(`   ☁️  Uploading to Supabase: ${storagePath}`);

      const { data: uploadData, error } = await supabase.storage
        .from('refinitiv')
        .upload(storagePath, buffer, {
          contentType: 'application/pdf',
          upsert: true
        });

      if (error) {
        console.log(`   ❌ Supabase error: ${error.message}`);
        continue;
      }

      const { data: publicUrlData } = supabase.storage
        .from('refinitiv')
        .getPublicUrl(storagePath);

      console.log(`   ✅ Uploaded: ${publicUrlData.publicUrl}`);

      return {
        id: docIdentifier.id,
        type: docIdentifier.type,
        filename: cleanFilename,
        size: buffer.length,
        storagePath,
        publicUrl: publicUrlData.publicUrl
      };
    }

    return null;

  } catch (error: any) {
    console.log(`   ❌ Failed: ${error.message}`);
    return null;
  }
}

async function main() {
  console.log('🚀 Refinitiv Simple Document Downloader\n');

  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('❌ REFINITIV_BEARER_TOKEN not set!\n');
    console.log('Get token from: https://api.refinitiv.com');
    console.log('Then run: export REFINITIV_BEARER_TOKEN="<token>"');
    process.exit(1);
  }

  console.log(`✅ Token: ${REFINITIV_CONFIG.bearerToken.substring(0, 40)}...`);
  console.log(`✅ Will download ${DOCUMENT_IDS.length} documents\n`);

  const results = [];

  for (const docId of DOCUMENT_IDS) {
    const result = await downloadDocument(docId);
    if (result) {
      results.push(result);
    }
    // Small delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  console.log('\n' + '='.repeat(70));
  console.log(`📊 Summary: ${results.length} documents downloaded and uploaded`);
  console.log('='.repeat(70));

  if (results.length > 0) {
    console.log('\n✅ Successfully uploaded documents:');
    for (const doc of results) {
      console.log(`\n   ${doc.filename}`);
      console.log(`   ${doc.publicUrl}`);
    }
  }

  console.log('\n✅ All documents stored in Supabase bucket: refinitiv');
}

if (require.main === module) {
  main().catch(console.error);
}
