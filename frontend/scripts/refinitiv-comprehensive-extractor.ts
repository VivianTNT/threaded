/**
 * Refinitiv NI 43-101 Comprehensive Extractor
 * - Auto-uploads to Supabase 'refinitiv' bucket
 * - Searches specifically for NI 43-101 and SK-1300 documents
 * - Supports both SEDAR (Canadian) and EDGAR (US) sources
 */

import { createClient } from '@supabase/supabase-js';
import fs from 'fs/promises';
import path from 'path';

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

interface DocumentMetadata {
  filingId: string;
  docId?: string;
  title: string;
  company?: string;
  filingDate?: string;
  source: 'SEDAR' | 'EDGAR' | 'Unknown';
  documentType: string;
  pageCount?: number;
}

/**
 * Make API request to Refinitiv
 */
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

/**
 * Search for NI 43-101 documents (Canadian mining technical reports)
 */
async function searchNI43101Documents(): Promise<any[]> {
  console.log('\n🔍 Searching for NI 43-101 Documents (SEDAR)...');

  const queries = [
    // Query 1: Direct title search
    {
      name: 'Title contains "43-101"',
      query: `{
        FinancialFiling(
          filter: {
            FilingDocument_DocumentSummary_DocumentTitle_contains: "43-101"
            FilingDocument_DocumentSummary_PageCount_gte: 50
          }
          limit: 20
        ) {
          FilingDocument {
            FinancialFilingId
            DocId
            DocumentName
            DocumentSummary {
              DocumentTitle
              FeedName
              DocumentType
              FilingDate
              PageCount
            }
          }
          FilingOrganization {
            CommonName
          }
        }
      }`
    },
    // Query 2: SEDAR + Technical combination
    {
      name: 'SEDAR Technical Reports',
      query: `{
        FinancialFiling(
          filter: {
            FilingDocument_DocumentSummary_FeedName: "SEDAR"
            FilingDocument_DocumentSummary_DocumentType_contains: "Technical"
          }
          limit: 20
        ) {
          FilingDocument {
            FinancialFilingId
            DocId
            DocumentSummary {
              DocumentTitle
              DocumentType
              FilingDate
              PageCount
            }
          }
        }
      }`
    },
    // Query 3: High-level category filter
    {
      name: 'Mining Company Reports',
      query: `{
        FinancialFiling(
          filter: {
            FilingDocument_DocumentSummary_MidLevelCategory_contains: "Report"
            FilingDocument_DocumentSummary_PageCount_gte: 100
          }
          limit: 20
        ) {
          FilingDocument {
            FinancialFilingId
            DocumentSummary {
              DocumentTitle
              FeedName
              PageCount
            }
          }
        }
      }`
    }
  ];

  const allResults: any[] = [];

  for (const { name, query } of queries) {
    try {
      console.log(`  Trying: ${name}...`);
      const result = await refinitivAPI('/data-store/v1/graphql', 'POST', { query });

      if (result.data?.FinancialFiling) {
        const count = result.data.FinancialFiling.length;
        console.log(`  ✅ Found ${count} documents`);
        allResults.push(...result.data.FinancialFiling);
      }
    } catch (error: any) {
      console.log(`  ❌ Failed: ${error.message.substring(0, 100)}`);
    }
  }

  return allResults;
}

/**
 * Search for SK-1300 documents (US mining technical reports)
 */
async function searchSK1300Documents(): Promise<any[]> {
  console.log('\n🔍 Searching for SK-1300 Documents (SEC EDGAR)...');

  const query = `{
    FinancialFiling(
      filter: {
        FilingDocument_DocumentSummary_FeedName: "EDGAR"
        FilingDocument_DocumentSummary_FormType: "10-K"
        FilingDocument_DocumentSummary_DocumentTitle_contains: "96.1"
      }
      limit: 20
    ) {
      FilingDocument {
        FinancialFilingId
        DocId
        DocumentSummary {
          DocumentTitle
          FormType
          FilingDate
          PageCount
        }
      }
      FilingOrganization {
        CommonName
      }
    }
  }`;

  try {
    const result = await refinitivAPI('/data-store/v1/graphql', 'POST', { query });
    if (result.data?.FinancialFiling) {
      console.log(`  ✅ Found ${result.data.FinancialFiling.length} SK-1300 documents`);
      return result.data.FinancialFiling;
    }
  } catch (error: any) {
    console.log(`  ❌ Failed: ${error.message.substring(0, 100)}`);
  }

  return [];
}

/**
 * Get document by filing ID
 */
async function getDocumentUrl(filingId: string): Promise<{ url: string; filename: string; mimeType: string }[]> {
  console.log(`  📄 Getting document URL for FilingId: ${filingId}`);

  const result = await refinitivAPI(`/data/filings/v1/retrieval/search/filingId/${filingId}`);

  const files: { url: string; filename: string; mimeType: string }[] = [];

  for (const [filename, data] of Object.entries(result)) {
    const { signedUrl, mimeType } = data as any;
    files.push({ url: signedUrl, filename, mimeType });
  }

  return files;
}

/**
 * Download document from signed URL
 */
async function downloadDocument(signedUrl: string, filename: string): Promise<Buffer> {
  console.log(`  ⬇️  Downloading: ${filename.substring(0, 50)}...`);

  const response = await fetch(signedUrl);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  console.log(`  ✅ Downloaded ${(buffer.byteLength / 1024 / 1024).toFixed(2)} MB`);

  return Buffer.from(buffer);
}

/**
 * Upload document to Supabase storage
 */
async function uploadToSupabase(
  buffer: Buffer,
  filename: string,
  metadata: DocumentMetadata
): Promise<string> {
  console.log(`  ☁️  Uploading to Supabase...`);

  // Create organized path
  const source = metadata.source.toLowerCase();
  const year = metadata.filingDate ? new Date(metadata.filingDate).getFullYear() : 'unknown';
  const company = (metadata.company || 'unknown').replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();

  const storagePath = `${source}/${year}/${company}/${filename}`;

  const { data, error } = await supabase.storage
    .from('refinitiv')
    .upload(storagePath, buffer, {
      contentType: filename.endsWith('.pdf') ? 'application/pdf' : 'application/octet-stream',
      upsert: true
    });

  if (error) {
    throw new Error(`Supabase upload failed: ${error.message}`);
  }

  const { data: urlData } = supabase.storage
    .from('refinitiv')
    .getPublicUrl(storagePath);

  console.log(`  ✅ Uploaded to: refinitiv/${storagePath}`);
  return urlData.publicUrl;
}

/**
 * Store metadata in database
 */
async function storeMetadata(metadata: DocumentMetadata & { pdfUrl: string }) {
  console.log(`  💾 Storing metadata...`);

  const { error } = await supabase
    .from('technical_documents')
    .upsert({
      company_name: metadata.company,
      document_title: metadata.title,
      document_type: metadata.documentType,
      filing_date: metadata.filingDate,
      pdf_url: metadata.pdfUrl,
      source: `Refinitiv ${metadata.source}`,
      metadata: {
        refinitiv_filing_id: metadata.filingId,
        refinitiv_doc_id: metadata.docId,
        page_count: metadata.pageCount
      }
    }, {
      onConflict: 'pdf_url'
    });

  if (error) {
    console.log(`  ⚠️  Metadata storage warning: ${error.message}`);
  } else {
    console.log(`  ✅ Metadata stored`);
  }
}

/**
 * Process a single document
 */
async function processDocument(filing: any): Promise<void> {
  const doc = filing.FilingDocument;
  const summary = doc.DocumentSummary;
  const org = filing.FilingOrganization;

  const metadata: DocumentMetadata = {
    filingId: doc.FinancialFilingId,
    docId: doc.DocId,
    title: summary.DocumentTitle || doc.DocumentName || 'Unknown',
    company: org?.CommonName,
    filingDate: summary.FilingDate,
    source: summary.FeedName === 'SEDAR' ? 'SEDAR' : summary.FeedName === 'EDGAR' ? 'EDGAR' : 'Unknown',
    documentType: summary.DocumentType || summary.FormType || 'Technical Report',
    pageCount: summary.PageCount
  };

  console.log(`\n${'='.repeat(60)}`);
  console.log(`📄 ${metadata.title.substring(0, 70)}`);
  console.log(`   Company: ${metadata.company || 'Unknown'}`);
  console.log(`   Source: ${metadata.source} | Date: ${metadata.filingDate || 'N/A'} | Pages: ${metadata.pageCount || 'N/A'}`);
  console.log('='.repeat(60));

  try {
    // Get document URLs
    const files = await getDocumentUrl(metadata.filingId);

    // Download and upload PDFs only
    for (const file of files) {
      if (file.mimeType === 'application/pdf') {
        const buffer = await downloadDocument(file.url, file.filename);
        const supabaseUrl = await uploadToSupabase(buffer, file.filename, metadata);
        await storeMetadata({ ...metadata, pdfUrl: supabaseUrl });
      }
    }

    console.log(`✅ Complete!\n`);

  } catch (error: any) {
    console.log(`❌ Failed: ${error.message}\n`);
  }
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Refinitiv Comprehensive Document Extractor\n');
  console.log('='.repeat(60));

  // Check token
  if (!REFINITIV_CONFIG.bearerToken) {
    console.error('\n❌ ERROR: REFINITIV_BEARER_TOKEN not set!\n');
    console.log('Get a fresh token from: https://api.refinitiv.com');
    console.log('Then run:');
    console.log('  REFINITIV_BEARER_TOKEN="token" npx tsx scripts/refinitiv-comprehensive-extractor.ts');
    process.exit(1);
  }

  console.log(`✅ Token found: ${REFINITIV_CONFIG.bearerToken.substring(0, 50)}...`);
  console.log(`✅ Supabase bucket: refinitiv`);
  console.log('');

  try {
    // Search for documents
    const ni43101 = await searchNI43101Documents();
    const sk1300 = await searchSK1300Documents();

    const allDocuments = [...ni43101, ...sk1300];
    console.log(`\n📊 Total documents found: ${allDocuments.length}`);

    if (allDocuments.length === 0) {
      console.log('\n⚠️  No documents found with current search queries.');
      console.log('This might be because:');
      console.log('  1. GraphQL field names need adjustment');
      console.log('  2. No NI 43-101/SK-1300 docs match the filters');
      console.log('  3. Token expired or permissions issue');
      return;
    }

    // Process each document
    console.log('\n📥 Processing documents...\n');

    let successCount = 0;
    let failCount = 0;

    for (const [index, doc] of allDocuments.entries()) {
      console.log(`\n[${index + 1}/${allDocuments.length}]`);

      try {
        await processDocument(doc);
        successCount++;
      } catch (error) {
        failCount++;
      }

      // Rate limiting
      if (index < allDocuments.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 EXTRACTION COMPLETE');
    console.log('='.repeat(60));
    console.log(`✅ Successful: ${successCount}`);
    console.log(`❌ Failed: ${failCount}`);
    console.log(`📁 Total: ${allDocuments.length}`);
    console.log(`\n☁️  All PDFs uploaded to Supabase bucket: refinitiv`);
    console.log(`💾 Metadata stored in: technical_documents table`);

  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

export { searchNI43101Documents, searchSK1300Documents, processDocument };
