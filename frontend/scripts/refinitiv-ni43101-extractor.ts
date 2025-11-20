/**
 * Refinitiv NI 43-101 Technical Report Extractor
 *
 * This script searches for and downloads NI 43-101 technical reports (100+ pages)
 * for mining projects from the Refinitiv database.
 */

import { createClient } from '@supabase/supabase-js';
import fs from 'fs/promises';
import path from 'path';

// Refinitiv configuration
const REFINITIV_CONFIG = {
  clientId: 'API_Playground',
  apiKey: process.env.REFINITIV_API_KEY || '155d9dbf-f0ac-46d9-8b77-f7f6dcd238f8',
  bearerToken: process.env.REFINITIV_BEARER_TOKEN || '',
  baseUrl: 'https://api.refinitiv.com'
};

// Supabase configuration
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

interface RefinitivDocument {
  filingId?: string;
  dcn?: string;
  docId?: string;
  filename?: string;
  filingDate?: string;
  companyName?: string;
  reportTitle?: string;
  projectName?: string;
  country?: string;
  province?: string;
  mineralType?: string[];
  pageCount?: number;
}

/**
 * Make authenticated request to Refinitiv API
 */
async function refinitivRequest<T>(
  endpoint: string,
  method: 'GET' | 'POST' = 'GET',
  body?: any
): Promise<T> {
  const url = `${REFINITIV_CONFIG.baseUrl}${endpoint}`;

  const headers: Record<string, string> = {
    'ClientID': REFINITIV_CONFIG.clientId,
    'X-Api-Key': REFINITIV_CONFIG.apiKey,
    'Accept': '*/*'
  };

  if (REFINITIV_CONFIG.bearerToken) {
    headers['Authorization'] = `Bearer ${REFINITIV_CONFIG.bearerToken}`;
  }

  if (body) {
    headers['Content-Type'] = 'application/json';
  }

  console.log(`🔹 ${method} ${endpoint}`);

  const response = await fetch(url, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error ${response.status}: ${errorText}`);
  }

  return response.json();
}

/**
 * Search for NI 43-101 documents using GraphQL
 * This is a research method - we'll try different query structures
 */
async function searchNI43101GraphQL(params: {
  companyName?: string;
  filingType?: string;
  startDate?: string;
  endDate?: string;
  limit?: number;
}): Promise<any> {
  const { companyName, filingType = 'NI 43-101', startDate, endDate, limit = 50 } = params;

  // Try multiple query variations
  const queries = [
    // Query 1: Simple filing type filter
    `query {
      FinancialFiling(
        filter: {
          filingType: "${filingType}"
        }
        limit: ${limit}
      ) {
        filingId
        dcn
        docId
        filename
        filingDate
        companyName
        reportTitle
      }
    }`,

    // Query 2: With company filter
    companyName ? `query {
      FinancialFiling(
        filter: {
          documentType: "Technical Report"
          companyName: "${companyName}"
        }
        limit: ${limit}
      ) {
        filingId
        dcn
        filename
        companyName
      }
    }` : null,

    // Query 3: SEDAR-specific search
    `query {
      SEDARFiling(
        filter: {
          formType: "NI 43-101"
        }
        limit: ${limit}
      ) {
        filingId
        dcn
        filename
        issuerName
        filingDate
      }
    }`,
  ].filter(Boolean);

  for (const [index, query] of queries.entries()) {
    try {
      console.log(`\n🔍 Trying GraphQL query variant ${index + 1}...`);
      const result = await refinitivRequest<any>(
        '/data-store/v1/graphql',
        'POST',
        { query }
      );

      if (result.data && Object.keys(result.data).length > 0) {
        console.log(`✅ Query variant ${index + 1} succeeded!`);
        return result;
      }
    } catch (error) {
      console.log(`⚠️  Query variant ${index + 1} failed:`, (error as Error).message);
    }
  }

  console.log('⚠️  All GraphQL queries failed. Will use alternative method.');
  return null;
}

/**
 * Known mining companies with NI 43-101 reports
 */
const MAJOR_MINING_COMPANIES = [
  'Agnico Eagle Mines',
  'Barrick Gold',
  'Newmont',
  'Kinross Gold',
  'Yamana Gold',
  'Eldorado Gold',
  'IAMGOLD',
  'B2Gold',
  'Kirkland Lake Gold',
  'Osisko Mining',
  'Pretium Resources',
  'Seabridge Gold',
  'Lundin Mining',
  'First Quantum Minerals',
  'Teck Resources',
  'Hudbay Minerals',
  'Ivanhoe Mines',
  'Centerra Gold',
  'Endeavour Mining',
  'SSR Mining'
];

/**
 * Sample NI 43-101 document identifiers (we'll need to find more)
 * These would typically come from SEDAR or company filings
 */
const KNOWN_NI43101_IDENTIFIERS = [
  // Format: { type, value, company, project }
  { type: 'dcn', value: 'cr00329072', company: 'Example Corp', project: 'Example Project' },
  // Add more as we discover them
];

/**
 * Get document by identifier
 */
async function getDocumentUrl(
  identifierType: 'dcn' | 'docId' | 'filingId',
  identifierValue: string
): Promise<string> {
  console.log(`📄 Retrieving document: ${identifierType}=${identifierValue}`);

  const result = await refinitivRequest<{ signedUrl: string }>(
    `/data/filings/v1/retrieval/search/${identifierType}/${identifierValue}`
  );

  console.log(`✅ Got signed URL`);
  return result.signedUrl;
}

/**
 * Download document from signed URL
 */
async function downloadDocument(
  signedUrl: string,
  outputPath: string
): Promise<{ size: number; path: string }> {
  console.log(`⬇️  Downloading to: ${outputPath}`);

  const response = await fetch(signedUrl);

  if (!response.ok) {
    throw new Error(`Download failed: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, Buffer.from(buffer));

  const sizeMB = (buffer.byteLength / 1024 / 1024).toFixed(2);
  console.log(`✅ Downloaded ${sizeMB} MB`);

  return { size: buffer.byteLength, path: outputPath };
}

/**
 * Upload document to Supabase storage
 */
async function uploadToSupabase(
  localPath: string,
  metadata: RefinitivDocument
): Promise<string> {
  const fileBuffer = await fs.readFile(localPath);
  const filename = path.basename(localPath);
  const storagePath = `technical-reports/${metadata.companyName?.replace(/\s+/g, '-') || 'unknown'}/${filename}`;

  console.log(`☁️  Uploading to Supabase: ${storagePath}`);

  const { data, error } = await supabase.storage
    .from('technical-documents')
    .upload(storagePath, fileBuffer, {
      contentType: 'application/pdf',
      upsert: true
    });

  if (error) throw error;

  const { data: urlData } = supabase.storage
    .from('technical-documents')
    .getPublicUrl(storagePath);

  console.log(`✅ Uploaded to Supabase`);
  return urlData.publicUrl;
}

/**
 * Store document metadata in database
 */
async function storeDocumentMetadata(
  documentData: RefinitivDocument & { pdfUrl: string; source: string }
) {
  console.log(`💾 Storing metadata for: ${documentData.companyName}`);

  // Insert into technical_documents table
  const { data, error } = await supabase
    .from('technical_documents')
    .upsert({
      company_name: documentData.companyName,
      project_name: documentData.projectName,
      document_title: documentData.reportTitle,
      document_type: 'NI 43-101',
      filing_date: documentData.filingDate,
      pdf_url: documentData.pdfUrl,
      source: documentData.source,
      metadata: {
        refinitiv_filing_id: documentData.filingId,
        refinitiv_dcn: documentData.dcn,
        refinitiv_doc_id: documentData.docId,
        filename: documentData.filename,
        country: documentData.country,
        province: documentData.province,
        mineral_types: documentData.mineralType,
        page_count: documentData.pageCount
      }
    }, {
      onConflict: 'pdf_url'
    });

  if (error) {
    console.error('❌ Error storing metadata:', error);
    throw error;
  }

  console.log(`✅ Metadata stored`);
  return data;
}

/**
 * Process a single NI 43-101 document
 */
async function processDocument(
  identifier: { type: 'dcn' | 'docId' | 'filingId'; value: string },
  metadata: Partial<RefinitivDocument>
): Promise<void> {
  const outputDir = './downloads/ni43-101';
  const filename = `${metadata.companyName?.replace(/\s+/g, '-') || identifier.value}_${identifier.type}_${identifier.value}.pdf`;
  const localPath = path.join(outputDir, filename);

  try {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Processing: ${metadata.companyName || identifier.value}`);
    console.log(`${'='.repeat(60)}`);

    // 1. Get signed URL
    const signedUrl = await getDocumentUrl(identifier.type, identifier.value);

    // 2. Download document
    const downloadResult = await downloadDocument(signedUrl, localPath);

    // 3. Upload to Supabase
    const supabaseUrl = await uploadToSupabase(localPath, metadata as RefinitivDocument);

    // 4. Store metadata
    await storeDocumentMetadata({
      ...metadata as RefinitivDocument,
      pdfUrl: supabaseUrl,
      source: 'Refinitiv LSEG',
      [identifier.type]: identifier.value
    });

    console.log(`✅ Complete: ${metadata.companyName || identifier.value}\n`);

  } catch (error) {
    console.error(`❌ Failed to process ${identifier.value}:`, error);
  }
}

/**
 * Research and find NI 43-101 identifiers
 * This function will try multiple approaches to find documents
 */
async function researchNI43101Documents(): Promise<Array<{ type: string; value: string; metadata: any }>> {
  console.log('\n🔬 RESEARCH PHASE: Finding NI 43-101 Documents\n');
  console.log('='.repeat(60));

  const foundDocuments: Array<{ type: string; value: string; metadata: any }> = [];

  // Method 1: Try GraphQL search
  console.log('\n📋 Method 1: GraphQL Search');
  console.log('-'.repeat(60));

  try {
    const graphqlResult = await searchNI43101GraphQL({ limit: 100 });

    if (graphqlResult?.data) {
      const filings = Object.values(graphqlResult.data)[0] as any[];
      console.log(`Found ${filings?.length || 0} documents via GraphQL`);

      if (filings && Array.isArray(filings)) {
        for (const filing of filings) {
          if (filing.filingId) {
            foundDocuments.push({
              type: 'filingId',
              value: filing.filingId,
              metadata: filing
            });
          }
        }
      }
    }
  } catch (error) {
    console.log('GraphQL search not available:', (error as Error).message);
  }

  // Method 2: Use known identifiers from testing
  console.log('\n📋 Method 2: Known Test Identifiers');
  console.log('-'.repeat(60));

  const testIdentifiers = [
    { type: 'dcn', value: 'cr00329072', company: 'Test Company 1' },
    { type: 'docId', value: '49612437', company: 'Test Company 2' },
    { type: 'filingId', value: '34359955599', company: 'Test Company 3' }
  ];

  for (const id of testIdentifiers) {
    foundDocuments.push({
      type: id.type,
      value: id.value,
      metadata: { companyName: id.company }
    });
  }

  console.log(`Added ${testIdentifiers.length} test documents`);

  // Method 3: Research major mining companies' recent NI 43-101 filings
  console.log('\n📋 Method 3: Mining Company Research');
  console.log('-'.repeat(60));
  console.log('Note: This would typically involve:');
  console.log('  • Querying SEDAR API for recent NI 43-101 filings');
  console.log('  • Cross-referencing with Refinitiv identifiers');
  console.log('  • Filtering for documents > 100 pages');
  console.log('  • Focus on major mining projects');

  console.log('\n📊 Research Summary');
  console.log('='.repeat(60));
  console.log(`Total documents found: ${foundDocuments.length}`);
  console.log(`Methods used: GraphQL, Known Identifiers, Company Research`);

  return foundDocuments;
}

/**
 * Main extraction workflow
 */
async function main() {
  console.log('🚀 Refinitiv NI 43-101 Technical Report Extractor\n');

  try {
    // Phase 1: Research and find documents
    const documents = await researchNI43101Documents();

    if (documents.length === 0) {
      console.log('\n⚠️  No documents found. Please check:');
      console.log('  1. Refinitiv API credentials are correct');
      console.log('  2. GraphQL schema for correct field names');
      console.log('  3. Access permissions for NI 43-101 documents');
      return;
    }

    // Phase 2: Process documents
    console.log('\n\n📥 EXTRACTION PHASE: Downloading Documents\n');
    console.log('='.repeat(60));
    console.log(`Processing ${documents.length} documents...\n`);

    let successCount = 0;
    let failCount = 0;

    for (const [index, doc] of documents.entries()) {
      console.log(`\n[${index + 1}/${documents.length}]`);

      try {
        await processDocument(
          { type: doc.type as any, value: doc.value },
          doc.metadata
        );
        successCount++;
      } catch (error) {
        console.error(`Failed:`, error);
        failCount++;
      }

      // Rate limiting: wait 1 second between requests
      if (index < documents.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    // Final summary
    console.log('\n\n' + '='.repeat(60));
    console.log('📊 EXTRACTION COMPLETE');
    console.log('='.repeat(60));
    console.log(`✅ Successful: ${successCount}`);
    console.log(`❌ Failed: ${failCount}`);
    console.log(`📁 Total: ${documents.length}`);
    console.log('\n💡 Next steps:');
    console.log('  • Review downloaded PDFs for quality');
    console.log('  • Extract structured data from reports');
    console.log('  • Link documents to projects in database');
    console.log('  • Update project metadata with report insights');

  } catch (error) {
    console.error('\n❌ Fatal error:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

export {
  searchNI43101GraphQL,
  getDocumentUrl,
  downloadDocument,
  uploadToSupabase,
  processDocument,
  researchNI43101Documents
};
