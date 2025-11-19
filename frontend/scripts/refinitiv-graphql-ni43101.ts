import { createClient } from '@supabase/supabase-js';
import axios from 'axios';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);

// Refinitiv GraphQL API endpoint
const REFINITIV_API_URL = 'https://api.refinitiv.com/data-store/v1/graphql';
const REFINITIV_API_KEY = process.env.REFINITIV_API_KEY!;
const REFINITIV_USERNAME = process.env.REFINITIV_USERNAME || '';
const REFINITIV_PASSWORD = process.env.REFINITIV_PASSWORD || '';
const REFINITIV_BEARER_TOKEN = process.env.REFINITIV_BEARER_TOKEN || '';

// Canadian mining companies - we'll search for their OrganizationIds
const CANADIAN_MINING_COMPANIES = [
  'Barrick Gold Corporation',
  'Teck Resources Limited',
  'First Quantum Minerals Ltd',
  'Kinross Gold Corporation',
  'Agnico Eagle Mines Limited',
  'Lundin Mining Corporation',
  'Hudbay Minerals Inc',
  'Capstone Copper Corp',
  'Ivanhoe Mines Ltd',
  'Filo Mining Corp',
  'Trilogy Metals Inc',
  'Copper Mountain Mining Corporation',
  'Talon Metals Corp',
  'Sigma Lithium Corporation',
  'Patriot Battery Metals Inc',
  'Foran Mining Corporation',
  'Neo Lithium Corp'
];

interface FilingDocument {
  DocId: string;
  FinancialFilingId: string;
  DocumentSummary: {
    DocumentTitle: string;
    FilingDate: string;
    FeedName: string;
    FormType: string;
    PageCount: number;
    SizeInBytes: number;
    OriginalFilename: string;
  };
  FilesMetaData: Array<{
    FileName: string;
    MimeType: string;
  }>;
}

interface FilingResult {
  FilingOrganization: {
    Names: {
      Name: {
        OrganizationName: Array<{
          Name: string;
        }>;
      };
    };
  };
  FilingDocument: FilingDocument;
}

// Global variable to store bearer token
let bearerToken: string | null = null;

async function getRefinitivBearerToken(): Promise<string> {
  // If bearer token is provided directly, use it
  if (REFINITIV_BEARER_TOKEN) {
    console.log('✅ Using provided bearer token\n');
    return REFINITIV_BEARER_TOKEN;
  }

  if (bearerToken) {
    return bearerToken;
  }

  try {
    console.log('🔐 Authenticating with Refinitiv...');

    const response = await axios.post(
      'https://api.refinitiv.com/auth/oauth2/v1/token',
      new URLSearchParams({
        'grant_type': 'password',
        'username': REFINITIV_USERNAME,
        'password': REFINITIV_PASSWORD,
        'client_id': REFINITIV_API_KEY,
        'scope': 'trapi'
      }),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Accept': 'application/json'
        }
      }
    );

    bearerToken = response.data.access_token;
    console.log('✅ Authentication successful\n');

    return bearerToken;
  } catch (error: any) {
    console.error('❌ Authentication failed:', error.response?.data || error.message);
    throw error;
  }
}

async function executeGraphQLQuery(query: string, variables: any): Promise<any> {
  try {
    const token = await getRefinitivBearerToken();

    const response = await axios.post(
      REFINITIV_API_URL,
      {
        query,
        variables
      },
      {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        }
      }
    );

    if (response.data.errors) {
      console.error('GraphQL errors:', JSON.stringify(response.data.errors, null, 2));
      return null;
    }

    return response.data.data;
  } catch (error: any) {
    console.error('API Error:', error.response?.data || error.message);
    return null;
  }
}

async function searchNI43101Filings(): Promise<FilingResult[]> {
  console.log('🔍 Searching for NI 43-101 Technical Reports via Refinitiv GraphQL...\n');

  // GraphQL query for NI 43-101 filings from SEDAR
  const query = `
    query FinancialFiling($FeedName: String!, $FormType: String!, $FilingDateFrom: DateTime!, $FilingDateTo: DateTime!, $LanguageId: Long) {
      FinancialFiling(
        filter: {
          AND: [
            {FilingDocument: {DocumentSummary: {FeedName: {EQ: $FeedName}}}},
            {FilingDocument: {DocumentSummary: {FormType: {CONT: $FormType}}}},
            {FilingDocument: {DocumentSummary: {FilingDate: {BETWN: {FROM: $FilingDateFrom, TO: $FilingDateTo}}}}},
            {FilingDocument: {DocumentSummary: {PageCount: {GT: 100}}}}
          ]
        },
        sort: {FilingDocument: {DocumentSummary: {FilingDate: DESC}}},
        limit: 50
      ) {
        _metadata {
          totalCount
        }
        FilingOrganization {
          Names {
            Name {
              OrganizationName(
                filter: {
                  AND: [
                    {LanguageId: {EQ: $LanguageId}},
                    {NameTypeCode: {EQ: "LNG"}}
                  ]
                }
              ) {
                Name
              }
            }
          }
        }
        FilingDocument {
          Identifiers {
            Dcn
          }
          DocId
          FinancialFilingId
          DocumentSummary {
            DocumentTitle
            FeedName
            FormType
            HighLevelCategory
            MidLevelCategory
            FilingDate
            PageCount
            SizeInBytes
            OriginalFilename
          }
          FilesMetaData {
            FileName
            MimeType
          }
        }
      }
    }
  `;

  const variables = {
    FeedName: 'SEDAR',
    FormType: '43-101',  // NI 43-101 Technical Reports
    FilingDateFrom: '2020-01-01T00:00:00Z',
    FilingDateTo: '2025-12-31T23:59:59Z',
    LanguageId: 505074  // English
  };

  console.log('📊 Query Parameters:');
  console.log(`   Feed: ${variables.FeedName} (SEDAR - Canadian Securities)`);
  console.log(`   Form Type: ${variables.FormType} (NI 43-101 Technical Reports)`);
  console.log(`   Date Range: ${variables.FilingDateFrom.split('T')[0]} to ${variables.FilingDateTo.split('T')[0]}`);
  console.log(`   Page Count: > 100 pages\n`);

  const data = await executeGraphQLQuery(query, variables);

  if (!data || !data.FinancialFiling) {
    console.log('❌ No filings found or API error');
    return [];
  }

  console.log(`✅ Found ${data.FinancialFiling.length} NI 43-101 reports\n`);
  return data.FinancialFiling;
}

async function downloadDocument(docId: string, filename: string): Promise<Buffer | null> {
  try {
    console.log(`   📥 Downloading document ${docId}...`);

    // Refinitiv document download endpoint
    const downloadUrl = `https://api.refinitiv.com/data/filing/v1/documents/${docId}`;

    const response = await axios.get(downloadUrl, {
      headers: {
        'Authorization': `Bearer ${REFINITIV_API_KEY}`,
        'Accept': 'application/pdf'
      },
      responseType: 'arraybuffer',
      timeout: 120000
    });

    const fileSizeMB = (response.data.byteLength / 1024 / 1024).toFixed(2);
    console.log(`   ✅ Downloaded: ${fileSizeMB} MB`);

    return Buffer.from(response.data);
  } catch (error: any) {
    console.error(`   ❌ Download error:`, error.message);
    return null;
  }
}

async function uploadToSupabase(pdfBuffer: Buffer, companyName: string, projectName: string): Promise<string | null> {
  try {
    const companySlug = companyName.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const projectSlug = projectName.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const storagePath = `sedar/ni43101/${companySlug}/${projectSlug}.pdf`;

    const { data, error } = await supabase.storage
      .from('refinitiv')
      .upload(storagePath, pdfBuffer, {
        contentType: 'application/pdf',
        upsert: true
      });

    if (error) {
      console.error(`   ❌ Upload error:`, error.message);
      return null;
    }

    const publicUrl = `https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/${storagePath}`;
    console.log(`   ✅ Uploaded to Supabase storage`);

    return publicUrl;
  } catch (error: any) {
    console.error(`   ❌ Upload error:`, error.message);
    return null;
  }
}

async function createProject(filing: FilingResult, pdfUrl: string) {
  try {
    const companyName = filing.FilingOrganization.Names.Name.OrganizationName[0]?.Name || 'Unknown Company';
    const projectName = filing.FilingDocument.DocumentSummary.DocumentTitle || 'Untitled Project';
    const pageCount = filing.FilingDocument.DocumentSummary.PageCount;

    console.log(`\n📊 Creating project for: ${companyName}`);
    console.log(`   Project: ${projectName}`);
    console.log(`   Pages: ${pageCount}`);

    // Get or create company
    let companyId: string;

    const { data: existingCompany } = await supabase
      .from('companies')
      .select('id')
      .eq('name', companyName)
      .single();

    if (existingCompany) {
      companyId = existingCompany.id;
      console.log(`   ✅ Found existing company`);
    } else {
      const { data: newCompany, error } = await supabase
        .from('companies')
        .insert({
          name: companyName,
          description: `Mining company with NI 43-101 technical report`
        })
        .select()
        .single();

      if (error) {
        console.error(`   ❌ Company creation error:`, error);
        return;
      }
      companyId = newCompany.id;
      console.log(`   ✅ Created company`);
    }

    // Check if project exists
    const { data: existingProject } = await supabase
      .from('projects')
      .select('id')
      .eq('name', projectName)
      .single();

    if (existingProject) {
      console.log(`   ⏭️  Project already exists`);
      return;
    }

    // Create project
    const { data: project, error: projectError } = await supabase
      .from('projects')
      .insert({
        name: projectName,
        company_id: companyId,
        description: `NI 43-101 Technical Report - ${pageCount} pages - SEDAR Filing`,
        urls: [pdfUrl],
        location: 'Canada',
        stage: 'To be determined',
        status: 'active',
        commodities: ['To be extracted']
      })
      .select()
      .single();

    if (projectError) {
      console.error(`   ❌ Project creation error:`, projectError);
    } else {
      console.log(`   ✅ Created project successfully`);
    }
  } catch (error: any) {
    console.error(`   ❌ Error:`, error.message);
  }
}

async function main() {
  console.log('🇨🇦 REFINITIV GRAPHQL NI 43-101 EXTRACTOR');
  console.log('='.repeat(70));
  console.log('📋 Report Type: NI 43-101 Technical Reports (100+ pages)');
  console.log('🎯 Source: SEDAR via Refinitiv Financial Filings GraphQL API');
  console.log('💎 Target: Canadian mining project technical reports');
  console.log('='.repeat(70));
  console.log();

  if (!REFINITIV_API_KEY || !REFINITIV_USERNAME || !REFINITIV_PASSWORD) {
    console.error('❌ Missing Refinitiv credentials');
    console.error('   Please set REFINITIV_API_KEY, REFINITIV_USERNAME, and REFINITIV_PASSWORD');
    console.error('   in your .env.local file');
    process.exit(1);
  }

  // Step 1: Search for NI 43-101 filings
  const filings = await searchNI43101Filings();

  if (filings.length === 0) {
    console.log('No NI 43-101 filings found. Exiting.');
    return;
  }

  // Step 2: Process each filing
  let processed = 0;
  const maxToProcess = 10; // Process top 10 documents

  for (const filing of filings.slice(0, maxToProcess)) {
    const companyName = filing.FilingOrganization.Names.Name.OrganizationName[0]?.Name || 'Unknown';
    const docTitle = filing.FilingDocument.DocumentSummary.DocumentTitle;
    const pageCount = filing.FilingDocument.DocumentSummary.PageCount;
    const docId = filing.FilingDocument.DocId;

    console.log(`\n${'='.repeat(70)}`);
    console.log(`📄 Processing: ${companyName}`);
    console.log(`   Title: ${docTitle}`);
    console.log(`   Pages: ${pageCount}`);
    console.log(`   Filing Date: ${filing.FilingDocument.DocumentSummary.FilingDate}`);

    // Download document
    const pdfBuffer = await downloadDocument(docId, docTitle);

    if (!pdfBuffer) {
      console.log(`   ⏭️  Skipping - download failed`);
      continue;
    }

    // Upload to Supabase
    const pdfUrl = await uploadToSupabase(pdfBuffer, companyName, docTitle);

    if (!pdfUrl) {
      console.log(`   ⏭️  Skipping - upload failed`);
      continue;
    }

    // Create project
    await createProject(filing, pdfUrl);

    processed++;

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 2000));
  }

  console.log('\n' + '='.repeat(70));
  console.log('✨ Extraction Complete!');
  console.log('='.repeat(70));
  console.log(`📥 Processed: ${processed} / ${Math.min(filings.length, maxToProcess)} documents`);
  console.log(`📊 Total NI 43-101 reports found: ${filings.length}`);
  console.log('🎯 All projects have NI 43-101 technical documentation (100+ pages)');
  console.log('='.repeat(70));
}

main().catch(console.error);
