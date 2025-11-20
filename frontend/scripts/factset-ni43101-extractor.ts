import { createClient } from '@supabase/supabase-js';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || '',
  process.env.SUPABASE_SERVICE_ROLE_KEY || ''
);

// Mining companies to search for NI 43-101 reports
const MINING_COMPANIES = [
  'Barrick Gold',
  'Newmont',
  'Agnico Eagle',
  'Kinross Gold',
  'Eldorado Gold',
  'B2Gold',
  'Lundin Mining',
  'First Quantum Minerals',
  'Hudbay Minerals',
  'Teck Resources',
  'Ivanhoe Mines',
  'Capstone Copper',
  'Copper Mountain Mining',
  'Neo Lithium',
  'Sigma Lithium',
  'Patriot Battery Metals',
  'Brunswick Exploration',
  'Foran Mining',
  'Talon Metals',
  'Trilogy Metals'
];

interface Document {
  documentId: string;
  headline: string;
  summary?: string;
  publicationDateTime: string;
  url?: string;
  source: string;
}

async function searchNI43101Documents() {
  console.log('🔍 Searching for NI 43-101 Technical Reports from FactSet...\n');

  const allDocuments: Document[] = [];

  for (const company of MINING_COMPANIES) {
    console.log(`\n📊 Searching for ${company} NI 43-101 reports...`);

    try {
      const response = await axios.post(
        'https://api.factset.com/research/news/v1/search',
        {
          query: {
            q: `"${company}" AND ("NI 43-101" OR "NI 43101" OR "Technical Report" OR "Mineral Resource" OR "Mineral Reserve") AND (NPV OR IRR OR CAPEX)`,
            fql: {
              operator: 'AND',
              children: [
                {
                  field: 'source',
                  value: 'SEDAR',
                  operator: 'eq'
                }
              ]
            }
          },
          sort: {
            field: 'publicationDateTime',
            order: 'desc'
          },
          pagination: {
            limit: 20,
            offset: 0
          }
        },
        {
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.FACTSET_API_KEY}`
          }
        }
      );

      if (response.data?.data?.documents) {
        const docs = response.data.data.documents;
        console.log(`   Found ${docs.length} documents`);
        allDocuments.push(...docs.map((doc: any) => ({
          ...doc,
          company: company
        })));
      }
    } catch (error: any) {
      console.error(`   Error searching for ${company}:`, error.response?.data || error.message);
    }

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  console.log(`\n\n📈 Total NI 43-101 documents found: ${allDocuments.length}\n`);

  return allDocuments;
}

async function downloadDocument(documentId: string, company: string): Promise<string | null> {
  try {
    console.log(`   Downloading ${documentId}...`);

    const response = await axios.get(
      `https://api.factset.com/research/news/v1/documents/${documentId}`,
      {
        headers: {
          'Accept': 'application/pdf',
          'Authorization': `Bearer ${process.env.FACTSET_API_KEY}`
        },
        responseType: 'arraybuffer'
      }
    );

    // Save to local storage first
    const companySlug = company.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    const localDir = path.join(process.cwd(), 'temp', 'ni43101', companySlug);

    if (!fs.existsSync(localDir)) {
      fs.mkdirSync(localDir, { recursive: true });
    }

    const localPath = path.join(localDir, `${documentId}.pdf`);
    fs.writeFileSync(localPath, response.data);

    // Upload to Supabase
    const storagePath = `factset/sedar/${companySlug}/${documentId}.pdf`;

    const { data, error } = await supabase.storage
      .from('refinitiv')
      .upload(storagePath, response.data, {
        contentType: 'application/pdf',
        upsert: true
      });

    if (error) {
      console.error(`   Error uploading to Supabase:`, error);
      return null;
    }

    const publicUrl = `https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/${storagePath}`;
    console.log(`   ✅ Uploaded: ${publicUrl}`);

    return publicUrl;
  } catch (error: any) {
    console.error(`   Error downloading document:`, error.response?.data || error.message);
    return null;
  }
}

async function main() {
  console.log('🚀 FactSet NI 43-101 Technical Report Extractor\n');
  console.log('='.repeat(60));

  if (!process.env.FACTSET_API_KEY) {
    console.error('❌ FACTSET_API_KEY environment variable not set');
    process.exit(1);
  }

  // Step 1: Search for NI 43-101 documents
  const documents = await searchNI43101Documents();

  if (documents.length === 0) {
    console.log('No NI 43-101 documents found. Exiting.');
    return;
  }

  // Step 2: Download and process top documents
  console.log('\n📥 Downloading top NI 43-101 documents...\n');

  const documentsToDownload = documents.slice(0, 10); // Download top 10
  const downloadedDocs: Array<{
    company: string;
    headline: string;
    url: string;
    publicationDate: string;
  }> = [];

  for (const doc of documentsToDownload) {
    console.log(`\n📄 ${doc.company}: ${doc.headline}`);
    console.log(`   Published: ${doc.publicationDateTime}`);

    const url = await downloadDocument(doc.documentId, (doc as any).company);

    if (url) {
      downloadedDocs.push({
        company: (doc as any).company,
        headline: doc.headline,
        url: url,
        publicationDate: doc.publicationDateTime
      });
    }

    // Rate limiting
    await new Promise(resolve => setTimeout(resolve, 1000));
  }

  // Step 3: Create projects from downloaded documents
  console.log('\n\n📊 Creating projects from NI 43-101 reports...\n');

  for (const doc of downloadedDocs) {
    // Get or create company
    let companyId: string | null = null;

    const { data: existingCompany } = await supabase
      .from('companies')
      .select('id')
      .eq('name', doc.company)
      .single();

    if (existingCompany) {
      companyId = existingCompany.id;
    } else {
      const { data: newCompany, error } = await supabase
        .from('companies')
        .insert({
          name: doc.company,
          description: `Mining company - NI 43-101 report available`
        })
        .select()
        .single();

      if (error) {
        console.error(`   Error creating company ${doc.company}:`, error);
        continue;
      }
      companyId = newCompany.id;
    }

    // Create project
    const projectName = `${doc.company} - ${doc.headline.substring(0, 80)}`;

    const { data: project, error: projectError } = await supabase
      .from('projects')
      .insert({
        name: projectName,
        company_id: companyId,
        description: doc.headline,
        urls: [doc.url],
        location: 'Canada', // SEDAR is Canadian
        stage: 'Exploration',
        status: 'active',
        commodities: ['To be extracted']
      })
      .select()
      .single();

    if (projectError) {
      console.error(`   Error creating project:`, projectError);
    } else {
      console.log(`   ✅ Created project: ${projectName}`);
    }
  }

  console.log('\n\n✨ NI 43-101 extraction complete!');
  console.log(`   Downloaded: ${downloadedDocs.length} documents`);
  console.log(`   Projects created: ${downloadedDocs.length}`);
}

main().catch(console.error);
