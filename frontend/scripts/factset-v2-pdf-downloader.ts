#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import * as pdfParse from 'pdf-parse';
import { MINING_METRICS, extractFinancialMetrics, validateDocumentMetrics } from '../lib/mining-metrics-extractor';
import crypto from 'crypto';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const factsetUsername = process.env.FACTSET_USERNAME!;
const factsetApiKey = process.env.FACTSET_API_KEY!;
const openAIKey = process.env.OPENAI_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

// FactSet V2 API endpoints
const FACTSET_BASE_URL = 'https://api.factset.com/global-filings/v2';

// Mining-specific search terms for technical reports
const TECHNICAL_REPORT_KEYWORDS = [
  'NI 43-101',
  'technical report',
  'feasibility study',
  'pre-feasibility study',
  'PEA',
  'preliminary economic assessment',
  'resource estimate',
  'reserve estimate',
  'mineral resource',
  'mineral reserve'
];

// Sources for mining documents
const MINING_SOURCES = ['SDR', 'SDRP', 'EDG', 'ASXD', 'HKEX', 'JSE SENS'];

interface FilingDocument {
  documentId: string;
  headline: string;
  filingsLink: string;
  filingsDateTime: string;
  formTypes?: string[];
  source: string;
  filingSize?: string;
  accession?: string;
  allIds?: string[];
}

interface ExtractedMetrics {
  npv?: number;
  irr?: number;
  capex?: number;
  opex?: number;
  aisc?: number;
  mineLife?: number;
  annualProduction?: string;
  resource?: string;
  reserve?: string;
  grade?: string;
  recovery?: number;
  throughput?: string;
  paybackPeriod?: number;
  ownership?: number;
}

async function getOAuthToken(): Promise<string> {
  try {
    // Use Basic Auth to get OAuth token
    const auth = Buffer.from(`${factsetUsername}:${factsetApiKey}`).toString('base64');

    // For FactSet v2, we use Basic Auth directly in headers
    return auth;
  } catch (error) {
    console.error('Failed to get OAuth token:', error);
    throw error;
  }
}

async function searchFactSetFilings(companyId: string, ticker: string): Promise<FilingDocument[]> {
  const documents: FilingDocument[] = [];
  const auth = await getOAuthToken();

  console.log(`\n🔍 Searching FactSet for ${companyId} (${ticker})...`);

  for (const source of MINING_SOURCES) {
    for (const keyword of TECHNICAL_REPORT_KEYWORDS.slice(0, 3)) { // Limit keywords to avoid too many requests
      try {
        const searchQuery = `"${ticker}" AND "${keyword}"`;

        const response = await axios.get(`${FACTSET_BASE_URL}/search`, {
          headers: {
            'Authorization': `Basic ${auth}`,
            'Accept': 'application/json'
          },
          params: {
            ids: [ticker],
            sources: [source],
            searchText: keyword,
            startDate: '20200101',
            endDate: '20241231',
            paginationLimit: 5,
            sort: '-filingsDateTime'
          }
        });

        if (response.data?.data) {
          for (const entry of response.data.data) {
            if (entry.documents) {
              for (const doc of entry.documents) {
                // Check if this is likely a technical report
                const headline = doc.headline?.toLowerCase() || '';
                const isLikelyTechnical = TECHNICAL_REPORT_KEYWORDS.some(term =>
                  headline.includes(term.toLowerCase())
                );

                if (isLikelyTechnical || (doc.filingSize && doc.filingSize.includes('MB'))) {
                  documents.push({
                    documentId: doc.document_id,
                    headline: doc.headline,
                    filingsLink: doc.filings_link,
                    filingsDateTime: doc.filings_date_time,
                    formTypes: doc.form_types,
                    source: doc.source,
                    filingSize: doc.filing_size,
                    accession: doc.accession,
                    allIds: doc.all_ids
                  });
                  console.log(`  ✓ Found: ${doc.headline.substring(0, 80)}...`);
                }
              }
            }
          }
        }
      } catch (error: any) {
        if (error.response?.status !== 404) {
          console.error(`  ⚠️ Search failed for ${source}:`, error.response?.data?.message || error.message);
        }
      }
    }
  }

  // Remove duplicates based on documentId
  const uniqueDocs = Array.from(
    new Map(documents.map(d => [d.documentId, d])).values()
  );

  console.log(`  📄 Found ${uniqueDocs.length} potential technical documents`);
  return uniqueDocs;
}

async function downloadPDF(url: string, auth: string): Promise<Buffer | null> {
  try {
    const response = await axios.get(url, {
      headers: {
        'Authorization': `Basic ${auth}`,
        'Accept': 'application/pdf,application/octet-stream,text/html'
      },
      responseType: 'arraybuffer',
      maxContentLength: 100 * 1024 * 1024, // 100MB max
      timeout: 60000 // 60 seconds
    });

    // Check if response is PDF or HTML
    const contentType = response.headers['content-type'];
    if (contentType?.includes('pdf') || contentType?.includes('octet-stream')) {
      return Buffer.from(response.data);
    } else if (contentType?.includes('html')) {
      // HTML content, try to extract PDF link from it
      const htmlContent = Buffer.from(response.data).toString('utf-8');
      const pdfLinkMatch = htmlContent.match(/href="([^"]*\.pdf[^"]*)"/i);

      if (pdfLinkMatch) {
        const pdfUrl = pdfLinkMatch[1];
        console.log(`    Redirecting to PDF: ${pdfUrl}`);
        return downloadPDF(pdfUrl, auth);
      }
    }

    return Buffer.from(response.data);
  } catch (error: any) {
    console.error(`  Failed to download: ${error.message}`);
    return null;
  }
}

async function extractMetricsFromPDF(pdfBuffer: Buffer, companyName: string, projectName: string): Promise<ExtractedMetrics> {
  const metrics: ExtractedMetrics = {};

  try {
    // Extract text from PDF
    const pdfData = await pdfParse(pdfBuffer);
    const text = pdfData.text;

    console.log(`    Extracted ${text.length} characters from PDF`);

    // Validate document contains mining metrics
    const validation = validateDocumentMetrics(text);
    if (validation.isValid) {
      console.log(`    ✓ Document validated: ${validation.percentage}% metrics coverage`);
    } else {
      console.log(`    ⚠️ Low metrics coverage: ${validation.percentage}%`);
    }

    // Use mining metrics extractor
    const extractedMetrics = extractFinancialMetrics(text);

    // Map extracted metrics to our structure
    if (extractedMetrics.post_tax_npv_usd_m) {
      metrics.npv = extractedMetrics.post_tax_npv_usd_m * 1000000;
    } else if (extractedMetrics.pre_tax_npv_usd_m) {
      metrics.npv = extractedMetrics.pre_tax_npv_usd_m * 1000000;
    }

    if (extractedMetrics.irr_percent) {
      metrics.irr = extractedMetrics.irr_percent;
    }

    if (extractedMetrics.capex_usd_m) {
      metrics.capex = extractedMetrics.capex_usd_m * 1000000;
    }

    if (extractedMetrics.opex_usd_per_tonne) {
      metrics.opex = extractedMetrics.opex_usd_per_tonne;
    }

    if (extractedMetrics.aisc_usd_per_tonne) {
      metrics.aisc = extractedMetrics.aisc_usd_per_tonne;
    }

    if (extractedMetrics.mine_life_years) {
      metrics.mineLife = extractedMetrics.mine_life_years;
    }

    if (extractedMetrics.annual_production_tonnes) {
      metrics.annualProduction = `${(extractedMetrics.annual_production_tonnes / 1000000).toFixed(2)} Mt`;
    }

    if (extractedMetrics.total_resource_tonnes) {
      metrics.resource = `${(extractedMetrics.total_resource_tonnes / 1000000).toFixed(2)} Mt`;
    }

    if (extractedMetrics.payback_years) {
      metrics.paybackPeriod = extractedMetrics.payback_years;
    }

    // Look for additional reserve and grade data
    const reserveMatch = text.match(/(?:proven|probable|total)\s*reserves?[^\d]*([\d,]+(?:\.\d+)?)\s*(?:Mt|million\s*tonnes?)/gi);
    if (reserveMatch) {
      const value = parseFloat(reserveMatch[0].replace(/[^\d.]/g, ''));
      metrics.reserve = `${value} Mt`;
    }

    const gradeMatch = text.match(/(?:average|ore|resource)?\s*grade[^\d]*([\d.]+)\s*(g\/t|%|ppm|oz\/t)/gi);
    if (gradeMatch) {
      metrics.grade = gradeMatch[0].replace(/.*grade\s*/i, '').trim();
    }

    // If metrics extraction failed, try OpenAI extraction as fallback
    if (Object.keys(metrics).length < 3 && openAIKey) {
      console.log('    Using AI extraction as fallback...');
      const aiMetrics = await extractWithOpenAI(text.substring(0, 30000), companyName, projectName);
      Object.assign(metrics, aiMetrics);
    }

    console.log(`    Extracted metrics:`, Object.keys(metrics).join(', '));
  } catch (error) {
    console.error('    PDF extraction error:', error);
  }

  return metrics;
}

async function extractWithOpenAI(text: string, companyName: string, projectName: string): Promise<ExtractedMetrics> {
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-4-turbo-preview',
        messages: [
          {
            role: 'system',
            content: 'You are a mining industry analyst. Extract key financial and technical metrics from mining technical reports. Return only the JSON object with numeric values where applicable.'
          },
          {
            role: 'user',
            content: `Extract mining project metrics from this technical report for ${companyName} - ${projectName}:

${text}

Return a JSON object with these fields (use null if not found):
- npv: Net Present Value in USD
- irr: Internal Rate of Return as percentage
- capex: Initial Capital Expenditure in USD
- opex: Operating Cost per unit
- aisc: All-in Sustaining Cost per unit
- mineLife: Mine life in years
- annualProduction: Annual production with units
- resource: Total resource with units
- reserve: Total reserve with units
- grade: Average grade with units
- recovery: Recovery rate as percentage
- throughput: Processing throughput with units
- paybackPeriod: Payback period in years
- ownership: Ownership percentage`
          }
        ],
        temperature: 0.1,
        max_tokens: 1000
      },
      {
        headers: {
          'Authorization': `Bearer ${openAIKey}`,
          'Content-Type': 'application/json'
        }
      }
    );

    const content = response.data.choices[0].message.content;
    return JSON.parse(content);
  } catch (error) {
    console.error('    OpenAI extraction failed:', error);
    return {};
  }
}

async function uploadToSupabase(pdfBuffer: Buffer, fileName: string, companyName: string): Promise<string | null> {
  try {
    const storagePath = `technical-reports/${companyName.replace(/[^a-z0-9]/gi, '_')}/${fileName}`;

    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(storagePath, pdfBuffer, {
        contentType: 'application/pdf',
        upsert: true
      });

    if (error) {
      console.error('    Storage upload error:', error);
      return null;
    }

    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(storagePath);

    return urlData?.publicUrl || null;
  } catch (error) {
    console.error('    Storage upload failed:', error);
    return null;
  }
}

async function main() {
  console.log('=== FactSet V2 PDF Downloader & Extractor ===\n');

  // Get companies from database
  const { data: companies, error: companiesError } = await supabase
    .from('companies')
    .select('id, name, ticker')
    .order('market_cap', { ascending: false })
    .limit(10); // Start with top 10 companies

  if (companiesError || !companies) {
    console.error('Failed to fetch companies:', companiesError);
    return;
  }

  console.log(`Processing ${companies.length} companies...\n`);

  let totalDocumentsProcessed = 0;
  let totalMetricsExtracted = 0;

  for (const company of companies) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Processing: ${company.name} (${company.ticker})`);
    console.log('='.repeat(60));

    // Search for documents
    const documents = await searchFactSetFilings(company.ticker, company.ticker);

    if (documents.length === 0) {
      console.log('  No technical documents found');
      continue;
    }

    // Process each document
    for (const doc of documents.slice(0, 2)) { // Limit to 2 documents per company for demo
      console.log(`\n  📄 Processing: ${doc.headline.substring(0, 80)}...`);
      console.log(`     Size: ${doc.filingSize}, Date: ${doc.filingsDateTime}`);

      const auth = await getOAuthToken();
      const pdfBuffer = await downloadPDF(doc.filingsLink, auth);

      if (!pdfBuffer) {
        console.log('    ❌ Failed to download document');
        continue;
      }

      console.log(`    ✓ Downloaded ${(pdfBuffer.length / 1024 / 1024).toFixed(2)} MB`);

      // Extract project name from headline if possible
      const projectNameMatch = doc.headline.match(/(?:for|on|at)\s+the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Project|Mine|Property)/i);
      const projectName = projectNameMatch ? projectNameMatch[1] : company.name;

      // Extract metrics
      const metrics = await extractMetricsFromPDF(pdfBuffer, company.name, projectName);

      if (Object.keys(metrics).length > 0) {
        totalMetricsExtracted++;

        // Upload to Supabase storage
        const fileName = `${doc.documentId}_${Date.now()}.pdf`;
        const storageUrl = await uploadToSupabase(pdfBuffer, fileName, company.name);

        if (storageUrl) {
          console.log('    ✓ Uploaded to storage');

          // Update or create project in database
          const projectId = crypto.randomUUID();
          const { error: projectError } = await supabase
            .from('projects')
            .upsert({
              id: projectId,
              name: projectName,
              company_id: company.id,
              location: 'See Technical Report',
              stage: 'feasibility',
              commodities: ['Gold'], // Default, should be extracted
              status: 'active',
              description: `Technical report: ${doc.headline}`,
              urls: [storageUrl],
              npv: metrics.npv,
              irr: metrics.irr,
              capex: metrics.capex,
              resource: metrics.resource,
              reserve: metrics.reserve,
              document_storage_path: storageUrl,
              is_private: false,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            });

          if (projectError) {
            console.error('    Failed to save project:', projectError.message);
          } else {
            console.log('    ✓ Project saved to database');

            // Save PDF highlights
            const highlights = [];
            if (metrics.npv) highlights.push({
              data_type: 'NPV',
              value: `$${(metrics.npv / 1000000).toFixed(0)}M`,
              page: 1,
              quote: 'Extracted from technical report'
            });
            if (metrics.irr) highlights.push({
              data_type: 'IRR',
              value: `${metrics.irr}%`,
              page: 1,
              quote: 'Extracted from technical report'
            });
            if (metrics.capex) highlights.push({
              data_type: 'CAPEX',
              value: `$${(metrics.capex / 1000000).toFixed(0)}M`,
              page: 1,
              quote: 'Extracted from technical report'
            });

            for (const highlight of highlights) {
              await supabase
                .from('pdf_highlights')
                .insert({
                  id: crypto.randomUUID(),
                  project_id: projectId,
                  ...highlight,
                  created_at: new Date().toISOString()
                });
            }

            if (highlights.length > 0) {
              console.log(`    ✓ Saved ${highlights.length} PDF highlights`);
            }
          }
        }
      }

      totalDocumentsProcessed++;
    }
  }

  // Final summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 EXTRACTION COMPLETE');
  console.log('='.repeat(60));
  console.log(`Total documents processed: ${totalDocumentsProcessed}`);
  console.log(`Documents with extracted metrics: ${totalMetricsExtracted}`);

  const { count: projectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  const { count: highlightCount } = await supabase
    .from('pdf_highlights')
    .select('*', { count: 'exact', head: true });

  console.log(`Total projects in database: ${projectCount}`);
  console.log(`Total PDF highlights: ${highlightCount}`);
}

main().catch(console.error);