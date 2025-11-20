#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import * as pdfParse from 'pdf-parse';
import { extractFinancialMetrics, validateDocumentMetrics } from '../lib/mining-metrics-extractor';
import crypto from 'crypto';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const openAIKey = process.env.OPENAI_API_KEY!;

// NEW CORRECT FACTSET CREDENTIALS
const FACTSET_USERNAME = 'LITHOS-2220379';
const FACTSET_API_KEY = '2BP7hqJOtH73DHvao1RJCQW0m1c4tVyOF81QVHxu';

const supabase = createClient(supabaseUrl, supabaseServiceKey);

// FactSet V2 API configuration
const FACTSET_BASE_URL = 'https://api.factset.com/content/global-filings/v2';

interface ExtractedData {
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
  ownership?: number;
}

// Canadian mining companies with SEDAR filings
const CANADIAN_MINING_COMPANIES = [
  { ticker: 'ABX-CA', name: 'Barrick Gold', exchange: 'TSX' },
  { ticker: 'AEM-CA', name: 'Agnico Eagle', exchange: 'TSX' },
  { ticker: 'K-CA', name: 'Kinross Gold', exchange: 'TSX' },
  { ticker: 'YRI-CA', name: 'Yamana Gold', exchange: 'TSX' },
  { ticker: 'IMG-CA', name: 'IAMGOLD', exchange: 'TSX' },
  { ticker: 'EDV-CA', name: 'Endeavour Mining', exchange: 'TSX' },
  { ticker: 'LUG-CA', name: 'Lundin Gold', exchange: 'TSX' },
  { ticker: 'OSK-CA', name: 'Osisko Gold Royalties', exchange: 'TSX' },
  { ticker: 'CCO-CA', name: 'Cameco', exchange: 'TSX' },
  { ticker: 'NXE-CA', name: 'NexGen Energy', exchange: 'TSX' },
  { ticker: 'DML-CA', name: 'Denison Mines', exchange: 'TSX' },
  { ticker: 'FM-CA', name: 'First Quantum', exchange: 'TSX' },
  { ticker: 'CS-CA', name: 'Capstone Copper', exchange: 'TSX' },
  { ticker: 'HBM-CA', name: 'Hudbay Minerals', exchange: 'TSX' },
  { ticker: 'LUN-CA', name: 'Lundin Mining', exchange: 'TSX' },
  { ticker: 'TECK.B-CA', name: 'Teck Resources', exchange: 'TSX' },
  { ticker: 'IVN-CA', name: 'Ivanhoe Mines', exchange: 'TSX' },
  { ticker: 'TKO-CA', name: 'Taseko Mines', exchange: 'TSX' }
];

async function searchFilings(ticker: string, companyName: string): Promise<any[]> {
  const filings: any[] = [];

  console.log(`\n🔍 Searching for ${companyName} (${ticker})...`);

  try {
    // Create Basic Auth header
    const auth = Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64');

    // Search for NI 43-101 technical reports
    const searchTerms = [
      'NI 43-101',
      'technical report',
      'feasibility study',
      'preliminary economic assessment',
      'PEA',
      'resource estimate',
      'mineral resource',
      'mineral reserve'
    ];

    for (const term of searchTerms.slice(0, 3)) { // Limit to avoid rate limiting
      try {
        const response = await axios.get(`${FACTSET_BASE_URL}/search`, {
          headers: {
            'Authorization': `Basic ${auth}`,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
          },
          params: {
            ids: ticker,
            sources: 'SDRP,SDR', // SEDAR and SEDAR+
            searchText: term,
            startDate: '2022-01-01',
            endDate: '2024-12-31',
            paginationLimit: 10,
            sort: '-filingsDateTime'
          },
          timeout: 30000
        });

        if (response.data?.data) {
          for (const item of response.data.data) {
            if (item.documents) {
              for (const doc of item.documents) {
                const headline = doc.headline?.toLowerCase() || '';
                const filingSize = doc.filingSize || '';

                // Filter for likely technical reports (large PDFs)
                if ((headline.includes('43-101') ||
                     headline.includes('technical report') ||
                     headline.includes('feasibility') ||
                     headline.includes('pea') ||
                     headline.includes('resource')) &&
                    (filingSize.includes('MB') || parseFloat(filingSize) > 500)) {

                  filings.push({
                    documentId: doc.documentId,
                    headline: doc.headline,
                    filingsLink: doc.filingsLink,
                    filingsDateTime: doc.filingsDateTime,
                    filingSize: doc.filingSize,
                    source: doc.source,
                    formTypes: doc.formTypes
                  });

                  console.log(`  ✓ Found: ${doc.headline.substring(0, 60)}...`);
                }
              }
            }
          }
        }
      } catch (error: any) {
        if (error.response?.status === 401) {
          console.error('  ❌ Authentication failed - check credentials');
        } else if (error.response?.status === 404) {
          console.log(`  ⚠️ No results for "${term}"`);
        } else {
          console.error(`  ⚠️ Search error:`, error.response?.data?.message || error.message);
        }
      }
    }
  } catch (error: any) {
    console.error(`  ❌ Failed to search:`, error.message);
  }

  // Remove duplicates
  const unique = Array.from(new Map(filings.map(f => [f.documentId, f])).values());
  console.log(`  📄 Total unique documents: ${unique.length}`);

  return unique;
}

async function downloadDocument(url: string): Promise<Buffer | null> {
  try {
    const auth = Buffer.from(`${FACTSET_USERNAME}:${FACTSET_API_KEY}`).toString('base64');

    const response = await axios.get(url, {
      headers: {
        'Authorization': `Basic ${auth}`,
        'Accept': 'application/pdf,application/octet-stream,text/html,*/*'
      },
      responseType: 'arraybuffer',
      maxContentLength: 200 * 1024 * 1024, // 200MB max
      maxBodyLength: 200 * 1024 * 1024,
      timeout: 120000 // 2 minutes
    });

    const contentType = response.headers['content-type'] || '';
    console.log(`    Content-Type: ${contentType}`);

    // Check if it's a PDF
    const buffer = Buffer.from(response.data);
    if (buffer.slice(0, 4).toString() === '%PDF') {
      return buffer;
    }

    // If HTML, try to extract PDF link
    if (contentType.includes('html')) {
      const html = buffer.toString('utf-8');
      const pdfMatch = html.match(/href="([^"]*\.pdf[^"]*)"/i);
      if (pdfMatch) {
        console.log(`    Following PDF link...`);
        return downloadDocument(pdfMatch[1]);
      }
    }

    return buffer;
  } catch (error: any) {
    console.error(`    Download failed: ${error.message}`);
    return null;
  }
}

async function extractFromPDF(pdfBuffer: Buffer): Promise<ExtractedData> {
  const data: ExtractedData = {};

  try {
    // Parse PDF
    const pdfData = await pdfParse(pdfBuffer);
    const text = pdfData.text;

    console.log(`    Extracted ${text.length} characters`);

    // Validate document
    const validation = validateDocumentMetrics(text);
    console.log(`    Validation: ${validation.percentage}% coverage (${validation.metricsFound}/${validation.totalMetrics} metrics)`);

    // Extract metrics
    const metrics = extractFinancialMetrics(text);

    // Map to our structure
    if (metrics.post_tax_npv_usd_m) {
      data.npv = metrics.post_tax_npv_usd_m * 1000000;
    } else if (metrics.pre_tax_npv_usd_m) {
      data.npv = metrics.pre_tax_npv_usd_m * 1000000;
    }

    if (metrics.irr_percent) data.irr = metrics.irr_percent;
    if (metrics.capex_usd_m) data.capex = metrics.capex_usd_m * 1000000;
    if (metrics.opex_usd_per_tonne) data.opex = metrics.opex_usd_per_tonne;
    if (metrics.aisc_usd_per_tonne) data.aisc = metrics.aisc_usd_per_tonne;
    if (metrics.mine_life_years) data.mineLife = metrics.mine_life_years;

    // Extract resource/reserve data
    const resourceMatch = text.match(/(?:total|measured|indicated|inferred)\s+resources?[:\s]+([0-9,]+(?:\.\d+)?)\s*(Mt|Moz|Mlb|tonnes?|ounces?)/gi);
    if (resourceMatch) {
      data.resource = resourceMatch[0].replace(/.*resources?[:\s]+/i, '').trim();
    }

    const reserveMatch = text.match(/(?:proven|probable|total)\s+reserves?[:\s]+([0-9,]+(?:\.\d+)?)\s*(Mt|Moz|Mlb|tonnes?|ounces?)/gi);
    if (reserveMatch) {
      data.reserve = reserveMatch[0].replace(/.*reserves?[:\s]+/i, '').trim();
    }

    // If insufficient data extracted, use OpenAI
    if (Object.keys(data).length < 3 && openAIKey) {
      console.log('    Using AI extraction fallback...');
      const aiData = await extractWithAI(text.substring(0, 50000));
      Object.assign(data, aiData);
    }

    console.log(`    Extracted fields: ${Object.keys(data).join(', ')}`);
  } catch (error) {
    console.error('    Extraction error:', error);
  }

  return data;
}

async function extractWithAI(text: string): Promise<ExtractedData> {
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-4-turbo-preview',
        messages: [
          {
            role: 'system',
            content: 'Extract mining project financial metrics from technical reports. Return only JSON with numeric values.'
          },
          {
            role: 'user',
            content: `Extract these metrics from the technical report:

${text}

Return JSON with:
- npv (in USD)
- irr (percentage)
- capex (in USD)
- opex (per tonne)
- mineLife (years)
- resource (with units)
- reserve (with units)

Use null if not found.`
          }
        ],
        temperature: 0.1,
        max_tokens: 500
      },
      {
        headers: {
          'Authorization': `Bearer ${openAIKey}`,
          'Content-Type': 'application/json'
        }
      }
    );

    return JSON.parse(response.data.choices[0].message.content);
  } catch (error) {
    console.error('    AI extraction failed:', error);
    return {};
  }
}

async function uploadToStorage(buffer: Buffer, fileName: string): Promise<string | null> {
  try {
    const path = `technical-reports/${fileName}`;

    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(path, buffer, {
        contentType: 'application/pdf',
        upsert: true
      });

    if (error) {
      console.error('    Storage error:', error);
      return null;
    }

    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(path);

    return urlData?.publicUrl || null;
  } catch (error) {
    console.error('    Upload failed:', error);
    return null;
  }
}

async function main() {
  console.log('=== FactSet V2 PDF Extraction Pipeline ===');
  console.log('Using updated credentials\n');

  // First, ensure we have companies in database
  const { data: existingCompanies } = await supabase
    .from('companies')
    .select('id, name, ticker')
    .in('ticker', CANADIAN_MINING_COMPANIES.map(c => c.ticker.replace('-CA', '')))
    .limit(50);

  const companyMap = new Map((existingCompanies || []).map(c => [c.ticker, c]));

  // Add missing companies
  for (const company of CANADIAN_MINING_COMPANIES) {
    const ticker = company.ticker.replace('-CA', '');
    if (!companyMap.has(ticker)) {
      const { data, error } = await supabase
        .from('companies')
        .insert({
          id: crypto.randomUUID(),
          name: company.name,
          ticker: ticker,
          exchange: company.exchange,
          website: `https://www.${company.name.toLowerCase().replace(/\s+/g, '')}.com`,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        .select()
        .single();

      if (data) {
        companyMap.set(ticker, data);
        console.log(`✓ Added company: ${company.name}`);
      }
    }
  }

  console.log(`\nProcessing ${CANADIAN_MINING_COMPANIES.length} Canadian mining companies...\n`);

  let totalDocuments = 0;
  let totalExtracted = 0;
  let totalProjects = 0;

  // Process each company
  for (const company of CANADIAN_MINING_COMPANIES.slice(0, 5)) { // Start with first 5
    console.log('='.repeat(60));

    const filings = await searchFilings(company.ticker, company.name);

    if (filings.length === 0) {
      continue;
    }

    // Process up to 2 documents per company
    for (const filing of filings.slice(0, 2)) {
      console.log(`\n  📄 ${filing.headline.substring(0, 70)}...`);
      console.log(`     Size: ${filing.filingSize}, Date: ${filing.filingsDateTime}`);

      const pdfBuffer = await downloadDocument(filing.filingsLink);

      if (!pdfBuffer) {
        console.log('    ❌ Download failed');
        continue;
      }

      console.log(`    ✓ Downloaded ${(pdfBuffer.length / 1024 / 1024).toFixed(1)} MB`);
      totalDocuments++;

      // Extract data
      const extractedData = await extractFromPDF(pdfBuffer);

      if (Object.keys(extractedData).length > 0) {
        totalExtracted++;

        // Upload to storage
        const fileName = `${company.ticker}_${filing.documentId}_${Date.now()}.pdf`;
        const storageUrl = await uploadToStorage(pdfBuffer, fileName);

        if (storageUrl) {
          console.log('    ✓ Uploaded to storage');

          // Extract project name from headline
          const projectMatch = filing.headline.match(/(?:for|on|at)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Project|Mine|Property|Deposit)/i);
          const projectName = projectMatch ? projectMatch[1] : `${company.name} Project`;

          // Get company from database
          const dbCompany = companyMap.get(company.ticker.replace('-CA', ''));

          if (dbCompany) {
            // Create or update project
            const projectId = crypto.randomUUID();
            const { error } = await supabase
              .from('projects')
              .upsert({
                id: projectId,
                name: projectName,
                company_id: dbCompany.id,
                location: 'Canada', // Default for Canadian companies
                stage: 'feasibility',
                commodities: ['Gold'], // Should be extracted
                status: 'active',
                description: `Technical Report: ${filing.headline}`,
                urls: [storageUrl],
                npv: extractedData.npv,
                irr: extractedData.irr,
                capex: extractedData.capex,
                resource: extractedData.resource,
                reserve: extractedData.reserve,
                document_storage_path: storageUrl,
                is_private: false,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString()
              });

            if (!error) {
              console.log('    ✓ Project saved');
              totalProjects++;

              // Save highlights
              const highlights = [];
              if (extractedData.npv) {
                highlights.push({
                  project_id: projectId,
                  data_type: 'NPV',
                  value: `$${(extractedData.npv / 1000000).toFixed(0)}M`,
                  page: 1,
                  quote: 'Extracted from technical report'
                });
              }
              if (extractedData.irr) {
                highlights.push({
                  project_id: projectId,
                  data_type: 'IRR',
                  value: `${extractedData.irr}%`,
                  page: 1,
                  quote: 'Extracted from technical report'
                });
              }
              if (extractedData.capex) {
                highlights.push({
                  project_id: projectId,
                  data_type: 'CAPEX',
                  value: `$${(extractedData.capex / 1000000).toFixed(0)}M`,
                  page: 1,
                  quote: 'Extracted from technical report'
                });
              }

              for (const highlight of highlights) {
                await supabase
                  .from('pdf_highlights')
                  .insert({
                    id: crypto.randomUUID(),
                    ...highlight,
                    created_at: new Date().toISOString()
                  });
              }

              if (highlights.length > 0) {
                console.log(`    ✓ Saved ${highlights.length} highlights`);
              }
            } else {
              console.error('    Project save error:', error);
            }
          }
        }
      }
    }
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 EXTRACTION SUMMARY');
  console.log('='.repeat(60));
  console.log(`Documents downloaded: ${totalDocuments}`);
  console.log(`Documents with extracted data: ${totalExtracted}`);
  console.log(`Projects created/updated: ${totalProjects}`);

  const { count: projectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  const { count: highlightCount } = await supabase
    .from('pdf_highlights')
    .select('*', { count: 'exact', head: true });

  console.log(`\nTotal projects in database: ${projectCount}`);
  console.log(`Total PDF highlights: ${highlightCount}`);
}

main().catch(console.error);