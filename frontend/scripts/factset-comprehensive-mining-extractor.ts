import { createClient } from '@supabase/supabase-js';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import OpenAI from 'openai';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const factsetUsername = process.env.FACTSET_USERNAME!;
const factsetApiKey = process.env.FACTSET_API_KEY!;
const openAIKey = process.env.OPENAI_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);
const openai = new OpenAI({ apiKey: openAIKey });

// Comprehensive list of major mining companies globally
const MINING_COMPANIES = [
  // Major Global Mining Companies
  { ticker: 'BHP', exchange: 'NYSE', name: 'BHP Group', website: 'https://www.bhp.com' },
  { ticker: 'RIO', exchange: 'NYSE', name: 'Rio Tinto', website: 'https://www.riotinto.com' },
  { ticker: 'VALE', exchange: 'NYSE', name: 'Vale S.A.', website: 'https://www.vale.com' },
  { ticker: 'GLNCY', exchange: 'OTC', name: 'Glencore', website: 'https://www.glencore.com' },
  { ticker: 'FCX', exchange: 'NYSE', name: 'Freeport-McMoRan', website: 'https://www.fcx.com' },
  { ticker: 'NEM', exchange: 'NYSE', name: 'Newmont Corporation', website: 'https://www.newmont.com' },
  { ticker: 'GOLD', exchange: 'NYSE', name: 'Barrick Gold', website: 'https://www.barrick.com' },
  { ticker: 'AEM', exchange: 'NYSE', name: 'Agnico Eagle Mines', website: 'https://www.agnicoeagle.com' },
  { ticker: 'KGC', exchange: 'NYSE', name: 'Kinross Gold', website: 'https://www.kinross.com' },
  { ticker: 'WPM', exchange: 'NYSE', name: 'Wheaton Precious Metals', website: 'https://www.wheatonpm.com' },
  { ticker: 'FNV', exchange: 'NYSE', name: 'Franco-Nevada', website: 'https://www.franco-nevada.com' },
  { ticker: 'PAAS', exchange: 'NYSE', name: 'Pan American Silver', website: 'https://www.panamericansilver.com' },
  { ticker: 'HL', exchange: 'NYSE', name: 'Hecla Mining', website: 'https://www.hecla-mining.com' },
  { ticker: 'CDE', exchange: 'NYSE', name: 'Coeur Mining', website: 'https://www.coeur.com' },
  { ticker: 'AG', exchange: 'NYSE', name: 'First Majestic Silver', website: 'https://www.firstmajestic.com' },
  { ticker: 'EGO', exchange: 'NYSE', name: 'Eldorado Gold', website: 'https://www.eldoradogold.com' },
  { ticker: 'IAG', exchange: 'NYSE', name: 'IAMGOLD', website: 'https://www.iamgold.com' },
  { ticker: 'NGD', exchange: 'NYSE', name: 'New Gold', website: 'https://www.newgold.com' },
  { ticker: 'BTG', exchange: 'NYSE', name: 'B2Gold', website: 'https://www.b2gold.com' },
  { ticker: 'SSRM', exchange: 'NASDAQ', name: 'SSR Mining', website: 'https://www.ssrmining.com' },
  { ticker: 'AU', exchange: 'NYSE', name: 'AngloGold Ashanti', website: 'https://www.anglogoldashanti.com' },
  { ticker: 'HMY', exchange: 'NYSE', name: 'Harmony Gold Mining', website: 'https://www.harmony.co.za' },
  { ticker: 'SBSW', exchange: 'NYSE', name: 'Sibanye Stillwater', website: 'https://www.sibanyestillwater.com' },
  { ticker: 'DRD', exchange: 'NYSE', name: 'DRDGOLD', website: 'https://www.drdgold.com' },
  { ticker: 'SCCO', exchange: 'NYSE', name: 'Southern Copper', website: 'https://www.southerncoppercorp.com' },
  { ticker: 'TECK', exchange: 'NYSE', name: 'Teck Resources', website: 'https://www.teck.com' },
  { ticker: 'AA', exchange: 'NYSE', name: 'Alcoa Corporation', website: 'https://www.alcoa.com' },
  { ticker: 'CENX', exchange: 'NASDAQ', name: 'Century Aluminum', website: 'https://www.centuryaluminum.com' },
  { ticker: 'MP', exchange: 'NYSE', name: 'MP Materials', website: 'https://www.mpmaterials.com' },
  { ticker: 'LAC', exchange: 'NYSE', name: 'Lithium Americas', website: 'https://www.lithiumamericas.com' },
  { ticker: 'ALB', exchange: 'NYSE', name: 'Albemarle Corporation', website: 'https://www.albemarle.com' },
  { ticker: 'SQM', exchange: 'NYSE', name: 'Sociedad Química y Minera', website: 'https://www.sqm.com' },
  { ticker: 'LI', exchange: 'NYSE', name: 'Li Auto', website: 'https://www.lixiang.com' },
  { ticker: 'PLL', exchange: 'NYSE', name: 'Piedmont Lithium', website: 'https://www.piedmontlithium.com' },
  { ticker: 'LTHM', exchange: 'NYSE', name: 'Livent Corporation', website: 'https://www.livent.com' },

  // Canadian Mining Companies (TSX)
  { ticker: 'ABX.TO', exchange: 'TSX', name: 'Barrick Gold (TSX)', website: 'https://www.barrick.com' },
  { ticker: 'K.TO', exchange: 'TSX', name: 'Kinross Gold (TSX)', website: 'https://www.kinross.com' },
  { ticker: 'YRI.TO', exchange: 'TSX', name: 'Yamana Gold', website: 'https://www.yamana.com' },
  { ticker: 'IMG.TO', exchange: 'TSX', name: 'IAMGOLD (TSX)', website: 'https://www.iamgold.com' },
  { ticker: 'LUN.TO', exchange: 'TSX', name: 'Lundin Mining', website: 'https://www.lundinmining.com' },
  { ticker: 'FM.TO', exchange: 'TSX', name: 'First Quantum Minerals', website: 'https://www.first-quantum.com' },
  { ticker: 'HBM.TO', exchange: 'TSX', name: 'Hudbay Minerals', website: 'https://www.hudbayminerals.com' },
  { ticker: 'CS.TO', exchange: 'TSX', name: 'Capstone Mining', website: 'https://www.capstonemining.com' },
  { ticker: 'ERO.TO', exchange: 'TSX', name: 'Ero Copper', website: 'https://www.erocopper.com' },
  { ticker: 'IVN.TO', exchange: 'TSX', name: 'Ivanhoe Mines', website: 'https://www.ivanhoemines.com' },
  { ticker: 'TKO.TO', exchange: 'TSX', name: 'Taseko Mines', website: 'https://www.tasekomines.com' },
  { ticker: 'EDV.TO', exchange: 'TSX', name: 'Endeavour Mining', website: 'https://www.endeavourmining.com' },
  { ticker: 'OSK.TO', exchange: 'TSX', name: 'Osisko Gold Royalties', website: 'https://www.osiskogr.com' },
  { ticker: 'OR.TO', exchange: 'TSX', name: 'Osisko Mining', website: 'https://www.osiskomining.com' },
  { ticker: 'WDO.TO', exchange: 'TSX', name: 'Wesdome Gold Mines', website: 'https://www.wesdome.com' },
  { ticker: 'CG.TO', exchange: 'TSX', name: 'Centerra Gold', website: 'https://www.centerragold.com' },
  { ticker: 'SEA.TO', exchange: 'TSX', name: 'Seabridge Gold', website: 'https://www.seabridgegold.com' },
  { ticker: 'NDM.TO', exchange: 'TSX', name: 'Northern Dynasty Minerals', website: 'https://www.northerndynastyminerals.com' },
  { ticker: 'WM.TO', exchange: 'TSX', name: 'Wallbridge Mining', website: 'https://www.wallbridgemining.com' },
  { ticker: 'TML.TO', exchange: 'TSX', name: 'Treasury Metals', website: 'https://www.treasurymetals.com' },
  { ticker: 'SVM.TO', exchange: 'TSX', name: 'Silvercorp Metals', website: 'https://www.silvercorp.ca' },
  { ticker: 'FR.TO', exchange: 'TSX', name: 'First Mining Gold', website: 'https://www.firstmininggold.com' },
  { ticker: 'MAG.TO', exchange: 'TSX', name: 'MAG Silver', website: 'https://www.magsilver.com' },
  { ticker: 'EQX.TO', exchange: 'TSX', name: 'Equinox Gold', website: 'https://www.equinoxgold.com' },
  { ticker: 'LUG.TO', exchange: 'TSX', name: 'Lundin Gold', website: 'https://www.lundingold.com' },

  // Australian Mining Companies (ASX)
  { ticker: 'BHP.AX', exchange: 'ASX', name: 'BHP Group (ASX)', website: 'https://www.bhp.com' },
  { ticker: 'RIO.AX', exchange: 'ASX', name: 'Rio Tinto (ASX)', website: 'https://www.riotinto.com' },
  { ticker: 'FMG.AX', exchange: 'ASX', name: 'Fortescue Metals', website: 'https://www.fortescue.com' },
  { ticker: 'NCM.AX', exchange: 'ASX', name: 'Newcrest Mining', website: 'https://www.newcrest.com' },
  { ticker: 'NST.AX', exchange: 'ASX', name: 'Northern Star Resources', website: 'https://www.nsrltd.com' },
  { ticker: 'EVN.AX', exchange: 'ASX', name: 'Evolution Mining', website: 'https://www.evolutionmining.com.au' },
  { ticker: 'SFR.AX', exchange: 'ASX', name: 'Sandfire Resources', website: 'https://www.sandfire.com.au' },
  { ticker: 'OZL.AX', exchange: 'ASX', name: 'OZ Minerals', website: 'https://www.ozminerals.com' },
  { ticker: 'MIN.AX', exchange: 'ASX', name: 'Mineral Resources', website: 'https://www.mineralresources.com.au' },
  { ticker: 'IGO.AX', exchange: 'ASX', name: 'IGO Limited', website: 'https://www.igo.com.au' },
  { ticker: 'PLS.AX', exchange: 'ASX', name: 'Pilbara Minerals', website: 'https://www.pilbaraminerals.com.au' },
  { ticker: 'ORE.AX', exchange: 'ASX', name: 'Orocobre Limited', website: 'https://www.orocobre.com' },
  { ticker: 'GXY.AX', exchange: 'ASX', name: 'Galaxy Resources', website: 'https://www.galaxyresources.com.au' },
  { ticker: 'LYC.AX', exchange: 'ASX', name: 'Lynas Rare Earths', website: 'https://www.lynasrareearths.com' },
  { ticker: 'ILU.AX', exchange: 'ASX', name: 'Iluka Resources', website: 'https://www.iluka.com' },

  // UK/London Mining Companies (LSE)
  { ticker: 'AAL.L', exchange: 'LSE', name: 'Anglo American', website: 'https://www.angloamerican.com' },
  { ticker: 'GLEN.L', exchange: 'LSE', name: 'Glencore (LSE)', website: 'https://www.glencore.com' },
  { ticker: 'RIO.L', exchange: 'LSE', name: 'Rio Tinto (LSE)', website: 'https://www.riotinto.com' },
  { ticker: 'BHP.L', exchange: 'LSE', name: 'BHP Group (LSE)', website: 'https://www.bhp.com' },
  { ticker: 'ANTO.L', exchange: 'LSE', name: 'Antofagasta', website: 'https://www.antofagasta.co.uk' },
  { ticker: 'FRES.L', exchange: 'LSE', name: 'Fresnillo', website: 'https://www.fresnilloplc.com' },
  { ticker: 'POLY.L', exchange: 'LSE', name: 'Polymetal', website: 'https://www.polymetal.com' },
  { ticker: 'CEY.L', exchange: 'LSE', name: 'Centamin', website: 'https://www.centamin.com' },
  { ticker: 'HOC.L', exchange: 'LSE', name: 'Hochschild Mining', website: 'https://www.hochschildmining.com' },
  { ticker: 'KAZ.L', exchange: 'LSE', name: 'KAZ Minerals', website: 'https://www.kazminerals.com' },

  // South African Mining Companies
  { ticker: 'ANG.JO', exchange: 'JSE', name: 'AngloGold Ashanti (JSE)', website: 'https://www.anglogoldashanti.com' },
  { ticker: 'GFI.JO', exchange: 'JSE', name: 'Gold Fields', website: 'https://www.goldfields.com' },
  { ticker: 'HAR.JO', exchange: 'JSE', name: 'Harmony Gold (JSE)', website: 'https://www.harmony.co.za' },
  { ticker: 'SSW.JO', exchange: 'JSE', name: 'Sibanye Stillwater (JSE)', website: 'https://www.sibanyestillwater.com' },
  { ticker: 'IMP.JO', exchange: 'JSE', name: 'Impala Platinum', website: 'https://www.implats.co.za' },
  { ticker: 'AMS.JO', exchange: 'JSE', name: 'Anglo American Platinum', website: 'https://www.angloamericanplatinum.com' },
  { ticker: 'NHM.JO', exchange: 'JSE', name: 'Northam Platinum', website: 'https://www.northam.co.za' },

  // Chinese Mining Companies
  { ticker: '601899.SS', exchange: 'SSE', name: 'Zijin Mining', website: 'https://www.zijinmining.com' },
  { ticker: '600111.SS', exchange: 'SSE', name: 'China Northern Rare Earth', website: 'https://www.reht.com' },
  { ticker: '000975.SZ', exchange: 'SZSE', name: 'Yintai Gold', website: 'https://www.yintaigold.com' },
  { ticker: '002155.SZ', exchange: 'SZSE', name: 'Hunan Gold', website: 'https://www.hngold.com' },
  { ticker: '600489.SS', exchange: 'SSE', name: 'Zhongjin Gold', website: 'https://www.zjgold.com' },

  // Russian Mining Companies
  { ticker: 'PLZL.MM', exchange: 'MOEX', name: 'Polyus', website: 'https://www.polyus.com' },
  { ticker: 'GMKN.MM', exchange: 'MOEX', name: 'Nornickel', website: 'https://www.nornickel.com' },
  { ticker: 'RUAL.MM', exchange: 'MOEX', name: 'Rusal', website: 'https://www.rusal.com' },
  { ticker: 'POGR.MM', exchange: 'MOEX', name: 'Petropavlovsk', website: 'https://www.petropavlovsk.net' },

  // Additional Critical Minerals Companies
  { ticker: 'CMCL', exchange: 'OTC', name: 'Caledonia Mining', website: 'https://www.caledoniamining.com' },
  { ticker: 'GORO', exchange: 'NYSE', name: 'Gold Resource Corporation', website: 'https://www.goldresourcecorp.com' },
  { ticker: 'GSS', exchange: 'NYSE', name: 'Golden Star Resources', website: 'https://www.goldenstarmining.com' },
  { ticker: 'MUX', exchange: 'NYSE', name: 'McEwen Mining', website: 'https://www.mcewenmining.com' },
  { ticker: 'AUY', exchange: 'NYSE', name: 'Yamana Gold (NYSE)', website: 'https://www.yamana.com' },
  { ticker: 'OR', exchange: 'TSX', name: 'Osisko Gold Royalties', website: 'https://www.osiskogr.com' },
  { ticker: 'SAND', exchange: 'NYSE', name: 'Sandstorm Gold', website: 'https://www.sandstormgold.com' },
  { ticker: 'RGLD', exchange: 'NASDAQ', name: 'Royal Gold', website: 'https://www.royalgold.com' },
  { ticker: 'MMX', exchange: 'TSX', name: 'Maverix Metals', website: 'https://www.maverixmetals.com' },
  { ticker: 'EXN', exchange: 'TSX', name: 'Excellon Resources', website: 'https://www.excellonresources.com' },
  { ticker: 'FSM', exchange: 'NYSE', name: 'Fortuna Silver Mines', website: 'https://www.fortunasilver.com' },
  { ticker: 'USAS', exchange: 'NYSE', name: 'Americas Gold and Silver', website: 'https://www.americasgoldandsilver.com' },
  { ticker: 'ASM', exchange: 'NYSE', name: 'Avino Silver & Gold', website: 'https://www.avino.com' },
  { ticker: 'SVM', exchange: 'NYSE', name: 'Silvercorp Metals', website: 'https://www.silvercorp.ca' },
  { ticker: 'GPL', exchange: 'NYSE', name: 'Great Panther Mining', website: 'https://www.greatpanther.com' },
  { ticker: 'EXK', exchange: 'NYSE', name: 'Endeavour Silver', website: 'https://www.edrsilver.com' },
  { ticker: 'MAG', exchange: 'NYSE', name: 'MAG Silver', website: 'https://www.magsilver.com' },
  { ticker: 'SILV', exchange: 'NYSE', name: 'SilverCrest Metals', website: 'https://www.silvercrestmetals.com' },
  { ticker: 'PPTA', exchange: 'OTC', name: 'Perpetua Resources', website: 'https://www.perpetuaresources.com' },
  { ticker: 'UUUU', exchange: 'NYSE', name: 'Energy Fuels', website: 'https://www.energyfuels.com' },
  { ticker: 'DNN', exchange: 'NYSE', name: 'Denison Mines', website: 'https://www.denisonmines.com' },
  { ticker: 'CCJ', exchange: 'NYSE', name: 'Cameco Corporation', website: 'https://www.cameco.com' },
  { ticker: 'UEC', exchange: 'NYSE', name: 'Uranium Energy Corp', website: 'https://www.uraniumenergy.com' },
  { ticker: 'URG', exchange: 'NYSE', name: 'Ur-Energy', website: 'https://www.ur-energy.com' },
  { ticker: 'NXE', exchange: 'NYSE', name: 'NexGen Energy', website: 'https://www.nexgenenergy.ca' },
  { ticker: 'PALAF', exchange: 'OTC', name: 'Paladin Energy', website: 'https://www.paladinenergy.com.au' },
  { ticker: 'BNNLF', exchange: 'OTC', name: 'Bannerman Energy', website: 'https://www.bannermanenergy.com' },
  { ticker: 'GLATF', exchange: 'OTC', name: 'Global Atomic', website: 'https://www.globalatomiccorp.com' },
  { ticker: 'FCUUF', exchange: 'OTC', name: 'Fission Uranium', website: 'https://www.fissionuranium.com' },
];

interface MiningProject {
  name: string;
  company_id: string;
  location: string;
  commodities: string[];
  stage: string;
  ownership_percentage?: number;
  resource?: string;
  reserve?: string;
  mine_life_years?: number;
  npv?: number;
  irr?: number;
  capex?: number;
  opex?: number;
  payback_years?: number;
  strip_ratio?: number;
  technical_report_date?: string;
  document_urls: string[];
  document_storage_path?: string;
  source?: string;
  extraction_date: Date;
}

// FactSet API configuration
const factsetConfig = {
  auth: {
    username: factsetUsername,
    password: factsetApiKey
  }
};

async function getMarketCap(ticker: string, exchange: string): Promise<number | null> {
  try {
    // Use FactSet Fundamentals API for market cap
    const factsetTicker = formatFactSetTicker(ticker, exchange);
    const response = await axios.post(
      'https://api.factset.com/content/factset-fundamentals/v2/metrics',
      {
        ids: [factsetTicker],
        metrics: ['FF_MKT_CAP'],
        startDate: '2024-01-01',
        endDate: '2024-12-31',
        frequency: 'QTR'
      },
      factsetConfig
    );

    if (response.data?.data?.length > 0) {
      const marketCapData = response.data.data[0];
      return marketCapData.FF_MKT_CAP || null;
    }
  } catch (error) {
    console.error(`Failed to get market cap for ${ticker}:`, error);
  }

  return null;
}

function formatFactSetTicker(ticker: string, exchange: string): string {
  // Format ticker for FactSet API based on exchange
  const exchangeMap: { [key: string]: string } = {
    'NYSE': '-US',
    'NASDAQ': '-US',
    'TSX': '-CN',
    'ASX': '-AU',
    'LSE': '-GB',
    'JSE': '-ZA',
    'SSE': '-CN',
    'SZSE': '-CN',
    'MOEX': '-RU',
    'OTC': '-US'
  };

  const suffix = exchangeMap[exchange] || '';
  return ticker.replace('.TO', '').replace('.AX', '').replace('.L', '').replace('.JO', '').replace('.SS', '').replace('.SZ', '').replace('.MM', '') + suffix;
}

async function searchFactSetDocuments(companyTicker: string, exchange: string): Promise<any[]> {
  try {
    const factsetTicker = formatFactSetTicker(companyTicker, exchange);

    // Search for technical reports using FactSet Global Filings V2
    const searchRequest = {
      query: {
        and: [
          { equals: { field: 'ticker', value: factsetTicker } },
          { or: [
            { contains: { field: 'title', value: 'technical report' } },
            { contains: { field: 'title', value: 'NI 43-101' } },
            { contains: { field: 'title', value: 'feasibility study' } },
            { contains: { field: 'title', value: 'resource estimate' } },
            { contains: { field: 'title', value: 'reserve estimate' } },
            { contains: { field: 'title', value: 'PEA' } },
            { contains: { field: 'title', value: 'preliminary economic assessment' } },
            { contains: { field: 'title', value: 'mineral resource' } },
            { contains: { field: 'title', value: 'mineral reserve' } },
            { contains: { field: 'title', value: 'mining project' } }
          ]}
        ]
      },
      limit: 50,
      sort: [{ field: 'filingDate', direction: 'desc' }]
    };

    const response = await axios.post(
      'https://api.factset.com/global-filings/v2/search',
      searchRequest,
      factsetConfig
    );

    return response.data?.data || [];
  } catch (error) {
    console.error(`Failed to search documents for ${companyTicker}:`, error);
    return [];
  }
}

async function downloadDocument(documentId: string, fileName: string): Promise<Buffer | null> {
  try {
    const response = await axios.get(
      `https://api.factset.com/global-filings/v2/documents/${documentId}/download`,
      {
        ...factsetConfig,
        responseType: 'arraybuffer'
      }
    );

    return Buffer.from(response.data);
  } catch (error) {
    console.error(`Failed to download document ${documentId}:`, error);
    return null;
  }
}

async function uploadToSupabaseStorage(file: Buffer, fileName: string): Promise<string | null> {
  try {
    const { data, error } = await supabase.storage
      .from('factset-documents')
      .upload(`technical-reports/${fileName}`, file, {
        contentType: 'application/pdf',
        upsert: true
      });

    if (error) {
      console.error('Storage upload error:', error);
      return null;
    }

    // Get public URL
    const { data: urlData } = supabase.storage
      .from('factset-documents')
      .getPublicUrl(`technical-reports/${fileName}`);

    return urlData?.publicUrl || null;
  } catch (error) {
    console.error('Failed to upload to Supabase:', error);
    return null;
  }
}

async function extractProjectDataFromPDF(pdfBuffer: Buffer, companyName: string): Promise<Partial<MiningProject> | null> {
  try {
    // Convert PDF to base64 for OpenAI
    const base64PDF = pdfBuffer.toString('base64');

    // Use OpenAI to extract structured data
    const response = await openai.chat.completions.create({
      model: 'gpt-4-vision-preview',
      messages: [
        {
          role: 'system',
          content: `Extract mining project information from this technical report. Return a JSON object with:
          - name: Project name
          - location: Project location (country, state/province)
          - commodities: Array of commodities
          - stage: Development stage (exploration, development, production)
          - resource: Measured + Indicated resources
          - reserve: Proven + Probable reserves
          - mine_life_years: Expected mine life
          - npv: Net Present Value (number in USD)
          - irr: Internal Rate of Return (percentage)
          - capex: Capital expenditure (number in USD)
          - opex: Operating expenditure (per unit)
          - payback_years: Payback period
          - ownership_percentage: Company ownership %
          - technical_report_date: Date of report

          Return only valid JSON, no markdown.`
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: `Extract project data from this ${companyName} technical report`
            }
          ]
        }
      ],
      max_tokens: 2000
    });

    const extractedData = JSON.parse(response.choices[0].message.content || '{}');
    return extractedData;
  } catch (error) {
    console.error('Failed to extract project data:', error);
    return null;
  }
}

async function processCompany(company: typeof MINING_COMPANIES[0]) {
  console.log(`\nProcessing ${company.name} (${company.ticker})...`);

  // Get market cap
  const marketCap = await getMarketCap(company.ticker, company.exchange);

  // Insert company into database
  const { data: companyData, error: companyError } = await supabase
    .from('companies')
    .insert({
      name: company.name,
      ticker: company.ticker,
      website: company.website,
      market_cap: marketCap,
      exchange: company.exchange,
      sector: 'Mining',
      created_at: new Date().toISOString()
    })
    .select()
    .single();

  if (companyError) {
    console.error(`Failed to insert company ${company.name}:`, companyError);
    return;
  }

  console.log(`✓ Inserted company: ${company.name}`);

  // Search for technical documents
  const documents = await searchFactSetDocuments(company.ticker, company.exchange);
  console.log(`Found ${documents.length} technical documents`);

  // Process each document
  for (const doc of documents.slice(0, 5)) { // Limit to 5 most recent per company
    try {
      const fileName = `${company.ticker}_${doc.documentId}_${doc.filingDate}.pdf`;

      // Download document
      const pdfBuffer = await downloadDocument(doc.documentId, fileName);
      if (!pdfBuffer) continue;

      // Upload to Supabase storage
      const storageUrl = await uploadToSupabaseStorage(pdfBuffer, fileName);
      if (!storageUrl) continue;

      // Extract project data
      const projectData = await extractProjectDataFromPDF(pdfBuffer, company.name);
      if (!projectData?.name) continue;

      // Insert project into database
      const { error: projectError } = await supabase
        .from('projects')
        .insert({
          ...projectData,
          company_id: companyData.id,
          document_urls: [doc.url],
          document_storage_path: storageUrl,
          source: 'FactSet',
          extraction_date: new Date().toISOString()
        });

      if (projectError) {
        console.error(`Failed to insert project:`, projectError);
      } else {
        console.log(`  ✓ Added project: ${projectData.name}`);
      }

      // Add a small delay to avoid rate limits
      await new Promise(resolve => setTimeout(resolve, 2000));
    } catch (error) {
      console.error(`Failed to process document ${doc.documentId}:`, error);
    }
  }
}

async function main() {
  console.log('Starting comprehensive mining data extraction...');
  console.log(`Processing ${MINING_COMPANIES.length} mining companies`);

  // Process companies in batches
  const batchSize = 5;
  for (let i = 0; i < MINING_COMPANIES.length; i += batchSize) {
    const batch = MINING_COMPANIES.slice(i, i + batchSize);

    await Promise.all(batch.map(company => processCompany(company)));

    console.log(`\nCompleted batch ${Math.floor(i / batchSize) + 1} of ${Math.ceil(MINING_COMPANIES.length / batchSize)}`);

    // Longer delay between batches
    if (i + batchSize < MINING_COMPANIES.length) {
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }

  // Final summary
  const { count: companyCount } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true });

  const { count: projectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('\n=== Extraction Complete ===');
  console.log(`Total companies added: ${companyCount}`);
  console.log(`Total projects added: ${projectCount}`);
}

main().catch(console.error);