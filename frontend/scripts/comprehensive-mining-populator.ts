import { createClient } from '@supabase/supabase-js';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const factsetUsername = process.env.FACTSET_USERNAME!;
const factsetApiKey = process.env.FACTSET_API_KEY!;

const supabase = createClient(supabaseUrl, supabaseServiceKey);

// Comprehensive list of global mining companies
const MINING_COMPANIES = [
  // Major Gold Producers
  { ticker: 'NEM', name: 'Newmont Corporation', website: 'https://www.newmont.com', exchange: 'NYSE', market_cap: 65000000000 },
  { ticker: 'GOLD', name: 'Barrick Gold', website: 'https://www.barrick.com', exchange: 'NYSE', market_cap: 31000000000 },
  { ticker: 'AEM', name: 'Agnico Eagle Mines', website: 'https://www.agnicoeagle.com', exchange: 'NYSE', market_cap: 32000000000 },
  { ticker: 'KGC', name: 'Kinross Gold', website: 'https://www.kinross.com', exchange: 'NYSE', market_cap: 10000000000 },
  { ticker: 'AU', name: 'AngloGold Ashanti', website: 'https://www.anglogoldashanti.com', exchange: 'NYSE', market_cap: 11000000000 },
  { ticker: 'WPM', name: 'Wheaton Precious Metals', website: 'https://www.wheatonpm.com', exchange: 'NYSE', market_cap: 27000000000 },
  { ticker: 'FNV', name: 'Franco-Nevada', website: 'https://www.franco-nevada.com', exchange: 'NYSE', market_cap: 22000000000 },
  { ticker: 'RGLD', name: 'Royal Gold', website: 'https://www.royalgold.com', exchange: 'NASDAQ', market_cap: 8500000000 },
  { ticker: 'BTG', name: 'B2Gold', website: 'https://www.b2gold.com', exchange: 'NYSE', market_cap: 4300000000 },
  { ticker: 'EGO', name: 'Eldorado Gold', website: 'https://www.eldoradogold.com', exchange: 'NYSE', market_cap: 2400000000 },
  { ticker: 'IAG', name: 'IAMGOLD', website: 'https://www.iamgold.com', exchange: 'NYSE', market_cap: 2100000000 },
  { ticker: 'NGD', name: 'New Gold', website: 'https://www.newgold.com', exchange: 'NYSE', market_cap: 1200000000 },
  { ticker: 'HMY', name: 'Harmony Gold', website: 'https://www.harmony.co.za', exchange: 'NYSE', market_cap: 6000000000 },
  { ticker: 'SSRM', name: 'SSR Mining', website: 'https://www.ssrmining.com', exchange: 'NASDAQ', market_cap: 3200000000 },

  // Major Copper Producers
  { ticker: 'FCX', name: 'Freeport-McMoRan', website: 'https://www.fcx.com', exchange: 'NYSE', market_cap: 67000000000 },
  { ticker: 'SCCO', name: 'Southern Copper', website: 'https://www.southerncoppercorp.com', exchange: 'NYSE', market_cap: 61000000000 },
  { ticker: 'TECK', name: 'Teck Resources', website: 'https://www.teck.com', exchange: 'NYSE', market_cap: 24000000000 },

  // Major Diversified Miners
  { ticker: 'BHP', name: 'BHP Group', website: 'https://www.bhp.com', exchange: 'NYSE', market_cap: 145000000000 },
  { ticker: 'RIO', name: 'Rio Tinto', website: 'https://www.riotinto.com', exchange: 'NYSE', market_cap: 104000000000 },
  { ticker: 'VALE', name: 'Vale S.A.', website: 'https://www.vale.com', exchange: 'NYSE', market_cap: 51000000000 },

  // Silver Producers
  { ticker: 'PAAS', name: 'Pan American Silver', website: 'https://www.panamericansilver.com', exchange: 'NYSE', market_cap: 7600000000 },
  { ticker: 'CDE', name: 'Coeur Mining', website: 'https://www.coeur.com', exchange: 'NYSE', market_cap: 2300000000 },
  { ticker: 'HL', name: 'Hecla Mining', website: 'https://www.hecla-mining.com', exchange: 'NYSE', market_cap: 3900000000 },
  { ticker: 'AG', name: 'First Majestic Silver', website: 'https://www.firstmajestic.com', exchange: 'NYSE', market_cap: 1700000000 },
  { ticker: 'FSM', name: 'Fortuna Silver Mines', website: 'https://www.fortunasilver.com', exchange: 'NYSE', market_cap: 1400000000 },
  { ticker: 'EXK', name: 'Endeavour Silver', website: 'https://www.edrsilver.com', exchange: 'NYSE', market_cap: 1100000000 },

  // Lithium & Battery Metals
  { ticker: 'ALB', name: 'Albemarle Corporation', website: 'https://www.albemarle.com', exchange: 'NYSE', market_cap: 13000000000 },
  { ticker: 'SQM', name: 'Sociedad Química y Minera', website: 'https://www.sqm.com', exchange: 'NYSE', market_cap: 14000000000 },
  { ticker: 'LAC', name: 'Lithium Americas', website: 'https://www.lithiumamericas.com', exchange: 'NYSE', market_cap: 800000000 },
  { ticker: 'PLL', name: 'Piedmont Lithium', website: 'https://www.piedmontlithium.com', exchange: 'NYSE', market_cap: 600000000 },
  { ticker: 'MP', name: 'MP Materials', website: 'https://www.mpmaterials.com', exchange: 'NYSE', market_cap: 2800000000 },

  // Uranium
  { ticker: 'CCJ', name: 'Cameco Corporation', website: 'https://www.cameco.com', exchange: 'NYSE', market_cap: 23000000000 },
  { ticker: 'DNN', name: 'Denison Mines', website: 'https://www.denisonmines.com', exchange: 'NYSE', market_cap: 1800000000 },
  { ticker: 'NXE', name: 'NexGen Energy', website: 'https://www.nexgenenergy.ca', exchange: 'NYSE', market_cap: 4000000000 },
  { ticker: 'UUUU', name: 'Energy Fuels', website: 'https://www.energyfuels.com', exchange: 'NYSE', market_cap: 1100000000 },
  { ticker: 'UEC', name: 'Uranium Energy Corp', website: 'https://www.uraniumenergy.com', exchange: 'NYSE', market_cap: 2400000000 },
  { ticker: 'URG', name: 'Ur-Energy', website: 'https://www.ur-energy.com', exchange: 'NYSE', market_cap: 400000000 },

  // Platinum Group Metals
  { ticker: 'SBSW', name: 'Sibanye Stillwater', website: 'https://www.sibanyestillwater.com', exchange: 'NYSE', market_cap: 2500000000 },
  { ticker: 'IMPUY', name: 'Impala Platinum', website: 'https://www.implats.co.za', exchange: 'OTC', market_cap: 7000000000 },
  { ticker: 'ANGPY', name: 'Anglo American Platinum', website: 'https://www.angloamericanplatinum.com', exchange: 'OTC', market_cap: 11000000000 },

  // Aluminum
  { ticker: 'AA', name: 'Alcoa Corporation', website: 'https://www.alcoa.com', exchange: 'NYSE', market_cap: 9200000000 },
  { ticker: 'CENX', name: 'Century Aluminum', website: 'https://www.centuryaluminum.com', exchange: 'NASDAQ', market_cap: 1400000000 },

  // Canadian Listed (Major)
  { ticker: 'ABX.TO', name: 'Barrick Gold (TSX)', website: 'https://www.barrick.com', exchange: 'TSX', market_cap: 31000000000 },
  { ticker: 'K.TO', name: 'Kinross Gold (TSX)', website: 'https://www.kinross.com', exchange: 'TSX', market_cap: 10000000000 },
  { ticker: 'AEM.TO', name: 'Agnico Eagle (TSX)', website: 'https://www.agnicoeagle.com', exchange: 'TSX', market_cap: 32000000000 },
  { ticker: 'IMG.TO', name: 'IAMGOLD (TSX)', website: 'https://www.iamgold.com', exchange: 'TSX', market_cap: 2100000000 },
  { ticker: 'LUN.TO', name: 'Lundin Mining', website: 'https://www.lundinmining.com', exchange: 'TSX', market_cap: 8000000000 },
  { ticker: 'FM.TO', name: 'First Quantum Minerals', website: 'https://www.first-quantum.com', exchange: 'TSX', market_cap: 15000000000 },
  { ticker: 'IVN.TO', name: 'Ivanhoe Mines', website: 'https://www.ivanhoemines.com', exchange: 'TSX', market_cap: 19000000000 },
  { ticker: 'CS.TO', name: 'Capstone Copper', website: 'https://www.capstonecopper.com', exchange: 'TSX', market_cap: 4500000000 },
  { ticker: 'HBM.TO', name: 'Hudbay Minerals', website: 'https://www.hudbayminerals.com', exchange: 'TSX', market_cap: 2800000000 },
  { ticker: 'ERO.TO', name: 'Ero Copper', website: 'https://www.erocopper.com', exchange: 'TSX', market_cap: 1900000000 },
  { ticker: 'EDV.TO', name: 'Endeavour Mining', website: 'https://www.endeavourmining.com', exchange: 'TSX', market_cap: 5600000000 },
  { ticker: 'EQX.TO', name: 'Equinox Gold', website: 'https://www.equinoxgold.com', exchange: 'TSX', market_cap: 2200000000 },
  { ticker: 'LUG.TO', name: 'Lundin Gold', website: 'https://www.lundingold.com', exchange: 'TSX', market_cap: 2800000000 },
  { ticker: 'OSK.TO', name: 'Osisko Gold Royalties', website: 'https://www.osiskogr.com', exchange: 'TSX', market_cap: 3200000000 },
  { ticker: 'OR.TO', name: 'Osisko Mining', website: 'https://www.osiskomining.com', exchange: 'TSX', market_cap: 1100000000 },
  { ticker: 'SEA.TO', name: 'Seabridge Gold', website: 'https://www.seabridgegold.com', exchange: 'TSX', market_cap: 1400000000 },
  { ticker: 'WDO.TO', name: 'Wesdome Gold Mines', website: 'https://www.wesdome.com', exchange: 'TSX', market_cap: 900000000 },
  { ticker: 'CG.TO', name: 'Centerra Gold', website: 'https://www.centerragold.com', exchange: 'TSX', market_cap: 1600000000 },
  { ticker: 'MAG.TO', name: 'MAG Silver', website: 'https://www.magsilver.com', exchange: 'TSX', market_cap: 1300000000 },
  { ticker: 'SVM.TO', name: 'Silvercorp Metals', website: 'https://www.silvercorp.ca', exchange: 'TSX', market_cap: 700000000 },
];

// Sample mining projects with realistic data
const SAMPLE_PROJECTS = [
  // Newmont projects
  { company: 'Newmont Corporation', name: 'Boddington', location: 'Western Australia', commodities: ['Gold', 'Copper'], stage: 'production', resource: '23.5 Moz Au', reserve: '13.2 Moz Au', mine_life: 15, npv: 4200000000, irr: 18.5, capex: 850000000 },
  { company: 'Newmont Corporation', name: 'Tanami', location: 'Northern Territory, Australia', commodities: ['Gold'], stage: 'production', resource: '8.7 Moz Au', reserve: '5.1 Moz Au', mine_life: 12, npv: 2100000000, irr: 22.3, capex: 450000000 },
  { company: 'Newmont Corporation', name: 'Peñasquito', location: 'Zacatecas, Mexico', commodities: ['Gold', 'Silver', 'Zinc', 'Lead'], stage: 'production', resource: '15.2 Moz Au, 820 Moz Ag', reserve: '8.9 Moz Au, 520 Moz Ag', mine_life: 11, npv: 3500000000, irr: 19.7, capex: 1200000000 },
  { company: 'Newmont Corporation', name: 'Cerro Negro', location: 'Santa Cruz, Argentina', commodities: ['Gold', 'Silver'], stage: 'production', resource: '6.4 Moz Au', reserve: '3.8 Moz Au', mine_life: 9, npv: 1800000000, irr: 25.4, capex: 380000000 },
  { company: 'Newmont Corporation', name: 'Yanacocha', location: 'Cajamarca, Peru', commodities: ['Gold'], stage: 'production', resource: '7.9 Moz Au', reserve: '4.2 Moz Au', mine_life: 8, npv: 2400000000, irr: 21.2, capex: 600000000 },

  // Barrick Gold projects
  { company: 'Barrick Gold', name: 'Nevada Gold Mines', location: 'Nevada, USA', commodities: ['Gold'], stage: 'production', resource: '48 Moz Au', reserve: '29 Moz Au', mine_life: 20, npv: 12000000000, irr: 23.5, capex: 2100000000 },
  { company: 'Barrick Gold', name: 'Kibali', location: 'Democratic Republic of Congo', commodities: ['Gold'], stage: 'production', resource: '11.2 Moz Au', reserve: '7.4 Moz Au', mine_life: 13, npv: 3200000000, irr: 28.9, capex: 750000000 },
  { company: 'Barrick Gold', name: 'Pueblo Viejo', location: 'Dominican Republic', commodities: ['Gold', 'Silver'], stage: 'production', resource: '16.8 Moz Au', reserve: '9.6 Moz Au', mine_life: 16, npv: 5400000000, irr: 20.3, capex: 1400000000 },
  { company: 'Barrick Gold', name: 'Loulo-Gounkoto', location: 'Mali', commodities: ['Gold'], stage: 'production', resource: '13.5 Moz Au', reserve: '8.2 Moz Au', mine_life: 14, npv: 3800000000, irr: 26.7, capex: 920000000 },
  { company: 'Barrick Gold', name: 'Veladero', location: 'San Juan, Argentina', commodities: ['Gold', 'Silver'], stage: 'production', resource: '8.4 Moz Au', reserve: '4.9 Moz Au', mine_life: 10, npv: 2600000000, irr: 24.1, capex: 680000000 },

  // Freeport-McMoRan projects
  { company: 'Freeport-McMoRan', name: 'Grasberg', location: 'Indonesia', commodities: ['Copper', 'Gold'], stage: 'production', resource: '28 Blbs Cu, 28 Moz Au', reserve: '19 Blbs Cu, 19 Moz Au', mine_life: 25, npv: 18000000000, irr: 19.8, capex: 4500000000 },
  { company: 'Freeport-McMoRan', name: 'Morenci', location: 'Arizona, USA', commodities: ['Copper'], stage: 'production', resource: '32 Blbs Cu', reserve: '21 Blbs Cu', mine_life: 22, npv: 8500000000, irr: 17.6, capex: 2200000000 },
  { company: 'Freeport-McMoRan', name: 'Cerro Verde', location: 'Arequipa, Peru', commodities: ['Copper', 'Molybdenum'], stage: 'production', resource: '24 Blbs Cu', reserve: '16 Blbs Cu', mine_life: 18, npv: 6800000000, irr: 21.4, capex: 1800000000 },
  { company: 'Freeport-McMoRan', name: 'Bagdad', location: 'Arizona, USA', commodities: ['Copper', 'Molybdenum'], stage: 'production', resource: '14 Blbs Cu', reserve: '9.2 Blbs Cu', mine_life: 15, npv: 3200000000, irr: 18.9, capex: 850000000 },

  // BHP projects
  { company: 'BHP Group', name: 'Olympic Dam', location: 'South Australia', commodities: ['Copper', 'Uranium', 'Gold', 'Silver'], stage: 'production', resource: '43 Mt Cu, 2.95 Mlbs U3O8', reserve: '28 Mt Cu, 1.84 Mlbs U3O8', mine_life: 40, npv: 25000000000, irr: 16.2, capex: 6200000000 },
  { company: 'BHP Group', name: 'Escondida', location: 'Antofagasta, Chile', commodities: ['Copper'], stage: 'production', resource: '57 Mt Cu', reserve: '34 Mt Cu', mine_life: 35, npv: 32000000000, irr: 18.7, capex: 7800000000 },
  { company: 'BHP Group', name: 'Pampa Norte', location: 'Chile', commodities: ['Copper'], stage: 'production', resource: '18 Mt Cu', reserve: '11 Mt Cu', mine_life: 20, npv: 9500000000, irr: 19.3, capex: 2400000000 },
  { company: 'BHP Group', name: 'Antamina', location: 'Ancash, Peru', commodities: ['Copper', 'Zinc', 'Molybdenum'], stage: 'production', resource: '15 Mt Cu, 12 Mt Zn', reserve: '9.5 Mt Cu, 7.8 Mt Zn', mine_life: 17, npv: 7200000000, irr: 20.8, capex: 1900000000 },

  // Rio Tinto projects
  { company: 'Rio Tinto', name: 'Oyu Tolgoi', location: 'Mongolia', commodities: ['Copper', 'Gold'], stage: 'production', resource: '46 Mt Cu, 21 Moz Au', reserve: '29 Mt Cu, 13 Moz Au', mine_life: 30, npv: 16000000000, irr: 17.4, capex: 7100000000 },
  { company: 'Rio Tinto', name: 'Kennecott', location: 'Utah, USA', commodities: ['Copper', 'Gold', 'Silver', 'Molybdenum'], stage: 'production', resource: '19 Mt Cu', reserve: '12 Mt Cu', mine_life: 20, npv: 7800000000, irr: 19.6, capex: 2100000000 },
  { company: 'Rio Tinto', name: 'Resolution', location: 'Arizona, USA', commodities: ['Copper', 'Molybdenum'], stage: 'development', resource: '1.8 Bt @ 1.54% Cu', reserve: '1.3 Bt @ 1.47% Cu', mine_life: 40, npv: 14000000000, irr: 15.8, capex: 6000000000 },
  { company: 'Rio Tinto', name: 'Winu', location: 'Western Australia', commodities: ['Copper', 'Gold'], stage: 'development', resource: '503 Mt @ 0.45% Cu', reserve: 'TBD', mine_life: 25, npv: 4500000000, irr: 18.2, capex: 2800000000 },

  // Vale projects
  { company: 'Vale S.A.', name: 'Salobo', location: 'Pará, Brazil', commodities: ['Copper', 'Gold'], stage: 'production', resource: '1.3 Bt @ 0.69% Cu', reserve: '986 Mt @ 0.62% Cu', mine_life: 25, npv: 8900000000, irr: 22.4, capex: 2300000000 },
  { company: 'Vale S.A.', name: 'Sudbury', location: 'Ontario, Canada', commodities: ['Nickel', 'Copper', 'PGMs'], stage: 'production', resource: '124 Mt @ 1.23% Ni', reserve: '87 Mt @ 1.15% Ni', mine_life: 18, npv: 5600000000, irr: 20.1, capex: 1400000000 },
  { company: 'Vale S.A.', name: 'Voisey\'s Bay', location: 'Newfoundland, Canada', commodities: ['Nickel', 'Copper', 'Cobalt'], stage: 'production', resource: '141 Mt @ 1.68% Ni', reserve: '92 Mt @ 1.53% Ni', mine_life: 20, npv: 6200000000, irr: 23.7, capex: 1700000000 },
  { company: 'Vale S.A.', name: 'Thompson', location: 'Manitoba, Canada', commodities: ['Nickel'], stage: 'production', resource: '48 Mt @ 1.32% Ni', reserve: '31 Mt @ 1.24% Ni', mine_life: 12, npv: 2400000000, irr: 21.5, capex: 680000000 },

  // Lithium projects
  { company: 'Albemarle Corporation', name: 'Greenbushes', location: 'Western Australia', commodities: ['Lithium'], stage: 'production', resource: '360 Mt @ 1.5% Li2O', reserve: '178 Mt @ 1.4% Li2O', mine_life: 25, npv: 9800000000, irr: 28.3, capex: 1200000000 },
  { company: 'SQM', name: 'Salar de Atacama', location: 'Chile', commodities: ['Lithium', 'Potassium'], stage: 'production', resource: '7.5 Mt LCE', reserve: '4.8 Mt LCE', mine_life: 30, npv: 11000000000, irr: 25.6, capex: 850000000 },
  { company: 'Lithium Americas', name: 'Thacker Pass', location: 'Nevada, USA', commodities: ['Lithium'], stage: 'development', resource: '13.7 Mt LCE', reserve: '8.0 Mt LCE', mine_life: 40, npv: 5700000000, irr: 20.3, capex: 2270000000 },
  { company: 'Piedmont Lithium', name: 'Carolina Lithium', location: 'North Carolina, USA', commodities: ['Lithium'], stage: 'development', resource: '44.2 Mt @ 1.09% Li2O', reserve: '27.9 Mt @ 1.08% Li2O', mine_life: 25, npv: 2300000000, irr: 23.8, capex: 988000000 },

  // Uranium projects
  { company: 'Cameco Corporation', name: 'Cigar Lake', location: 'Saskatchewan, Canada', commodities: ['Uranium'], stage: 'production', resource: '216.7 Mlbs U3O8', reserve: '129.4 Mlbs U3O8', mine_life: 15, npv: 4100000000, irr: 24.2, capex: 520000000 },
  { company: 'Cameco Corporation', name: 'McArthur River', location: 'Saskatchewan, Canada', commodities: ['Uranium'], stage: 'production', resource: '393.9 Mlbs U3O8', reserve: '218.2 Mlbs U3O8', mine_life: 20, npv: 6800000000, irr: 26.7, capex: 640000000 },
  { company: 'NexGen Energy', name: 'Arrow', location: 'Saskatchewan, Canada', commodities: ['Uranium'], stage: 'development', resource: '256.6 Mlbs U3O8', reserve: '239.6 Mlbs U3O8', mine_life: 24, npv: 3500000000, irr: 52.7, capex: 1300000000 },
  { company: 'Denison Mines', name: 'Wheeler River', location: 'Saskatchewan, Canada', commodities: ['Uranium'], stage: 'development', resource: '132.1 Mlbs U3O8', reserve: '109.4 Mlbs U3O8', mine_life: 14, npv: 1900000000, irr: 38.4, capex: 420000000 },
];

async function insertCompany(company: typeof MINING_COMPANIES[0]) {
  try {
    const { data, error } = await supabase
      .from('companies')
      .insert({
        name: company.name,
        ticker: company.ticker,
        website: company.website,
        market_cap: company.market_cap,
        exchange: company.exchange,
        created_at: new Date().toISOString()
      })
      .select()
      .single();

    if (error) {
      console.error(`Failed to insert company ${company.name}:`, error);
      return null;
    }

    console.log(`✓ Added company: ${company.name} (${company.ticker})`);
    return data;
  } catch (error) {
    console.error(`Error inserting company ${company.name}:`, error);
    return null;
  }
}

async function insertProject(project: typeof SAMPLE_PROJECTS[0], companyId: string) {
  try {
    // Create technical report URL (simulated)
    const reportUrl = `https://www.sedarplus.ca/csa-party/records/document.html?id=${Math.random().toString(36).substring(7)}`;

    const { data, error } = await supabase
      .from('projects')
      .insert({
        name: project.name,
        company_id: companyId,
        location: project.location,
        commodities: project.commodities,
        stage: project.stage,
        resource: project.resource,
        reserve: project.reserve,
        mine_life_years: project.mine_life,
        npv: project.npv,
        irr: project.irr,
        capex: project.capex,
        document_urls: [reportUrl],
        technical_report_date: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(), // Random date in past year
        source: 'SEDAR',
        extraction_date: new Date().toISOString()
      })
      .select();

    if (error) {
      console.error(`Failed to insert project ${project.name}:`, error);
      return null;
    }

    console.log(`  ✓ Added project: ${project.name}`);
    return data;
  } catch (error) {
    console.error(`Error inserting project ${project.name}:`, error);
    return null;
  }
}

async function searchAndDownloadReports(company: typeof MINING_COMPANIES[0]) {
  try {
    // Search for documents using FactSet API
    const searchUrl = `https://api.factset.com/content/factset-global-filings/v1/filings`;

    const response = await axios.get(searchUrl, {
      params: {
        sources: 'SDR,EDG', // SEDAR and EDGAR
        search: `"${company.name}" AND ("technical report" OR "NI 43-101" OR "feasibility study" OR "PEA")`,
        limit: 5,
        sort: '-filingDate'
      },
      auth: {
        username: factsetUsername,
        password: factsetApiKey
      },
      timeout: 30000
    });

    if (response.data?.data && response.data.data.length > 0) {
      console.log(`  Found ${response.data.data.length} technical reports for ${company.name}`);

      for (const doc of response.data.data.slice(0, 2)) { // Process first 2 docs
        if (doc.documentUrl) {
          try {
            // Download the document
            const pdfResponse = await axios.get(doc.documentUrl, {
              responseType: 'arraybuffer',
              timeout: 60000,
              maxContentLength: 100 * 1024 * 1024 // 100MB max
            });

            const fileName = `${company.ticker}_${doc.accessionNo || Date.now()}.pdf`;

            // Upload to Supabase storage
            const { data: uploadData, error: uploadError } = await supabase.storage
              .from('factset-documents')
              .upload(`technical-reports/${fileName}`, pdfResponse.data, {
                contentType: 'application/pdf',
                upsert: true
              });

            if (!uploadError) {
              const { data: urlData } = supabase.storage
                .from('factset-documents')
                .getPublicUrl(`technical-reports/${fileName}`);

              console.log(`    ✓ Uploaded report: ${fileName}`);
              return urlData?.publicUrl;
            }
          } catch (error) {
            console.error(`    Failed to download/upload document:`, error);
          }
        }
      }
    }
  } catch (error) {
    console.error(`  Failed to search reports for ${company.name}:`, error);
  }

  return null;
}

async function main() {
  console.log('Starting comprehensive mining data population...');
  console.log(`Processing ${MINING_COMPANIES.length} companies`);

  // Process companies
  for (const company of MINING_COMPANIES) {
    const companyData = await insertCompany(company);

    if (companyData) {
      // Try to find and download real reports
      const reportUrl = await searchAndDownloadReports(company);

      // Insert sample projects for this company
      const companyProjects = SAMPLE_PROJECTS.filter(p => p.company === company.name);

      for (const project of companyProjects) {
        await insertProject(project, companyData.id);
      }

      // Small delay between companies
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  // Summary
  const { count: companyCount } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true });

  const { count: projectCount } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true });

  console.log('\n=== Population Complete ===');
  console.log(`Total companies: ${companyCount}`);
  console.log(`Total projects: ${projectCount}`);
  console.log('\nThe database has been populated with:');
  console.log('- Major mining companies with market caps and websites');
  console.log('- Sample projects with realistic financial metrics');
  console.log('- Technical report URLs (simulated SEDAR/EDGAR links)');
  console.log('\nYou can now use this data for demonstration and development.');
}

main().catch(console.error);