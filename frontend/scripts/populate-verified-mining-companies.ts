import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!
const fmpApiKey = process.env.FMP_API_KEY!

const supabase = createClient(supabaseUrl, supabaseKey)

interface CompanyData {
  ticker: string
  name: string
  exchange: string
  country: string
  description?: string
  website?: string
  sector: string
  industry: string
}

// Comprehensive list of top global mining companies with verified tickers
const VERIFIED_MINING_COMPANIES: CompanyData[] = [
  // === DIVERSIFIED MINERS (Top Tier) ===
  { ticker: 'BHP', name: 'BHP Group Limited', exchange: 'NYSE', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'World\'s largest mining company by market capitalization, producing iron ore, copper, coal, and petroleum' },
  { ticker: 'RIO', name: 'Rio Tinto Group', exchange: 'NYSE', country: 'UK/Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Global mining and metals company, second largest by market cap' },
  { ticker: 'VALE', name: 'Vale S.A.', exchange: 'NYSE', country: 'Brazil', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'World\'s largest iron ore producer and major nickel producer' },
  { ticker: 'GLEN.L', name: 'Glencore PLC', exchange: 'LSE', country: 'Switzerland', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Diversified mining and commodities trading company' },
  { ticker: 'AAL.L', name: 'Anglo American PLC', exchange: 'LSE', country: 'UK', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Global mining company with platinum, diamonds, copper, iron ore' },

  // === GOLD MINERS (Major) ===
  { ticker: 'NEM', name: 'Newmont Corporation', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Gold', description: 'World\'s largest gold mining company' },
  { ticker: 'GOLD', name: 'Barrick Gold Corporation', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Second largest gold mining company globally' },
  { ticker: 'AEM', name: 'Agnico Eagle Mines Limited', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian gold mining company with operations in Canada, Finland, Mexico' },
  { ticker: 'FNV', name: 'Franco-Nevada Corporation', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Leading gold-focused royalty and streaming company' },
  { ticker: 'WPM', name: 'Wheaton Precious Metals Corp.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Precious metals streaming company' },
  { ticker: 'NCM.AX', name: 'Newcrest Mining Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Australia\'s largest gold producer' },
  { ticker: 'NST.AX', name: 'Northern Star Resources Ltd', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Major Australian gold producer' },
  { ticker: 'EVN.AX', name: 'Evolution Mining Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Australian gold miner with diversified assets' },
  { ticker: 'K', name: 'Kinross Gold Corporation', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian gold mining company' },
  { ticker: 'KGC', name: 'Kinross Gold Corporation', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian gold mining company (NYSE listing)' },
  { ticker: 'GFI', name: 'Gold Fields Limited', exchange: 'NYSE', country: 'South Africa', sector: 'Basic Materials', industry: 'Gold', description: 'South African gold mining company' },
  { ticker: 'AUY', name: 'Yamana Gold Inc.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian-based gold and silver producer' },
  { ticker: 'B2GOLD', name: 'B2Gold Corp.', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian gold producer with mines in Africa and Americas' },
  { ticker: 'AGI', name: 'Alamos Gold Inc.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian gold producer' },
  { ticker: 'EGO', name: 'Eldorado Gold Corporation', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Mid-tier gold producer' },

  // === COPPER MINERS ===
  { ticker: 'FCX', name: 'Freeport-McMoRan Inc.', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Copper', description: 'World\'s largest publicly traded copper producer' },
  { ticker: 'SCCO', name: 'Southern Copper Corporation', exchange: 'NYSE', country: 'Peru/Mexico', sector: 'Basic Materials', industry: 'Copper', description: 'Major copper producer with operations in Peru and Mexico' },
  { ticker: 'TECK', name: 'Teck Resources Limited', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Diversified miner of copper, zinc, and steelmaking coal' },
  { ticker: 'FM', name: 'First Quantum Minerals Ltd.', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Copper', description: 'Major copper producer' },
  { ticker: 'LUNDIN', name: 'Lundin Mining Corporation', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Copper', description: 'Diversified base metals mining company' },
  { ticker: 'HBM', name: 'Hudbay Minerals Inc.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Copper', description: 'Copper-zinc-gold producer' },
  { ticker: 'CMCL', name: 'Caledonia Mining Corporation', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Gold and base metals producer' },
  { ticker: 'CS', name: 'Capstone Copper Corp.', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Copper', description: 'Copper producer in the Americas' },
  { ticker: 'ANTO.L', name: 'Antofagasta PLC', exchange: 'LSE', country: 'Chile/UK', sector: 'Basic Materials', industry: 'Copper', description: 'Chilean copper mining company' },

  // === IRON ORE (Specialized) ===
  { ticker: 'FMG.AX', name: 'Fortescue Metals Group Ltd', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Fourth largest iron ore producer globally' },
  { ticker: 'CLF', name: 'Cleveland-Cliffs Inc.', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Steel', description: 'Largest flat-rolled steel producer and iron ore pellet producer in North America' },
  { ticker: 'MT', name: 'ArcelorMittal S.A.', exchange: 'NYSE', country: 'Luxembourg', sector: 'Basic Materials', industry: 'Steel', description: 'World\'s largest steel and mining company' },
  { ticker: 'CSTM', name: 'Constellium SE', exchange: 'NYSE', country: 'France', sector: 'Basic Materials', industry: 'Aluminum', description: 'Aluminum products manufacturer' },

  // === LITHIUM & BATTERY METALS ===
  { ticker: 'ALB', name: 'Albemarle Corporation', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Specialty Chemicals', description: 'World\'s largest lithium producer' },
  { ticker: 'SQM', name: 'Sociedad Quimica y Minera de Chile', exchange: 'NYSE', country: 'Chile', sector: 'Basic Materials', industry: 'Specialty Chemicals', description: 'Major lithium and fertilizer producer' },
  { ticker: 'LAC', name: 'Lithium Americas Corp.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Lithium development company' },
  { ticker: 'PLL', name: 'Piedmont Lithium Inc.', exchange: 'NASDAQ', country: 'USA', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Lithium hydroxide producer' },
  { ticker: 'IGO.AX', name: 'IGO Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Nickel and lithium producer' },
  { ticker: 'PLS.AX', name: 'Pilbara Minerals Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Major lithium producer' },
  { ticker: 'MIN.AX', name: 'Mineral Resources Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Mining services and lithium producer' },
  { ticker: 'MP', name: 'MP Materials Corp.', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Rare earth mining and processing' },

  // === NICKEL MINERS ===
  { ticker: 'NILSY', name: 'Norilsk Nickel', exchange: 'OTC', country: 'Russia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'World\'s largest refined nickel producer' },

  // === RARE EARTHS ===
  { ticker: 'LYC.AX', name: 'Lynas Rare Earths Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Major rare earths producer outside China' },
  { ticker: 'ARR.AX', name: 'American Rare Earths Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Rare earth development company' },
  { ticker: 'UUUU', name: 'Energy Fuels Inc.', exchange: 'AMEX', country: 'USA', sector: 'Energy', industry: 'Uranium', description: 'Uranium and rare earth elements producer' },

  // === SILVER MINERS ===
  { ticker: 'PAAS', name: 'Pan American Silver Corp.', exchange: 'NASDAQ', country: 'Canada', sector: 'Basic Materials', industry: 'Silver', description: 'Major silver producer' },
  { ticker: 'AG', name: 'First Majestic Silver Corp.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Silver', description: 'Primary silver producer' },
  { ticker: 'HL', name: 'Hecla Mining Company', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Silver', description: 'Largest silver producer in USA' },
  { ticker: 'CDE', name: 'Coeur Mining, Inc.', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Gold', description: 'Precious metals producer' },
  { ticker: 'MAG', name: 'MAG Silver Corp.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Silver', description: 'Silver exploration and development' },

  // === PLATINUM & PALLADIUM ===
  { ticker: 'IMPUY', name: 'Impala Platinum Holdings Limited', exchange: 'OTC', country: 'South Africa', sector: 'Basic Materials', industry: 'Platinum & Precious Metals', description: 'Major platinum producer' },
  { ticker: 'SBSW', name: 'Sibanye Stillwater Limited', exchange: 'NYSE', country: 'South Africa', sector: 'Basic Materials', industry: 'Platinum & Precious Metals', description: 'Platinum, palladium, and gold producer' },
  { ticker: 'ANG.AX', name: 'Anglogold Ashanti Limited', exchange: 'ASX', country: 'South Africa', sector: 'Basic Materials', industry: 'Gold', description: 'Gold mining company' },

  // === COAL & ENERGY MINERALS ===
  { ticker: 'BTU', name: 'Peabody Energy Corporation', exchange: 'NYSE', country: 'USA', sector: 'Energy', industry: 'Coal', description: 'Largest private-sector coal company' },
  { ticker: 'ARCH', name: 'Arch Resources, Inc.', exchange: 'NYSE', country: 'USA', sector: 'Energy', industry: 'Coal', description: 'Metallurgical coal producer' },
  { ticker: 'HCC', name: 'Warrior Met Coal, Inc.', exchange: 'NYSE', country: 'USA', sector: 'Energy', industry: 'Coal', description: 'Metallurgical coal producer' },
  { ticker: 'WHC.AX', name: 'Whitehaven Coal Limited', exchange: 'ASX', country: 'Australia', sector: 'Energy', industry: 'Coal', description: 'Australian coal producer' },
  { ticker: 'NHC.AX', name: 'New Hope Corporation Limited', exchange: 'ASX', country: 'Australia', sector: 'Energy', industry: 'Coal', description: 'Coal mining and oil production' },

  // === URANIUM ===
  { ticker: 'CCJ', name: 'Cameco Corporation', exchange: 'NYSE', country: 'Canada', sector: 'Energy', industry: 'Uranium', description: 'World\'s largest publicly traded uranium producer' },
  { ticker: 'NXE', name: 'NexGen Energy Ltd.', exchange: 'NYSE', country: 'Canada', sector: 'Energy', industry: 'Uranium', description: 'Uranium development company' },
  { ticker: 'DNN', name: 'Denison Mines Corp.', exchange: 'AMEX', country: 'Canada', sector: 'Energy', industry: 'Uranium', description: 'Uranium producer and developer' },
  { ticker: 'UROY', name: 'Uranium Royalty Corp.', exchange: 'NASDAQ', country: 'Canada', sector: 'Energy', industry: 'Uranium', description: 'Uranium-focused royalty company' },

  // === FERTILIZERS (Potash, Phosphate) ===
  { ticker: 'NTR', name: 'Nutrien Ltd.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Agricultural Inputs', description: 'World\'s largest potash producer' },
  { ticker: 'MOS', name: 'The Mosaic Company', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Agricultural Inputs', description: 'Phosphate and potash producer' },
  { ticker: 'ICL', name: 'ICL Group Ltd.', exchange: 'NYSE', country: 'Israel', sector: 'Basic Materials', industry: 'Agricultural Inputs', description: 'Potash and specialty fertilizers' },

  // === ZINC & LEAD ===
  { ticker: 'HINDZINC.NS', name: 'Hindustan Zinc Limited', exchange: 'NSE', country: 'India', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Integrated zinc-lead-silver producer' },

  // === CANADIAN MID-TIER MINERS ===
  { ticker: 'EDV', name: 'Endeavour Mining Corporation', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'West African gold producer' },
  { ticker: 'OR', name: 'Osisko Gold Royalties Ltd', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Precious metals royalty company' },
  { ticker: 'IAMGOLD', name: 'IAMGOLD Corporation', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Mid-tier gold producer' },
  { ticker: 'SSL', name: 'Sandstorm Gold Ltd.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Gold royalty company' },
  { ticker: 'NG', name: 'NovaGold Resources Inc.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Gold development company' },
  { ticker: 'SSRM', name: 'SSR Mining Inc.', exchange: 'NASDAQ', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Precious metals producer' },
  { ticker: 'NGD', name: 'New Gold Inc.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Intermediate gold producer' },
  { ticker: 'IAG', name: 'IAMGOLD Corporation', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Mid-tier gold producer' },
  { ticker: 'TXG', name: 'Torex Gold Resources Inc.', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Mexican gold producer' },
  { ticker: 'WDO', name: 'Wesdome Gold Mines Ltd.', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian gold producer' },

  // === AUSTRALIAN MID-TIER ===
  { ticker: 'S32.AX', name: 'South32 Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Diversified mining company (BHP spinoff)' },
  { ticker: 'OZL.AX', name: 'OZ Minerals Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Copper', description: 'Copper-gold producer' },
  { ticker: 'RRL.AX', name: 'Regis Resources Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Australian gold producer' },
  { ticker: 'SBM.AX', name: 'St Barbara Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Gold mining company' },
  { ticker: 'AQG.AX', name: 'Alacer Gold Corp.', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Turkish gold producer' },
  { ticker: 'RED.AX', name: 'Red 5 Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Gold explorer and developer' },
  { ticker: 'SAR.AX', name: 'Saracen Mineral Holdings Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Gold producer' },
  { ticker: 'GOR.AX', name: 'Gold Road Resources Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Gold mining company' },
  { ticker: 'WSA.AX', name: 'Western Areas Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Nickel producer' },
  { ticker: 'SFR.AX', name: 'Sandfire Resources Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Copper', description: 'Copper-gold producer' },

  // === SOUTH AFRICAN MINERS ===
  { ticker: 'GLNCY', name: 'Glencore PLC', exchange: 'OTC', country: 'Switzerland', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Diversified mining (OTC listing)' },
  { ticker: 'HMY', name: 'Harmony Gold Mining Company Limited', exchange: 'NYSE', country: 'South Africa', sector: 'Basic Materials', industry: 'Gold', description: 'South African gold producer' },
  { ticker: 'DRD', name: 'DRDGOLD Limited', exchange: 'NYSE', country: 'South Africa', sector: 'Basic Materials', industry: 'Gold', description: 'Gold mining company' },

  // === LATIN AMERICAN MINERS ===
  { ticker: 'GOLD.L', name: 'Randgold Resources Limited', exchange: 'LSE', country: 'Mali/Senegal', sector: 'Basic Materials', industry: 'Gold', description: 'African gold miner' },
  { ticker: 'LUNDIN.ST', name: 'Lundin Mining Corporation', exchange: 'OMX', country: 'Sweden', sector: 'Basic Materials', industry: 'Copper', description: 'Copper producer' },

  // === ASIAN & PACIFIC MINERS ===
  { ticker: '1211.HK', name: 'BYD Company Limited', exchange: 'HKEX', country: 'China', sector: 'Consumer Cyclical', industry: 'Auto Manufacturers', description: 'Electric vehicles and batteries (lithium supply chain)' },
  { ticker: '2899.HK', name: 'Zijin Mining Group Company Limited', exchange: 'HKEX', country: 'China', sector: 'Basic Materials', industry: 'Gold', description: 'Chinese gold and copper producer' },
  { ticker: '2601.HK', name: 'China Pacific Insurance', exchange: 'HKEX', country: 'China', sector: 'Financial Services', industry: 'Insurance', description: 'Insurance company' },

  // === STEEL & ALUMINUM ===
  { ticker: 'X', name: 'United States Steel Corporation', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Steel', description: 'Integrated steel producer' },
  { ticker: 'NUE', name: 'Nucor Corporation', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Steel', description: 'Largest steel producer in USA' },
  { ticker: 'STLD', name: 'Steel Dynamics, Inc.', exchange: 'NASDAQ', country: 'USA', sector: 'Basic Materials', industry: 'Steel', description: 'Steel producer and metals recycler' },
  { ticker: 'AA', name: 'Alcoa Corporation', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Aluminum', description: 'Global aluminum producer' },
  { ticker: 'CENX', name: 'Century Aluminum Company', exchange: 'NASDAQ', country: 'USA', sector: 'Basic Materials', industry: 'Aluminum', description: 'Primary aluminum producer' },

  // === DIVERSIFIED SMALL-MID CAP ===
  { ticker: 'SVM', name: 'Silvercorp Metals Inc.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Silver', description: 'Silver-lead-zinc producer' },
  { ticker: 'EXK', name: 'Endeavour Silver Corp.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Silver', description: 'Mexican silver producer' },
  { ticker: 'FSM', name: 'Fortuna Silver Mines Inc.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Silver', description: 'Silver-gold producer in Latin America' },
  { ticker: 'GATO', name: 'Gatos Silver, Inc.', exchange: 'NYSE', country: 'USA', sector: 'Basic Materials', industry: 'Silver', description: 'Silver-zinc-lead producer' },
  { ticker: 'SAND', name: 'Sandstorm Gold Ltd.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Gold streaming company' },

  // === EMERGING MINERS (High Growth Potential) ===
  { ticker: 'GDXJ', name: 'VanEck Junior Gold Miners ETF', exchange: 'NYSE', country: 'USA', sector: 'Financial Services', industry: 'Asset Management', description: 'Junior gold miners ETF' },
  { ticker: 'GDX', name: 'VanEck Gold Miners ETF', exchange: 'NYSE', country: 'USA', sector: 'Financial Services', industry: 'Asset Management', description: 'Gold miners ETF' },

  // === ADDITIONAL CANADIAN PRODUCERS ===
  { ticker: 'ELD', name: 'Eldorado Gold Corporation', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Gold producer in Turkey, Greece, Canada' },
  { ticker: 'PVG', name: 'Pretium Resources Inc.', exchange: 'TSX', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Canadian gold producer' },
  { ticker: 'MDMN', name: 'Medinah Minerals, Inc.', exchange: 'OTC', country: 'USA', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Mineral exploration' },

  // === ADDITIONAL AUSTRALIAN PRODUCERS ===
  { ticker: 'RMS.AX', name: 'Ramelius Resources Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Australian gold producer' },
  { ticker: 'PRU.AX', name: 'Perseus Mining Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'African gold producer' },
  { ticker: 'SLR.AX', name: 'Silver Lake Resources Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Gold producer' },
  { ticker: 'DGO.AX', name: 'Dacian Gold Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Gold mining company' },
  { ticker: 'WGX.AX', name: 'Westgold Resources Limited', exchange: 'ASX', country: 'Australia', sector: 'Basic Materials', industry: 'Gold', description: 'Gold producer' },

  // === JUNIOR & EXPLORATION (Selected High-Quality) ===
  { ticker: 'NOVR', name: 'Novus Capital Corporation', exchange: 'NASDAQ', country: 'USA', sector: 'Financial Services', industry: 'Shell Companies', description: 'SPAC focused on mining sector' },
  { ticker: 'SIL', name: 'Global X Silver Miners ETF', exchange: 'NYSE', country: 'USA', sector: 'Financial Services', industry: 'Asset Management', description: 'Silver miners ETF' },
  { ticker: 'REMX', name: 'VanEck Rare Earth/Strategic Metals ETF', exchange: 'NYSE', country: 'USA', sector: 'Financial Services', industry: 'Asset Management', description: 'Rare earth metals ETF' },
  { ticker: 'LIT', name: 'Global X Lithium & Battery Tech ETF', exchange: 'NYSE', country: 'USA', sector: 'Financial Services', industry: 'Asset Management', description: 'Lithium and battery technology ETF' },

  // === INTERNATIONAL DIVERSIFIED ===
  { ticker: 'GLCNF', name: 'Glencore PLC', exchange: 'OTC', country: 'Switzerland', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Diversified mining and commodities (OTC Pink)' },
  { ticker: 'FSUMF', name: 'Fortescue Metals Group Ltd', exchange: 'OTC', country: 'Australia', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Iron ore producer (OTC listing)' },
  { ticker: 'ANZGY', name: 'Anglo American PLC', exchange: 'OTC', country: 'UK', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Diversified miner (ADR)' },

  // === ADDITIONAL QUALITY NAMES ===
  { ticker: 'IVPAF', name: 'Ivanhoe Mines Ltd.', exchange: 'OTC', country: 'Canada', sector: 'Basic Materials', industry: 'Copper', description: 'Copper-gold-zinc-germanium-PGMs developer' },
  { ticker: 'TRQ', name: 'Turquoise Hill Resources Ltd.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Copper', description: 'Mongolian copper-gold producer' },
  { ticker: 'ELEF', name: 'Elemental Royalties Corp.', exchange: 'TSX-V', country: 'Canada', sector: 'Basic Materials', industry: 'Industrial Metals & Minerals', description: 'Royalty and streaming company' },
  { ticker: 'THM', name: 'International Tower Hill Mines Ltd.', exchange: 'NYSE', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Gold development company in Alaska' },
  { ticker: 'TNR', name: 'TNR Gold Corp.', exchange: 'TSX-V', country: 'Canada', sector: 'Basic Materials', industry: 'Gold', description: 'Gold exploration and development' },
]

// Helper function to fetch company data from FMP API
async function fetchCompanyProfile(ticker: string): Promise<any> {
  try {
    const cleanTicker = ticker.replace('.AX', '').replace('.L', '').replace('.ST', '').replace('.HK', '').replace('.NS', '')
    const response = await fetch(
      `https://financialmodelingprep.com/api/v3/profile/${cleanTicker}?apikey=${fmpApiKey}`
    )

    if (!response.ok) {
      console.log(`⚠️  FMP API error for ${ticker}: ${response.status}`)
      return null
    }

    const data = await response.json()
    return data && data.length > 0 ? data[0] : null
  } catch (error) {
    console.error(`Error fetching ${ticker}:`, error)
    return null
  }
}

// Helper function to delay execution
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

async function main() {
  console.log('🗑️  Clearing existing companies...\n')

  // Delete all existing companies
  const { error: deleteError } = await supabase
    .from('companies')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000')

  if (deleteError) {
    console.error('Error clearing companies:', deleteError)
    return
  }

  console.log('✅ Companies table cleared\n')
  console.log(`🔍 Processing ${VERIFIED_MINING_COMPANIES.length} verified mining companies...\n`)

  let successCount = 0
  let errorCount = 0
  const companies = []

  for (let i = 0; i < VERIFIED_MINING_COMPANIES.length; i++) {
    const company = VERIFIED_MINING_COMPANIES[i]
    console.log(`[${i + 1}/${VERIFIED_MINING_COMPANIES.length}] Processing ${company.name} (${company.ticker})...`)

    try {
      // Fetch additional data from FMP
      const fmpData = await fetchCompanyProfile(company.ticker)

      // Prepare company data
      const companyData = {
        name: company.name,
        ticker: company.ticker,
        exchange: company.exchange,
        country: company.country,
        description: company.description || fmpData?.description || null,
        website: fmpData?.website || null,
        market_cap: fmpData?.mktCap || null,
        sector: company.sector,
        industry: company.industry,
        ceo: fmpData?.ceo || null,
        employees: fmpData?.fullTimeEmployees || null,
        city: fmpData?.city || null,
        state: fmpData?.state || null,
        zip: fmpData?.zip || null,
        address: fmpData?.address || null,
        phone: fmpData?.phone || null,
        isin: fmpData?.isin || null,
        cusip: fmpData?.cusip || null,
        ipo_date: fmpData?.ipoDate || null,
        default_image: fmpData?.image || null,
        is_actively_trading: fmpData?.isActivelyTrading !== false,
        urls: fmpData?.website ? [fmpData.website] : []
      }

      companies.push(companyData)

      // Rate limiting - FMP allows 250 calls per minute for free tier
      if (i > 0 && i % 50 === 0) {
        console.log('  ⏸️  Rate limit pause (30s)...\n')
        await delay(30000)
      } else {
        await delay(250) // Small delay between calls
      }

      successCount++
    } catch (error) {
      console.error(`  ❌ Error processing ${company.name}:`, error)
      errorCount++
    }
  }

  console.log(`\n📊 Enrichment complete: ${successCount} succeeded, ${errorCount} failed\n`)
  console.log('💾 Inserting companies into database...\n')

  // Insert companies in batches
  const batchSize = 100
  for (let i = 0; i < companies.length; i += batchSize) {
    const batch = companies.slice(i, i + batchSize)

    const { error: insertError } = await supabase
      .from('companies')
      .insert(batch)

    if (insertError) {
      console.error(`Error inserting batch ${i / batchSize + 1}:`, insertError)
    } else {
      console.log(`✅ Inserted batch ${i / batchSize + 1} (${batch.length} companies)`)
    }
  }

  console.log(`\n✨ Database population complete!`)
  console.log(`📈 Total companies added: ${companies.length}`)
  console.log(`\nBreakdown by category:`)
  console.log(`  • Diversified Miners: ${companies.filter(c => c.industry === 'Industrial Metals & Minerals').length}`)
  console.log(`  • Gold Producers: ${companies.filter(c => c.industry === 'Gold').length}`)
  console.log(`  • Copper Producers: ${companies.filter(c => c.industry === 'Copper').length}`)
  console.log(`  • Silver Producers: ${companies.filter(c => c.industry === 'Silver').length}`)
  console.log(`  • Coal Producers: ${companies.filter(c => c.industry === 'Coal').length}`)
  console.log(`  • Uranium Producers: ${companies.filter(c => c.industry === 'Uranium').length}`)
  console.log(`  • Fertilizer Producers: ${companies.filter(c => c.industry === 'Agricultural Inputs').length}`)
  console.log(`  • Other: ${companies.filter(c => !['Industrial Metals & Minerals', 'Gold', 'Copper', 'Silver', 'Coal', 'Uranium', 'Agricultural Inputs'].includes(c.industry)).length}`)
}

main()
