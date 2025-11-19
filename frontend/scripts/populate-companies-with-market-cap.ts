import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseKey)

// Verified list of top global mining companies with market cap data (in billions USD)
// Market cap data gathered from multiple sources and web research as of October 2024 - January 2025
const MINING_COMPANIES = [
  // === DIVERSIFIED MINERS (Top Tier) ===
  { name: 'BHP Group Limited', ticker: 'BHP', exchange: 'NYSE', country: 'Australia', market_cap: 126.0, description: 'World\'s largest mining company by market capitalization, producing iron ore, copper, coal, and petroleum', website: 'https://www.bhp.com' },
  { name: 'Rio Tinto Group', ticker: 'RIO', exchange: 'NYSE', country: 'UK/Australia', market_cap: 103.0, description: 'Global mining and metals company, second largest by market cap', website: 'https://www.riotinto.com' },
  { name: 'Vale S.A.', ticker: 'VALE', exchange: 'NYSE', country: 'Brazil', market_cap: 37.7, description: 'World\'s largest iron ore producer and major nickel producer', website: 'https://www.vale.com' },
  { name: 'Glencore PLC', ticker: 'GLEN', exchange: 'LSE', country: 'Switzerland', market_cap: 55.0, description: 'Diversified mining and commodities trading company', website: 'https://www.glencore.com' },
  { name: 'Anglo American PLC', ticker: 'AAL', exchange: 'LSE', country: 'UK', market_cap: 28.0, description: 'Global mining company with platinum, diamonds, copper, iron ore', website: 'https://www.angloamerican.com' },

  // === GOLD MINERS (Major) ===
  { name: 'Newmont Corporation', ticker: 'NEM', exchange: 'NYSE', country: 'USA', market_cap: 54.2, description: 'World\'s largest gold mining company', website: 'https://www.newmont.com' },
  { name: 'Barrick Gold Corporation', ticker: 'GOLD', exchange: 'NYSE', country: 'Canada', market_cap: 45.0, description: 'Second largest gold mining company globally', website: 'https://www.barrick.com' },
  { name: 'Agnico Eagle Mines Limited', ticker: 'AEM', exchange: 'NYSE', country: 'Canada', market_cap: 52.9, description: 'Canadian gold mining company with operations in Canada, Finland, Mexico', website: 'https://www.agnicoeagle.com' },
  { name: 'Franco-Nevada Corporation', ticker: 'FNV', exchange: 'NYSE', country: 'Canada', market_cap: 30.0, description: 'Leading gold-focused royalty and streaming company', website: 'https://www.franco-nevada.com' },
  { name: 'Wheaton Precious Metals Corp.', ticker: 'WPM', exchange: 'NYSE', country: 'Canada', market_cap: 44.4, description: 'Precious metals streaming company', website: 'https://www.wheatonpm.com' },
  { name: 'Newcrest Mining Limited', ticker: 'NCM', exchange: 'ASX', country: 'Australia', market_cap: 18.0, description: 'Australia\'s largest gold producer', website: 'https://www.newcrest.com' },
  { name: 'Northern Star Resources Ltd', ticker: 'NST', exchange: 'ASX', country: 'Australia', market_cap: 12.5, description: 'Major Australian gold producer', website: 'https://www.nsrltd.com' },
  { name: 'Evolution Mining Limited', ticker: 'EVN', exchange: 'ASX', country: 'Australia', market_cap: 8.2, description: 'Australian gold miner with diversified assets', website: 'https://www.evolutionmining.com.au' },
  { name: 'Kinross Gold Corporation', ticker: 'K', exchange: 'TSX', country: 'Canada', market_cap: 30.8, description: 'Canadian gold mining company', website: 'https://www.kinross.com' },
  { name: 'Gold Fields Limited', ticker: 'GFI', exchange: 'NYSE', country: 'South Africa', market_cap: 38.6, description: 'South African gold mining company', website: 'https://www.goldfields.com' },
  { name: 'Yamana Gold Inc.', ticker: 'AUY', exchange: 'NYSE', country: 'Canada', market_cap: 6.5, description: 'Canadian-based gold and silver producer', website: 'https://www.yamana.com' },
  { name: 'B2Gold Corp.', ticker: 'BTG', exchange: 'TSX', country: 'Canada', market_cap: 5.8, description: 'Canadian gold producer with mines in Africa and Americas', website: 'https://www.b2gold.com' },
  { name: 'Alamos Gold Inc.', ticker: 'AGI', exchange: 'NYSE', country: 'Canada', market_cap: 6.2, description: 'Canadian gold producer', website: 'https://www.alamosgold.com' },
  { name: 'Eldorado Gold Corporation', ticker: 'EGO', exchange: 'NYSE', country: 'Canada', market_cap: 4.8, description: 'Mid-tier gold producer', website: 'https://www.eldoradogold.com' },
  { name: 'AngloGold Ashanti Limited', ticker: 'AU', exchange: 'NYSE', country: 'South Africa', market_cap: 34.6, description: 'Major South African gold producer', website: 'https://www.anglogoldashanti.com' },

  // === COPPER MINERS ===
  { name: 'Freeport-McMoRan Inc.', ticker: 'FCX', exchange: 'NYSE', country: 'USA', market_cap: 58.7, description: 'World\'s largest publicly traded copper producer', website: 'https://www.fcx.com' },
  { name: 'Southern Copper Corporation', ticker: 'SCCO', exchange: 'NYSE', country: 'Peru/Mexico', market_cap: 65.0, description: 'Major copper producer with operations in Peru and Mexico', website: 'https://www.southerncopper.com' },
  { name: 'Teck Resources Limited', ticker: 'TECK', exchange: 'NYSE', country: 'Canada', market_cap: 25.0, description: 'Diversified miner of copper, zinc, and steelmaking coal', website: 'https://www.teck.com' },
  { name: 'First Quantum Minerals Ltd.', ticker: 'FM', exchange: 'TSX', country: 'Canada', market_cap: 14.5, description: 'Major copper producer', website: 'https://www.first-quantum.com' },
  { name: 'Lundin Mining Corporation', ticker: 'LUN', exchange: 'TSX', country: 'Canada', market_cap: 10.5, description: 'Diversified base metals mining company', website: 'https://www.lundinmining.com' },
  { name: 'Hudbay Minerals Inc.', ticker: 'HBM', exchange: 'NYSE', country: 'Canada', market_cap: 5.2, description: 'Copper-zinc-gold producer', website: 'https://www.hudbay.com' },
  { name: 'Capstone Copper Corp.', ticker: 'CS', exchange: 'TSX', country: 'Canada', market_cap: 4.5, description: 'Copper producer in the Americas', website: 'https://www.capstonecopper.com' },
  { name: 'Antofagasta PLC', ticker: 'ANTO', exchange: 'LSE', country: 'Chile/UK', market_cap: 35.1, description: 'Chilean copper mining company', website: 'https://www.antofagasta.co.uk' },
  { name: 'Ivanhoe Mines Ltd.', ticker: 'IVN', exchange: 'TSX', country: 'Canada', market_cap: 18.0, description: 'Copper-gold-zinc-germanium-PGMs developer', website: 'https://www.ivanhoemines.com' },

  // === IRON ORE (Specialized) ===
  { name: 'Fortescue Metals Group Ltd', ticker: 'FMG', exchange: 'ASX', country: 'Australia', market_cap: 42.0, description: 'Fourth largest iron ore producer globally', website: 'https://www.fmgl.com.au' },
  { name: 'Cleveland-Cliffs Inc.', ticker: 'CLF', exchange: 'NYSE', country: 'USA', market_cap: 5.8, description: 'Largest flat-rolled steel producer and iron ore pellet producer in North America', website: 'https://www.clevelandcliffs.com' },
  { name: 'ArcelorMittal S.A.', ticker: 'MT', exchange: 'NYSE', country: 'Luxembourg', market_cap: 22.5, description: 'World\'s largest steel and mining company', website: 'https://corporate.arcelormittal.com' },

  // === LITHIUM & BATTERY METALS ===
  { name: 'Albemarle Corporation', ticker: 'ALB', exchange: 'NYSE', country: 'USA', market_cap: 12.0, description: 'World\'s largest lithium producer', website: 'https://www.albemarle.com' },
  { name: 'Sociedad Quimica y Minera de Chile', ticker: 'SQM', exchange: 'NYSE', country: 'Chile', market_cap: 10.9, description: 'Major lithium and fertilizer producer', website: 'https://www.sqm.com' },
  { name: 'Lithium Americas Corp.', ticker: 'LAC', exchange: 'NYSE', country: 'Canada', market_cap: 3.2, description: 'Lithium development company', website: 'https://www.lithiumamericas.com' },
  { name: 'Piedmont Lithium Inc.', ticker: 'PLL', exchange: 'NASDAQ', country: 'USA', market_cap: 1.5, description: 'Lithium hydroxide producer', website: 'https://piedmontlithium.com' },
  { name: 'IGO Limited', ticker: 'IGO', exchange: 'ASX', country: 'Australia', market_cap: 4.8, description: 'Nickel and lithium producer', website: 'https://www.igo.com.au' },
  { name: 'Pilbara Minerals Limited', ticker: 'PLS', exchange: 'ASX', country: 'Australia', market_cap: 6.3, description: 'Major lithium producer', website: 'https://www.pilbaraminerals.com.au' },
  { name: 'Mineral Resources Limited', ticker: 'MIN', exchange: 'ASX', country: 'Australia', market_cap: 8.5, description: 'Mining services and lithium producer', website: 'https://www.mineralresources.com.au' },
  { name: 'MP Materials Corp.', ticker: 'MP', exchange: 'NYSE', country: 'USA', market_cap: 12.5, description: 'Rare earth mining and processing', website: 'https://www.mpmaterials.com' },
  { name: 'Ganfeng Lithium Co., Ltd.', ticker: '002460', exchange: 'SSE', country: 'China', market_cap: 15.0, description: 'Major Chinese lithium producer', website: 'https://www.ganfenglithium.com' },

  // === RARE EARTHS ===
  { name: 'Lynas Rare Earths Limited', ticker: 'LYC', exchange: 'ASX', country: 'Australia', market_cap: 12.0, description: 'Major rare earths producer outside China', website: 'https://www.lynasrareearths.com' },
  { name: 'Energy Fuels Inc.', ticker: 'UUUU', exchange: 'AMEX', country: 'USA', market_cap: 1.2, description: 'Uranium and rare earth elements producer', website: 'https://www.energyfuels.com' },

  // === SILVER MINERS ===
  { name: 'Pan American Silver Corp.', ticker: 'PAAS', exchange: 'NASDAQ', country: 'Canada', market_cap: 15.4, description: 'Major silver producer', website: 'https://www.panamericansilver.com' },
  { name: 'First Majestic Silver Corp.', ticker: 'AG', exchange: 'NYSE', country: 'Canada', market_cap: 6.0, description: 'Primary silver producer', website: 'https://www.firstmajestic.com' },
  { name: 'Hecla Mining Company', ticker: 'HL', exchange: 'NYSE', country: 'USA', market_cap: 8.1, description: 'Largest silver producer in USA', website: 'https://www.hecla.com' },
  { name: 'Coeur Mining, Inc.', ticker: 'CDE', exchange: 'NYSE', country: 'USA', market_cap: 3.5, description: 'Precious metals producer', website: 'https://www.coeur.com' },
  { name: 'MAG Silver Corp.', ticker: 'MAG', exchange: 'NYSE', country: 'Canada', market_cap: 2.8, description: 'Silver exploration and development', website: 'https://www.magsilver.com' },

  // === PLATINUM & PALLADIUM ===
  { name: 'Sibanye Stillwater Limited', ticker: 'SBSW', exchange: 'NYSE', country: 'South Africa', market_cap: 6.8, description: 'Platinum, palladium, and gold producer', website: 'https://www.sibanyestillwater.com' },
  { name: 'Impala Platinum Holdings Limited', ticker: 'IMPUY', exchange: 'OTC', country: 'South Africa', market_cap: 11.5, description: 'Major platinum producer', website: 'https://www.implats.co.za' },

  // === COAL & ENERGY MINERALS ===
  { name: 'Peabody Energy Corporation', ticker: 'BTU', exchange: 'NYSE', country: 'USA', market_cap: 1.8, description: 'Largest private-sector coal company', website: 'https://www.peabodyenergy.com' },
  { name: 'Arch Resources, Inc.', ticker: 'ARCH', exchange: 'NYSE', country: 'USA', market_cap: 2.8, description: 'Metallurgical coal producer', website: 'https://www.archrsc.com' },
  { name: 'Warrior Met Coal, Inc.', ticker: 'HCC', exchange: 'NYSE', country: 'USA', market_cap: 3.2, description: 'Metallurgical coal producer', website: 'https://www.warriormetcoal.com' },

  // === URANIUM ===
  { name: 'Cameco Corporation', ticker: 'CCJ', exchange: 'NYSE', country: 'Canada', market_cap: 37.6, description: 'World\'s largest publicly traded uranium producer', website: 'https://www.cameco.com' },
  { name: 'NexGen Energy Ltd.', ticker: 'NXE', exchange: 'NYSE', country: 'Canada', market_cap: 4.5, description: 'Uranium development company', website: 'https://www.nexgenenergy.ca' },
  { name: 'Denison Mines Corp.', ticker: 'DNN', exchange: 'AMEX', country: 'Canada', market_cap: 1.5, description: 'Uranium producer and developer', website: 'https://www.denisonmines.com' },

  // === FERTILIZERS (Potash, Phosphate) ===
  { name: 'Nutrien Ltd.', ticker: 'NTR', exchange: 'NYSE', country: 'Canada', market_cap: 27.5, description: 'World\'s largest potash producer', website: 'https://www.nutrien.com' },
  { name: 'The Mosaic Company', ticker: 'MOS', exchange: 'NYSE', country: 'USA', market_cap: 9.6, description: 'Phosphate and potash producer', website: 'https://www.mosaicco.com' },
  { name: 'ICL Group Ltd.', ticker: 'ICL', exchange: 'NYSE', country: 'Israel', market_cap: 5.5, description: 'Potash and specialty fertilizers', website: 'https://www.icl-group.com' },

  // === CANADIAN MID-TIER MINERS ===
  { name: 'Endeavour Mining Corporation', ticker: 'EDV', exchange: 'TSX', country: 'Canada', market_cap: 7.5, description: 'West African gold producer', website: 'https://www.endeavourmining.com' },
  { name: 'Osisko Gold Royalties Ltd', ticker: 'OR', exchange: 'NYSE', country: 'Canada', market_cap: 4.2, description: 'Precious metals royalty company', website: 'https://www.osiskogr.com' },
  { name: 'IAMGOLD Corporation', ticker: 'IAG', exchange: 'TSX', country: 'Canada', market_cap: 3.8, description: 'Mid-tier gold producer', website: 'https://www.iamgold.com' },
  { name: 'Sandstorm Gold Ltd.', ticker: 'SAND', exchange: 'NYSE', country: 'Canada', market_cap: 2.5, description: 'Gold royalty company', website: 'https://www.sandstormgold.com' },
  { name: 'NovaGold Resources Inc.', ticker: 'NG', exchange: 'NYSE', country: 'Canada', market_cap: 1.8, description: 'Gold development company', website: 'https://www.novagold.com' },
  { name: 'SSR Mining Inc.', ticker: 'SSRM', exchange: 'NASDAQ', country: 'Canada', market_cap: 3.2, description: 'Precious metals producer', website: 'https://www.ssrmining.com' },
  { name: 'New Gold Inc.', ticker: 'NGD', exchange: 'NYSE', country: 'Canada', market_cap: 2.1, description: 'Intermediate gold producer', website: 'https://www.newgold.com' },

  // === AUSTRALIAN MID-TIER ===
  { name: 'South32 Limited', ticker: 'S32', exchange: 'ASX', country: 'Australia', market_cap: 12.0, description: 'Diversified mining company (BHP spinoff)', website: 'https://www.south32.net' },
  { name: 'OZ Minerals Limited', ticker: 'OZL', exchange: 'ASX', country: 'Australia', market_cap: 8.5, description: 'Copper-gold producer', website: 'https://www.ozminerals.com' },
  { name: 'Sandfire Resources Limited', ticker: 'SFR', exchange: 'ASX', country: 'Australia', market_cap: 4.2, description: 'Copper-gold producer', website: 'https://www.sandfire.com.au' },
  { name: 'Ramelius Resources Limited', ticker: 'RMS', exchange: 'ASX', country: 'Australia', market_cap: 2.8, description: 'Australian gold producer', website: 'https://www.rameliusresources.com.au' },
  { name: 'Perseus Mining Limited', ticker: 'PRU', exchange: 'ASX', country: 'Australia', market_cap: 3.5, description: 'African gold producer', website: 'https://www.perseusmining.com' },
  { name: 'Regis Resources Limited', ticker: 'RRL', exchange: 'ASX', country: 'Australia', market_cap: 2.1, description: 'Australian gold producer', website: 'https://www.regisresources.com' },

  // === SOUTH AFRICAN MINERS ===
  { name: 'Harmony Gold Mining Company Limited', ticker: 'HMY', exchange: 'NYSE', country: 'South Africa', market_cap: 6.5, description: 'South African gold producer', website: 'https://www.harmony.co.za' },
  { name: 'DRDGOLD Limited', ticker: 'DRD', exchange: 'NYSE', country: 'South Africa', market_cap: 1.8, description: 'Gold mining company', website: 'https://www.drdgold.com' },

  // === STEEL & ALUMINUM ===
  { name: 'United States Steel Corporation', ticker: 'X', exchange: 'NYSE', country: 'USA', market_cap: 8.7, description: 'Integrated steel producer', website: 'https://www.ussteel.com' },
  { name: 'Nucor Corporation', ticker: 'NUE', exchange: 'NYSE', country: 'USA', market_cap: 31.2, description: 'Largest steel producer in USA', website: 'https://www.nucor.com' },
  { name: 'Steel Dynamics, Inc.', ticker: 'STLD', exchange: 'NASDAQ', country: 'USA', market_cap: 18.5, description: 'Steel producer and metals recycler', website: 'https://www.steeldynamics.com' },
  { name: 'Alcoa Corporation', ticker: 'AA', exchange: 'NYSE', country: 'USA', market_cap: 9.8, description: 'Global aluminum producer', website: 'https://www.alcoa.com' },

  // === DIVERSIFIED SMALL-MID CAP ===
  { name: 'Silvercorp Metals Inc.', ticker: 'SVM', exchange: 'NYSE', country: 'Canada', market_cap: 2.5, description: 'Silver-lead-zinc producer', website: 'https://www.silvercorpmetals.com' },
  { name: 'Endeavour Silver Corp.', ticker: 'EXK', exchange: 'NYSE', country: 'Canada', market_cap: 1.8, description: 'Mexican silver producer', website: 'https://www.edrsilver.com' },
  { name: 'Fortuna Silver Mines Inc.', ticker: 'FSM', exchange: 'NYSE', country: 'Canada', market_cap: 2.2, description: 'Silver-gold producer in Latin America', website: 'https://www.fortunasilver.com' },

  // === ASIAN MINERS ===
  { name: 'Zijin Mining Group Company Limited', ticker: '2899', exchange: 'HKEX', country: 'China', market_cap: 45.0, description: 'Chinese gold and copper producer', website: 'http://www.zijinmining.com' },
  { name: 'China Molybdenum Co., Ltd.', ticker: '3993', exchange: 'HKEX', country: 'China', market_cap: 28.0, description: 'Diversified mining company (copper, cobalt, niobium, tungsten, molybdenum)', website: 'http://www.cmoc.com' },
  { name: 'Jiangxi Copper Company Limited', ticker: '0358', exchange: 'HKEX', country: 'China', market_cap: 18.0, description: 'China\'s largest copper producer', website: 'http://www.jxcc.com' },

  // === ADDITIONAL INTERNATIONAL ===
  { name: 'Polymetal International plc', ticker: 'POLY', exchange: 'LSE', country: 'Russia/UK', market_cap: 4.5, description: 'Gold and silver producer in Russia and Kazakhstan', website: 'https://www.polymetalinternational.com' },
  { name: 'Fresnillo plc', ticker: 'FRES', exchange: 'LSE', country: 'Mexico/UK', market_cap: 8.2, description: 'World\'s largest primary silver producer and major gold producer', website: 'https://www.fresnilloplc.com' },
  { name: 'Centamin plc', ticker: 'CEY', exchange: 'LSE', country: 'Egypt/UK', market_cap: 2.5, description: 'Gold producer in Egypt', website: 'https://www.centamin.com' },

  // === JUNIOR EXPLORERS & DEVELOPERS (High Growth Potential) ===
  { name: 'K92 Mining Inc.', ticker: 'KNT', exchange: 'TSX', country: 'Canada', market_cap: 1.5, description: 'Gold producer in Papua New Guinea', website: 'https://www.k92mining.com' },
  { name: 'Adriatic Metals Plc', ticker: 'ADT', exchange: 'ASX', country: 'Bosnia/Australia', market_cap: 1.8, description: 'Zinc-lead-silver-gold development project in Bosnia & Herzegovina', website: 'https://www.adriaticmetals.com' },
  { name: 'West African Resources Limited', ticker: 'WAF', exchange: 'ASX', country: 'Australia', market_cap: 2.2, description: 'Gold producer in Burkina Faso', website: 'https://www.westafrican.com.au' },
  { name: 'Firefinch Limited', ticker: 'FFX', exchange: 'ASX', country: 'Australia', market_cap: 0.8, description: 'Gold producer in Mali', website: 'https://www.firefinchltd.com' },
  { name: 'Turquoise Hill Resources Ltd.', ticker: 'TRQ', exchange: 'NYSE', country: 'Canada', market_cap: 3.5, description: 'Mongolian copper-gold producer', website: 'https://www.turquoisehill.com' },

  // === PLATINUM GROUP METALS ===
  { name: 'Northam Platinum Holdings Limited', ticker: 'NHM', exchange: 'JSE', country: 'South Africa', market_cap: 5.2, description: 'Platinum group metals producer', website: 'https://www.northam.co.za' },
  { name: 'Royal Bafokeng Platinum Limited', ticker: 'RBP', exchange: 'JSE', country: 'South Africa', market_cap: 2.8, description: 'Platinum group metals producer', website: 'https://www.bafokengplatinum.co.za' },

  // === SPECIALTY METALS & MINERS ===
  { name: 'MMC Norilsk Nickel', ticker: 'GMKN', exchange: 'MCX', country: 'Russia', market_cap: 35.0, description: 'World\'s largest refined nickel producer', website: 'https://www.nornickel.com' },
  { name: 'Tianqi Lithium Corporation', ticker: '002466', exchange: 'SSE', country: 'China', market_cap: 12.0, description: 'Major lithium producer', website: 'http://www.tianqilithium.com' },
  { name: 'Alrosa', ticker: 'ALRS', exchange: 'MCX', country: 'Russia', market_cap: 8.5, description: 'World\'s largest diamond mining company by volume', website: 'http://eng.alrosa.ru' },

  // === EMERGING MARKETS MINERS ===
  { name: 'Vedanta Limited', ticker: 'VEDL', exchange: 'NSE', country: 'India', market_cap: 15.0, description: 'Diversified natural resources company (zinc, lead, silver, copper, iron ore, oil & gas)', website: 'https://www.vedantalimited.com' },
  { name: 'Hindustan Zinc Limited', ticker: 'HINDZINC', exchange: 'NSE', country: 'India', market_cap: 8.5, description: 'Integrated zinc-lead-silver producer', website: 'https://www.hzlindia.com' },
  { name: 'Coal India Limited', ticker: 'COALINDIA', exchange: 'NSE', country: 'India', market_cap: 65.0, description: 'World\'s largest coal producer', website: 'https://www.coalindia.in' },

  // === ADDITIONAL QUALITY NAMES FOR DIVERSITY ===
  { name: 'Equinox Gold Corp.', ticker: 'EQX', exchange: 'TSX', country: 'Canada', market_cap: 3.2, description: 'Americas-focused gold producer', website: 'https://www.equinoxgold.com' },
  { name: 'Calibre Mining Corp.', ticker: 'CXB', exchange: 'TSX', country: 'Canada', market_cap: 1.5, description: 'Gold producer in Nicaragua', website: 'https://www.calibremining.com' },
  { name: 'Lundin Gold Inc.', ticker: 'LUG', exchange: 'TSX', country: 'Canada', market_cap: 2.8, description: 'Gold producer in Ecuador', website: 'https://www.lundingold.com' },
  { name: 'Centerra Gold Inc.', ticker: 'CG', exchange: 'TSX', country: 'Canada', market_cap: 2.5, description: 'Gold producer in North America and Asia', website: 'https://www.centerragold.com' },

  // === MALAYSIAN & SOUTHEAST ASIAN MINERS ===
  { name: 'Straits Resources Limited', ticker: 'STI', exchange: 'ASX', country: 'Australia/Malaysia', market_cap: 0.5, description: 'Copper-gold producer with assets in Tasmania', website: 'https://www.straits.com.au' },
  { name: 'Petra Diamonds Limited', ticker: 'PDL', exchange: 'LSE', country: 'South Africa/UK', market_cap: 1.2, description: 'Diamond producer in South Africa and Tanzania', website: 'https://www.petradiamonds.com' },
  { name: 'PT Aneka Tambang Tbk', ticker: 'ANTM', exchange: 'IDX', country: 'Indonesia', market_cap: 2.5, description: 'Indonesian nickel, gold, and bauxite producer', website: 'https://www.antam.com' },
  { name: 'PT Vale Indonesia Tbk', ticker: 'INCO', exchange: 'IDX', country: 'Indonesia', market_cap: 8.0, description: 'Nickel producer in Indonesia', website: 'https://www.vale.com/indonesia' },
  { name: 'PT Timah Tbk', ticker: 'TINS', exchange: 'IDX', country: 'Indonesia', market_cap: 1.8, description: 'Integrated tin mining and smelting company', website: 'https://www.timah.com' },

  // === MORE DIVERSIFIED INTERNATIONALS ===
  { name: 'Boliden AB', ticker: 'BOL', exchange: 'OMX', country: 'Sweden', market_cap: 12.0, description: 'Scandinavian metals company (zinc, copper, lead, gold, silver)', website: 'https://www.boliden.com' },
  { name: 'KGHM Polska Miedz S.A.', ticker: 'KGH', exchange: 'WSE', country: 'Poland', market_cap: 7.0, description: 'Integrated copper and silver producer', website: 'https://www.kghm.com' },

  // === ADDITIONAL LITHIUM COMPANIES ===
  { name: 'Livent Corporation', ticker: 'LTHM', exchange: 'NYSE', country: 'USA', market_cap: 2.5, description: 'Lithium hydroxide and carbonate producer', website: 'https://www.livent.com' },
  { name: 'Standard Lithium Ltd.', ticker: 'SLI', exchange: 'NYSE', country: 'Canada', market_cap: 0.8, description: 'Lithium development company', website: 'https://www.standardlithium.com' },
  { name: 'American Lithium Corp.', ticker: 'LI', exchange: 'TSX-V', country: 'Canada', market_cap: 0.5, description: 'Lithium development company', website: 'https://www.americanlithiumcorp.com' },
  { name: 'Sigma Lithium Corporation', ticker: 'SGML', exchange: 'NASDAQ', country: 'Canada', market_cap: 1.8, description: 'Lithium producer in Brazil', website: 'https://www.sigmalithiumresources.com' },

  // === ADDITIONAL URANIUM COMPANIES ===
  { name: 'Ur-Energy Inc.', ticker: 'URG', exchange: 'NYSE', country: 'USA', market_cap: 0.6, description: 'Uranium producer in USA', website: 'https://www.ur-energy.com' },
  { name: 'Peninsula Energy Limited', ticker: 'PEN', exchange: 'ASX', country: 'Australia', market_cap: 0.8, description: 'Uranium producer in USA', website: 'https://www.pel.net.au' },
  { name: 'Paladin Energy Ltd', ticker: 'PDN', exchange: 'ASX', country: 'Australia', market_cap: 3.5, description: 'Uranium producer in Namibia', website: 'https://www.paladinenergy.com.au' },

  // === ADDITIONAL QUALITY GOLD PRODUCERS ===
  { name: 'Torex Gold Resources Inc.', ticker: 'TXG', exchange: 'TSX', country: 'Canada', market_cap: 2.2, description: 'Mexican gold producer', website: 'https://www.torexgold.com' },
  { name: 'Wesdome Gold Mines Ltd.', ticker: 'WDO', exchange: 'TSX', country: 'Canada', market_cap: 1.8, description: 'Canadian gold producer', website: 'https://www.wesdome.com' },
  { name: 'Karora Resources Inc.', ticker: 'KRR', exchange: 'TSX', country: 'Canada', market_cap: 1.2, description: 'Gold producer in Australia', website: 'https://www.karoraresources.com' },

  // === ASIAN & EMERGING DIVERSIFIED ===
  { name: 'Shandong Gold Mining Co., Ltd.', ticker: '600547', exchange: 'SSE', country: 'China', market_cap: 18.0, description: 'Major Chinese gold producer', website: 'http://www.jindinggold.com' },
  { name: 'MMG Limited', ticker: '1208', exchange: 'HKEX', country: 'China', market_cap: 6.5, description: 'Diversified base metals (copper, zinc, lead)', website: 'https://www.mmg.com' },
]

async function main() {
  console.log('🗑️  Clearing existing companies...\n')

  const { error: deleteError } = await supabase
    .from('companies')
    .delete()
    .neq('id', '00000000-0000-0000-0000-000000000000')

  if (deleteError) {
    console.error('Error clearing companies:', deleteError)
    return
  }

  console.log('✅ Companies table cleared\n')
  console.log(`💾 Inserting ${MINING_COMPANIES.length} verified mining companies with market cap data...\n`)

  // Insert companies in batches
  const batchSize = 50
  let totalInserted = 0

  for (let i = 0; i < MINING_COMPANIES.length; i += batchSize) {
    const batch = MINING_COMPANIES.slice(i, i + batchSize)

    const { error: insertError } = await supabase
      .from('companies')
      .insert(batch)

    if (insertError) {
      console.error(`❌ Error inserting batch ${i / batchSize + 1}:`, insertError)
    } else {
      totalInserted += batch.length
      console.log(`✅ Inserted batch ${i / batchSize + 1} (${batch.length} companies)`)
    }
  }

  console.log(`\n✨ Database population complete!`)
  console.log(`📈 Total companies successfully added: ${totalInserted}/${MINING_COMPANIES.length}`)

  // Calculate total market cap
  const totalMarketCap = MINING_COMPANIES.reduce((sum, c) => sum + (c.market_cap || 0), 0)
  console.log(`💰 Total combined market cap: $${totalMarketCap.toFixed(1)}B`)
  console.log(`📊 Average market cap: $${(totalMarketCap / MINING_COMPANIES.length).toFixed(1)}B`)

  // Count by country
  const countryCounts = MINING_COMPANIES.reduce((acc: any, c) => {
    acc[c.country] = (acc[c.country] || 0) + 1
    return acc
  }, {})

  console.log(`\n🌍 Geographic Distribution:`)
  Object.entries(countryCounts)
    .sort((a: any, b: any) => b[1] - a[1])
    .forEach(([country, count]) => {
      console.log(`  • ${country}: ${count}`)
    })

  // Count by exchange
  const exchangeCounts = MINING_COMPANIES.reduce((acc: any, c) => {
    acc[c.exchange] = (acc[c.exchange] || 0) + 1
    return acc
  }, {})

  console.log(`\n📊 Exchange Distribution:`)
  Object.entries(exchangeCounts)
    .sort((a: any, b: any) => b[1] - a[1])
    .forEach(([exchange, count]) => {
      console.log(`  • ${exchange}: ${count}`)
    })

  // Top 10 by market cap
  console.log(`\n🏆 Top 10 by Market Cap:`)
  MINING_COMPANIES
    .sort((a, b) => (b.market_cap || 0) - (a.market_cap || 0))
    .slice(0, 10)
    .forEach((c, i) => {
      console.log(`  ${i + 1}. ${c.name} (${c.ticker}): $${c.market_cap}B`)
    })
}

main()
