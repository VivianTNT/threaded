import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

// Top 100 high-quality mining companies with complete data
const top100MiningCompanies = [
  // Tier 1: Over $50B Market Cap
  { name: "BHP Group", ticker: "BHP", exchange: "ASX/NYSE", country: "Australia", website: "https://www.bhp.com", description: "World's largest mining company, diversified across iron ore, copper, coal, and petroleum.", market_cap: 150000 },
  { name: "Rio Tinto", ticker: "RIO", exchange: "ASX/LSE/NYSE", country: "United Kingdom", website: "https://www.riotinto.com", description: "Global mining group focused on finding, mining and processing mineral resources including aluminum, copper, diamonds, iron ore and uranium.", market_cap: 99830 },
  { name: "Southern Copper Corporation", ticker: "SCCO", exchange: "NYSE", country: "United States", website: "https://www.southerncopper.com", description: "Leading integrated copper producer with operations in Peru and Mexico.", market_cap: 105000 },
  { name: "Zijin Mining Group", ticker: "2899", exchange: "HKG", country: "China", website: "https://www.zijinmining.com", description: "Major Chinese mining company with focus on gold, copper, and other metals.", market_cap: 102000 },
  { name: "Newmont Corporation", ticker: "NEM", exchange: "NYSE/TSX", country: "United States", website: "https://www.newmont.com", description: "World's leading gold company with significant copper, silver, zinc and lead assets.", market_cap: 100500 },

  // Tier 2: $20B - $50B Market Cap
  { name: "Agnico Eagle Mines", ticker: "AEM", exchange: "NYSE/TSX", country: "Canada", website: "https://www.agnicoeagle.com", description: "Senior Canadian gold mining company with operations in Canada, Australia, Finland and Mexico.", market_cap: 89000 },
  { name: "Freeport-McMoRan", ticker: "FCX", exchange: "NYSE", country: "United States", website: "https://www.fcx.com", description: "Leading international mining company with large, long-lived, geographically diverse assets in copper, gold and molybdenum.", market_cap: 75000 },
  { name: "Glencore", ticker: "GLEN", exchange: "LSE", country: "Switzerland", website: "https://www.glencore.com", description: "Diversified natural resource company with focus on metals and minerals, energy products and agricultural products.", market_cap: 53900 },
  { name: "Ma'aden", ticker: "1211", exchange: "TADAWUL", country: "Saudi Arabia", website: "https://www.maaden.com.sa", description: "Saudi Arabian Mining Company, focused on gold, phosphate, aluminum and industrial minerals.", market_cap: 49300 },
  { name: "Barrick Gold Corporation", ticker: "GOLD", exchange: "NYSE/TSX", country: "Canada", website: "https://www.barrick.com", description: "One of the largest gold mining companies with copper assets and operations across five continents.", market_cap: 47000 },

  // Tier 3: $10B - $20B Market Cap
  { name: "Franco-Nevada Corporation", ticker: "FNV", exchange: "NYSE/TSX", country: "Canada", website: "https://www.franco-nevada.com", description: "Leading gold-focused royalty and streaming company with diversified portfolio of cash-flow producing assets.", market_cap: 28500 },
  { name: "Wheaton Precious Metals", ticker: "WPM", exchange: "NYSE/TSX", country: "Canada", website: "https://www.wheatonpm.com", description: "World's premier precious metals streaming company with agreements for purchase of gold, silver, palladium and cobalt.", market_cap: 24800 },
  { name: "Nutrien", ticker: "NTR", exchange: "NYSE/TSX", country: "Canada", website: "https://www.nutrien.com", description: "World's largest provider of crop inputs and services with potash, nitrogen and phosphate operations.", market_cap: 23500 },
  { name: "Teck Resources", ticker: "TECK", exchange: "NYSE/TSX", country: "Canada", website: "https://www.teck.com", description: "Diversified resource company committed to responsible mining and mineral development with major business units in copper, zinc and steelmaking coal.", market_cap: 18900 },
  { name: "Cameco Corporation", ticker: "CCJ", exchange: "NYSE/TSX", country: "Canada", website: "https://www.cameco.com", description: "One of the world's largest publicly traded uranium producers accountable for about 15% of global production.", market_cap: 17800 },
  { name: "Anglo American", ticker: "AAL", exchange: "LSE", country: "United Kingdom", website: "https://www.angloamerican.com", description: "Global diversified mining business with focus on diamonds (De Beers), copper, platinum, iron ore and coal.", market_cap: 16500 },
  { name: "First Quantum Minerals", ticker: "FM", exchange: "TSX", country: "Canada", website: "https://www.first-quantum.com", description: "Mining and metals company with operations across Africa, Australia, South America and Europe focused on copper, nickel, gold and zinc.", market_cap: 16200 },
  { name: "Vale S.A.", ticker: "VALE", exchange: "NYSE", country: "Brazil", website: "https://www.vale.com", description: "Brazilian multinational corporation engaged in metals and mining and one of largest logistics operators in Brazil.", market_cap: 51000 },
  { name: "Fortescue Metals Group", ticker: "FMG", exchange: "ASX", country: "Australia", website: "https://www.fortescue.com", description: "Iron ore company with mining, rail and port operations in Western Australia's Pilbara region.", market_cap: 45000 },
  { name: "Polyus", ticker: "PLZL", exchange: "LSE/MOEX", country: "Russia", website: "https://www.polyus.com", description: "Largest gold producer in Russia with significant reserves and resources.", market_cap: 14500 },

  // Tier 4: $5B - $10B Market Cap
  { name: "Lundin Mining", ticker: "LUN", exchange: "TSX", country: "Canada", website: "https://www.lundinmining.com", description: "Diversified Canadian base metals mining company with operations in Brazil, Chile, Portugal, Sweden and the United States.", market_cap: 9800 },
  { name: "Kinross Gold", ticker: "KGC", exchange: "NYSE/TSX", country: "Canada", website: "https://www.kinross.com", description: "Senior gold mining company with mines and projects across Americas, West Africa and Russia.", market_cap: 9500 },
  { name: "Alamos Gold", ticker: "AGI", exchange: "NYSE/TSX", country: "Canada", website: "https://www.alamosgold.com", description: "Canadian gold producer with diversified portfolio of production and development stage assets in North America.", market_cap: 8900 },
  { name: "Pan American Silver", ticker: "PAAS", exchange: "NYSE/TSX", country: "Canada", website: "https://www.panamericansilver.com", description: "Silver and gold producer with operations in Mexico, Peru, Canada, Argentina and Bolivia.", market_cap: 8400 },
  { name: "Endeavour Mining", ticker: "EDV", exchange: "TSX/LSE", country: "United Kingdom", website: "https://www.endeavourmining.com", description: "Senior gold producer focused on West Africa with portfolio of mines across Senegal, Burkina Faso and Ivory Coast.", market_cap: 7900 },
  { name: "SSR Mining", ticker: "SSRM", exchange: "NASDAQ/TSX/ASX", country: "Canada", website: "https://www.ssrmining.com", description: "Intermediate gold company with assets located in USA, Turkey, Canada and Argentina.", market_cap: 7600 },
  { name: "B2Gold Corp", ticker: "BTG", exchange: "NYSE/TSX", country: "Canada", website: "https://www.b2gold.com", description: "International senior gold producer with three operating mines in Mali, Namibia and the Philippines.", market_cap: 7200 },
  { name: "Osisko Gold Royalties", ticker: "OR", exchange: "TSX/NYSE", country: "Canada", website: "https://www.osiskogr.com", description: "Intermediate precious metal royalty company focused on precious metals in Americas.", market_cap: 6800 },
  { name: "Sandstorm Gold", ticker: "SAND", exchange: "NYSE/TSX", country: "Canada", website: "https://www.sandstormgold.com", description: "Gold royalty company providing upfront financing to gold mining companies.", market_cap: 6500 },
  { name: "Ivanhoe Mines", ticker: "IVN", exchange: "TSX", country: "Canada", website: "https://www.ivanhoemines.com", description: "Mining company focused on advancing its three principal projects in Southern Africa: Kamoa-Kakula, Platreef and Kipushi.", market_cap: 13500 },

  // Tier 5: $2B - $5B Market Cap
  { name: "Capstone Copper", ticker: "CS", exchange: "TSX/ASX", country: "Canada", website: "https://www.capstoneco pper.com", description: "Copper mining company with operations in the Americas focused on Pinto Valley (Arizona), Cozamin (Mexico) and Mantoverde (Chile).", market_cap: 4800 },
  { name: "South32", ticker: "S32", exchange: "ASX/LSE", country: "Australia", website: "https://www.south32.net", description: "Diversified metals and mining company with quality operations in Australia, Southern Africa and South America.", market_cap: 11500 },
  { name: "Evolution Mining", ticker: "EVN", exchange: "ASX", country: "Australia", website: "https://www.evolutionmining.com.au", description: "Australian gold miner operating six wholly-owned mines across Australia and Canada.", market_cap: 6200 },
  { name: "Northern Star Resources", ticker: "NST", exchange: "ASX", country: "Australia", website: "https://www.nsrltd.com", description: "Australian gold producer with operations in Western Australia and Alaska.", market_cap: 9400 },
  { name: "OZ Minerals", ticker: "OZL", exchange: "ASX", country: "Australia", website: "https://www.ozminerals.com", description: "Modern mining company focused on copper production with operations in South Australia.", market_cap: 4200 },
  { name: "IGO Limited", ticker: "IGO", exchange: "ASX", country: "Australia", website: "https://www.igo.com.au", description: "Diversified mining company producing nickel, copper and cobalt with focus on battery materials.", market_cap: 4500 },
  { name: "Sandfire Resources", ticker: "SFR", exchange: "ASX", country: "Australia", website: "https://www.sandfire.com.au", description: "Mining and exploration company focused on copper production from operations in Australia and Botswana.", market_cap: 3200 },
  { name: "Centerra Gold", ticker: "CG", exchange: "TSX", country: "Canada", website: "https://www.centerragold.com", description: "Canadian-based gold mining company with operations in Canada, Turkey, and North America.", market_cap: 2800 },
  { name: "Eldorado Gold", ticker: "ELD", exchange: "NYSE/TSX", country: "Canada", website: "https://www.eldoradogold.com", description: "Canadian mid-tier gold producer with mining, development and exploration operations in Turkey, Canada and Greece.", market_cap: 3100 },
  { name: "IAMGOLD Corporation", ticker: "IAG", exchange: "NYSE/TSX", country: "Canada", website: "https://www.iamgold.com", description: "Mid-tier gold mining company with operations in North and South America and West Africa.", market_cap: 2400 },
  { name: "Hudbay Minerals", ticker: "HBM", exchange: "NYSE/TSX", country: "Canada", website: "https://www.hudbay.com", description: "Diversified mining company producing copper, gold and zinc concentrate with operations in Canada, Peru and the United States.", market_cap: 3400 },
  { name: "Lundin Gold", ticker: "LUG", exchange: "TSX", country: "Canada", website: "https://www.lundingold.com", description: "Canadian-based gold producer focused on its Fruta del Norte mine in Ecuador.", market_cap: 3800 },
  { name: "Torex Gold Resources", ticker: "TXG", exchange: "TSX", country: "Canada", website: "https://www.torexgold.com", description: "Intermediate gold producer based in Canada with operations in Morelos State, Mexico.", market_cap: 2200 },
  { name: "New Gold", ticker: "NGD", exchange: "NYSE/TSX", country: "Canada", website: "https://www.newgold.com", description: "Intermediate gold and copper producer with portfolio of two operating mines in Canada and the United States.", market_cap: 2100 },
  { name: "Wesdome Gold Mines", ticker: "WDO", exchange: "TSX", country: "Canada", website: "https://www.wesdome.com", description: "Canadian-focused gold producer with operations in Ontario and Quebec.", market_cap: 2300 },
  { name: "Seabridge Gold", ticker: "SA", exchange: "NYSE/TSX", country: "Canada", website: "https://www.seabridgegold.com", description: "Development stage company primarily focused on KSM and Courageous Lake projects in North America.", market_cap: 2600 },
  { name: "Equinox Gold", ticker: "EQX", exchange: "NYSE/TSX", country: "Canada", website: "https://www.equinoxgold.com", description: "Growth-focused gold producer with operations and projects in the United States, Brazil and Mexico.", market_cap: 3300 },
  { name: "K92 Mining", ticker: "KNT", exchange: "TSX", country: "Canada", website: "https://www.k92mining.com", description: "Gold and copper producer with Kainantu Gold Mine in Papua New Guinea.", market_cap: 2500 },

  // Tier 6: $1B - $2B Market Cap
  { name: "Hecla Mining", ticker: "HL", exchange: "NYSE", country: "United States", website: "https://www.hecla.com", description: "Leading low-cost US silver producer with complementary gold, lead and zinc production.", market_cap: 4100 },
  { name: "Coeur Mining", ticker: "CDE", exchange: "NYSE", country: "United States", website: "https://www.coeur.com", description: "US-based precious metals producer with mines in North America and portfolio of development projects.", market_cap: 1900 },
  { name: "Novagold Resources", ticker: "NG", exchange: "NYSE/TSX", country: "Canada", website: "https://www.novagold.com", description: "Exploration and development company focused on Donlin Gold project in Alaska.", market_cap: 1600 },
  { name: "Oceana Gold", ticker: "OGC", exchange: "TSX/ASX", country: "Australia", website: "https://www.oceanagold.com", description: "Gold producer with operations in the Philippines, New Zealand and the United States.", market_cap: 1800 },
  { name: "Pretium Resources", ticker: "PVG", exchange: "NYSE/TSX", country: "Canada", website: "https://www.pretivm.com", description: "Precious metals mining company producing from Brucejack gold mine in British Columbia.", market_cap: 1700 },
  { name: "Yamana Gold", ticker: "AUY", exchange: "NYSE/TSX", country: "Canada", website: "https://www.yamana.com", description: "Canadian-based precious metals producer with focus on gold and silver production primarily in Latin America.", market_cap: 6700 },
  { name: "Osisko Mining", ticker: "OSK", exchange: "TSX", country: "Canada", website: "https://www.osiskomining.com", description: "Mineral exploration company focused on Windfall gold project in Quebec.", market_cap: 1400 },
  { name: "Artemis Gold", ticker: "ARTG", exchange: "TSX", country: "Canada", website: "https://www.artemisgoldinc.com", description: "Gold developer focused on advancing Blackwater Gold Project in British Columbia.", market_cap: 1200 },
  { name: "Tudor Gold", ticker: "TUD", exchange: "TSXV", country: "Canada", website: "https://www.tudorgold.com", description: "Precious and base metals explorer focused on Treaty Creek project in British Columbia's Golden Triangle.", market_cap: 1100 },
  { name: "Skeena Resources", ticker: "SKE", exchange: "TSX/NYSE", country: "Canada", website: "https://www.skeenaresources.com", description: "Canadian mining company focused on revitalizing past-producing Eskay Creek gold-silver mine in British Columbia.", market_cap: 1300 },
  { name: "Rupert Resources", ticker: "RUP", exchange: "TSXV", country: "Canada", website: "https://www.rupertresources.com", description: "Gold explorer focused on advancing Ikkari discovery in Northern Finland.", market_cap: 1000 },
  { name: "Great Bear Resources", ticker: "GBR", exchange: "TSXV", country: "Canada", website: "https://www.greatbearresources.ca", description: "Gold exploration company focused on Dixie project in Red Lake, Ontario.", market_cap: 1500 },
  { name: "Karora Resources", ticker: "KRR", exchange: "TSX", country: "Canada", website: "https://www.karoraresources.com", description: "Mid-tier nickel and gold producer with operations in Western Australia.", market_cap: 1100 },
  { name: "Galiano Gold", ticker: "GAU", exchange: "NYSE/TSX", country: "Canada", website: "https://www.galianogold.com", description: "Canadian mining company focused on Asanko Gold Mine in Ghana.", market_cap: 900 },
  { name: "Cardinal Resources", ticker: "CDV", exchange: "ASX/TSX", country: "Australia", website: "https://www.cardinalresources.com.au", description: "Gold exploration and development company focused on Namdini Gold Project in Ghana.", market_cap: 800 },

  // Tier 7: $500M - $1B Market Cap
  { name: "Polymetal International", ticker: "POLY", exchange: "LSE/MOEX", country: "Russia", website: "https://www.polymetalinternational.com", description: "Precious metals mining company with assets in Russia and Kazakhstan.", market_cap: 3500 },
  { name: "Hochschild Mining", ticker: "HOC", exchange: "LSE", country: "Peru", website: "https://www.hochschildmining.com", description: "Precious metals company with focus on exploration, mining and processing of gold and silver in the Americas.", market_cap: 900 },
  { name: "Fresnillo", ticker: "FRES", exchange: "LSE", country: "Mexico", website: "https://www.fresnilloplc.com", description: "World's largest primary silver producer with operations in Mexico.", market_cap: 8200 },
  { name: "Buenaventura", ticker: "BVN", exchange: "NYSE", country: "Peru", website: "https://www.buenaventura.com", description: "Peru's largest publicly-traded precious metals company focused on gold and silver.", market_cap: 3200 },
  { name: "Gold Fields", ticker: "GFI", exchange: "NYSE/JSE", country: "South Africa", website: "https://www.goldfields.com", description: "One of world's largest gold producers with operations in South Africa, Ghana, Australia and Peru.", market_cap: 11500 },
  { name: "AngloGold Ashanti", ticker: "AU", exchange: "NYSE/JSE", country: "South Africa", website: "https://www.anglogoldashanti.com", description: "Global gold producer with operations on four continents and 14 operations in nine countries.", market_cap: 8900 },
  { name: "Sibanye Stillwater", ticker: "SBSW", exchange: "NYSE/JSE", country: "South Africa", website: "https://www.sibanyestillwater.com", description: "Multinational mining and metals processing group with diverse portfolio of PGM, gold and lithium operations.", market_cap: 7800 },
  { name: "Harmony Gold", ticker: "HMY", exchange: "NYSE/JSE", country: "South Africa", website: "https://www.harmony.co.za", description: "South African gold mining company with operations limited to South Africa and Papua New Guinea.", market_cap: 4500 },
  { name: "Pan African Resources", ticker: "PAF", exchange: "JSE/LSE", country: "South Africa", website: "https://www.panafricanresources.com", description: "Mid-tier gold producer with operations in South Africa.", market_cap: 700 },
  { name: "DRD Gold", ticker: "DRD", exchange: "NYSE/JSE", country: "South Africa", website: "https://www.drdgold.com", description: "South African gold producer focused on surface gold tailings retreatment operations.", market_cap: 600 },
  { name: "Impala Platinum", ticker: "IMP", exchange: "JSE", country: "South Africa", website: "https://www.implats.co.za", description: "One of world's foremost producers of platinum with mining, refining and marketing operations.", market_cap: 6200 },
  { name: "Northam Platinum", ticker: "NPH", exchange: "JSE", country: "South Africa", website: "https://www.northam.co.za", description: "One of world's pre-eminent platinum producers with operations focused on South Africa.", market_cap: 4100 },
  { name: "Royal Bafokeng Platinum", ticker: "RBP", exchange: "JSE", country: "South Africa", website: "https://www.bafokengplatinum.co.za", description: "South African platinum group metals producer on Western Limb of Bushveld Complex.", market_cap: 800 },
  { name: "Amplats (Anglo American Platinum)", ticker: "AMS", exchange: "JSE", country: "South Africa", website: "https://www.angloamericanplatinum.com", description: "World's leading primary producer of platinum group metals.", market_cap: 15000 },

  // Tier 8: Additional Quality Companies
  { name: "Lundin Mining", ticker: "LUN", exchange: "TSX", country: "Canada", website: "https://www.lundinmining.com", description: "Diversified base metals mining company with operations in Chile, United States, Portugal, Sweden and Brazil.", market_cap: 9800 },
  { name: "Taseko Mines", ticker: "TKO", exchange: "TSX/NYSE", country: "Canada", website: "https://www.tasekomines.com", description: "Canadian mining company focused on Gibraltar copper-molybdenum mine in British Columbia.", market_cap: 600 },
  { name: "Turquoise Hill Resources", ticker: "TRQ", exchange: "NYSE/TSX", country: "Canada", website: "https://www.turquoisehillresources.com", description: "Mining company focused on operation and continued development of Oyu Tolgoi copper-gold mine in Mongolia.", market_cap: 5400 },
  { name: "Trilogy Metals", ticker: "TMQ", exchange: "NYSE/TSX", country: "Canada", website: "https://www.trilogymetals.com", description: "Metals exploration and development company focused on copper, zinc, lead, gold and silver projects in Alaska.", market_cap: 500 },
  { name: "Arizona Metals", ticker: "AMC", exchange: "TSX", country: "Canada", website: "https://www.arizonametals.com", description: "Exploration company focused on copper and zinc projects in Arizona.", market_cap: 400 },
  { name: "Copper Mountain Mining", ticker: "CMMC", exchange: "TSX", country: "Canada", website: "https://www.cumtn.com", description: "Canadian-based copper producer focused on Copper Mountain mine in British Columbia.", market_cap: 900 },
  { name: "Huayou Cobalt", ticker: "603799", exchange: "SSE", country: "China", website: "https://www.huayou.com", description: "Leading integrated cobalt and nickel products supplier with operations across mining, smelting and precursor materials.", market_cap: 7200 },
  { name: "China Molybdenum", ticker: "3993", exchange: "HKG/SSE", country: "China", website: "https://www.cmoc.com", description: "Mining company focused on copper, molybdenum, tungsten, niobium, phosphates and cobalt.", market_cap: 12000 },
  { name: "Chinalco", ticker: "2600", exchange: "HKG/SSE", country: "China", website: "https://www.chalco.com.cn", description: "Aluminum Corporation of China, largest alumina and primary aluminum producer in China.", market_cap: 6500 },
  { name: "Zhejiang Huayou Cobalt", ticker: "603799", exchange: "SSE", country: "China", website: "https://www.huayou.com", description: "Leading global cobalt chemical producer with vertical integration from mining to refined products.", market_cap: 7200 },
  { name: "Shandong Gold", ticker: "600547", exchange: "SSE", country: "China", website: "https://www.sdgold.com.cn", description: "One of China's largest gold producers with operations in China and internationally.", market_cap: 5800 },
  { name: "Silvercorp Metals", ticker: "SVM", exchange: "NYSE/TSX", country: "Canada", website: "https://www.silvercorp.ca", description: "Canadian mining company producing silver, lead and zinc with operations in China.", market_cap: 1100 },
  { name: "MAG Silver", ticker: "MAG", exchange: "NYSE/TSX", country: "Canada", website: "https://www.magsilver.com", description: "Canadian development and exploration company focused on Juanicipio silver-gold joint venture project in Mexico.", market_cap: 1600 },
  { name: "Aya Gold & Silver", ticker: "AYA", exchange: "TSX", country: "Canada", website: "https://www.ayagoldsilver.com", description: "Moroccan-focused silver and gold producer with operations at Zgounder silver mine.", market_cap: 800 },
  { name: "Discovery Silver", ticker: "DSV", exchange: "TSX/NYSE", country: "Canada", website: "https://www.discoverysilver.com", description: "Silver developer focused on district-scale silver deposits in Mexico.", market_cap: 700 },
  { name: "Excellon Resources", ticker: "EXN", exchange: "TSX", country: "Canada", website: "https://www.excellonresources.com", description: "Silver mining company with Platosa mine in Mexico and development projects.", market_cap: 400 },
  { name: "First Majestic Silver", ticker: "AG", exchange: "NYSE/TSX", country: "Canada", website: "https://www.firstmajestic.com", description: "Pure silver producer with operations in Mexico focused on silver, gold, lead and zinc.", market_cap: 2100 },
  { name: "Hochschild Mining", ticker: "HOC", exchange: "LSE", country: "Peru", website: "https://www.hochschildmining.com", description: "Precious metals company focused on exploration, mining, and processing of gold and silver in the Americas.", market_cap: 900 },
  { name: "Centamin", ticker: "CEY", exchange: "LSE", country: "Egypt", website: "https://www.centamin.com", description: "Gold producer with operations in Egypt, Burkina Faso and Ivory Coast focused on Sukari gold mine.", market_cap: 1800 },
  { name: "Acacia Mining", ticker: "ACA", exchange: "LSE", country: "United Kingdom", website: "https://www.acaciamining.com", description: "Gold producer focused on Tanzania with operations at Bulyanhulu, Buzwagi and North Mara.", market_cap: 600 },
  { name: "Petra Diamonds", ticker: "PDL", exchange: "LSE", country: "United Kingdom", website: "https://www.petradiamonds.com", description: "Leading independent diamond mining group with operations in South Africa and Tanzania.", market_cap: 500 },
  { name: "Gem Diamonds", ticker: "GEMD", exchange: "LSE", country: "United Kingdom", website: "https://www.gemdiamonds.com", description: "Diamond producer focused on Letseng mine in Lesotho, known for large, high-quality diamonds.", market_cap: 400 },
  { name: "Lucara Diamond", ticker: "LUC", exchange: "TSX/NASDAQ", country: "Canada", website: "https://www.lucaradiamond.com", description: "Diamond mining company focused on Karowe mine in Botswana.", market_cap: 600 },
];

async function addTop100Companies() {
  console.log('Adding top 100 high-quality mining companies...\n');

  let successCount = 0;
  let errorCount = 0;

  for (const company of top100MiningCompanies) {
    try {
      const { data, error } = await supabase
        .from('companies')
        .insert({
          name: company.name,
          ticker: company.ticker,
          exchange: company.exchange,
          country: company.country,
          website: company.website,
          description: company.description,
          market_cap: company.market_cap,
          urls: [company.website],
          watchlist: false,
        });

      if (error) {
        console.error(`❌ Error adding ${company.name}:`, error.message);
        errorCount++;
      } else {
        console.log(`✅ Added: ${company.name} (${company.ticker}) - $${company.market_cap}M`);
        successCount++;
      }
    } catch (err) {
      console.error(`❌ Error adding ${company.name}:`, err);
      errorCount++;
    }
  }

  console.log(`\n✅ Successfully added: ${successCount} companies`);
  console.log(`❌ Errors: ${errorCount} companies`);
  console.log(`\nTotal companies in database now:`);

  const { count } = await supabase
    .from('companies')
    .select('*', { count: 'exact', head: true });

  console.log(`📊 Total: ${count} companies`);
}

addTop100Companies();
