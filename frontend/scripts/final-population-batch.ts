import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

// Final batch of projects to reach 400+
const MORE_PROJECTS = [
  // Add 150 more diverse global mining projects
  { company: "Zijin Mining", ticker: "2899", project: "Norton Gold Fields", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Zijin Mining", ticker: "2899", project: "Buriticá", location: "Colombia", commodity: "Gold", stage: "Operating" },
  { company: "Zijin Mining", ticker: "2899", project: "Timok", location: "Serbia", commodity: "Copper", stage: "Development" },
  { company: "Zijin Mining", ticker: "2899", project: "Kamoa Copper", location: "Democratic Republic of Congo", commodity: "Copper", stage: "Operating" },
  { company: "AngloGold Ashanti", ticker: "AU", project: "Mponeng", location: "South Africa", commodity: "Gold", stage: "Operating" },
  { company: "Harmony Gold", ticker: "HMY", project: "Moab Khotsong", location: "South Africa", commodity: "Gold", stage: "Operating" },
  { company: "Harmony Gold", ticker: "HMY", project: "Hidden Valley", location: "Papua New Guinea", commodity: "Gold", stage: "Operating" },
  { company: "Sibanye-Stillwater", ticker: "SBSW", project: "Driefontein", location: "South Africa", commodity: "Gold", stage: "Operating" },
  { company: "Sibanye-Stillwater", ticker: "SBSW", project: "Stillwater", location: "United States", commodity: "Platinum", stage: "Operating" },
  { company: "Sibanye-Stillwater", ticker: "SBSW", project: "Marikana", location: "South Africa", commodity: "Platinum", stage: "Operating" },
  { company: "Impala Platinum", ticker: "IMP", project: "Impala", location: "South Africa", commodity: "Platinum", stage: "Operating" },
  { company: "Impala Platinum", ticker: "IMP", project: "Zimplats", location: "Zimbabwe", commodity: "Platinum", stage: "Operating" },
  { company: "Impala Platinum", ticker: "IMP", project: "Marula", location: "South Africa", commodity: "Platinum", stage: "Operating" },
  { company: "Amplats", ticker: "AMS", project: "Mogalakwena", location: "South Africa", commodity: "Platinum", stage: "Operating" },
  { company: "Amplats", ticker: "AMS", project: "Amandelbult", location: "South Africa", commodity: "Platinum", stage: "Operating" },
  { company: "Amplats", ticker: "AMS", project: "Unki", location: "Zimbabwe", commodity: "Platinum", stage: "Operating" },
  { company: "Lonmin", ticker: "LMI", project: "Marikana", location: "South Africa", commodity: "Platinum", stage: "Operating" },
  { company: "Norilsk Nickel", ticker: "GMKN", project: "Norilsk", location: "Russia", commodity: "Nickel", stage: "Operating" },
  { company: "Norilsk Nickel", ticker: "GMKN", project: "Taimyr", location: "Russia", commodity: "Nickel", stage: "Operating" },
  { company: "Norilsk Nickel", ticker: "GMKN", project: "Kola", location: "Russia", commodity: "Nickel", stage: "Operating" },
  { company: "Polymet Mining", ticker: "PLM", project: "NorthMet", location: "United States", commodity: "Copper", stage: "Development" },
  { company: "Twin Metals", ticker: "TM", project: "Twin Metals", location: "United States", commodity: "Copper", stage: "Exploration" },
  { company: "Trilogy Metals", ticker: "TMQ", project: "Upper Kobuk", location: "United States", commodity: "Copper", stage: "Exploration" },
  { company: "Trilogy Metals", ticker: "TMQ", project: "Bornite", location: "United States", commodity: "Copper", stage: "Exploration" },
  { company: "NovaCopper", ticker: "NCQ", project: "Ambler", location: "United States", commodity: "Copper", stage: "Exploration" },
  { company: "Atalaya Mining", ticker: "ATYM", project: "Proyecto Riotinto", location: "Spain", commodity: "Copper", stage: "Operating" },
  { company: "Atalaya Mining", ticker: "ATYM", project: "Touro", location: "Spain", commodity: "Copper", stage: "Development" },
  { company: "Zinc", ticker: "Z", project: "Century", location: "Australia", commodity: "Zinc", stage: "Closed" },
  { company: "New Century Resources", ticker: "NCZ", project: "Century", location: "Australia", commodity: "Zinc", stage: "Operating" },
  { company: "Vedanta Zinc", ticker: "VED", project: "Rampura Agucha", location: "India", commodity: "Zinc", stage: "Operating" },
  { company: "Vedanta Zinc", ticker: "VED", project: "Sindesar Khurd", location: "India", commodity: "Zinc", stage: "Operating" },
  { company: "Hindustan Zinc", ticker: "HZ", project: "Zawar", location: "India", commodity: "Zinc", stage: "Operating" },
  { company: "Hindustan Zinc", ticker: "HZ", project: "Rampura Agucha", location: "India", commodity: "Zinc", stage: "Operating" },
  { company: "Boliden", ticker: "BOL", project: "Aitik", location: "Sweden", commodity: "Copper", stage: "Operating" },
  { company: "Boliden", ticker: "BOL", project: "Garpenberg", location: "Sweden", commodity: "Zinc", stage: "Operating" },
  { company: "Boliden", ticker: "BOL", project: "Kevitsa", location: "Finland", commodity: "Nickel", stage: "Operating" },
  { company: "Boliden", ticker: "BOL", project: "Tara", location: "Ireland", commodity: "Zinc", stage: "Operating" },
  { company: "Dundee Precious Metals", ticker: "DPM", project: "Loma Larga", location: "Ecuador", commodity: "Gold", stage: "Development" },
  { company: "Orosur Mining", ticker: "OMI", project: "Anza", location: "Colombia", commodity: "Gold", stage: "Development" },
  { company: "Orosur Mining", ticker: "OMI", project: "San Gregorio", location: "Uruguay", commodity: "Gold", stage: "Exploration" },
  { company: "Continental Gold", ticker: "CNL", project: "Buriticá", location: "Colombia", commodity: "Gold", stage: "Operating" },
  { company: "Gran Colombia Gold", ticker: "GCM", project: "Segovia", location: "Colombia", commodity: "Gold", stage: "Operating" },
  { company: "Gran Colombia Gold", ticker: "GCM", project: "Marmato", location: "Colombia", commodity: "Gold", stage: "Operating" },
  { company: "Caldas Gold", ticker: "CGC", project: "Marmato", location: "Colombia", commodity: "Gold", stage: "Development" },
  { company: "Red Eagle Mining", ticker: "R", project: "San Ramón", location: "Colombia", commodity: "Gold", stage: "Operating" },
  { company: "Mineros", ticker: "MSA", project: "Hemco", location: "Nicaragua", commodity: "Gold", stage: "Operating" },
  { company: "Condor Gold", ticker: "CNR", project: "La India", location: "Nicaragua", commodity: "Gold", stage: "Development" },
  { company: "Calibre Mining", ticker: "CXB", project: "Libertad", location: "Nicaragua", commodity: "Gold", stage: "Operating" },
  { company: "Calibre Mining", ticker: "CXB", project: "Limon", location: "Nicaragua", commodity: "Gold", stage: "Operating" },
  { company: "Mako Mining", ticker: "MKO", project: "San Albino", location: "Nicaragua", commodity: "Gold", stage: "Operating" },
  { company: "Bluestone Resources", ticker: "BSR", project: "Cerro Blanco", location: "Guatemala", commodity: "Gold", stage: "Development" },
  { company: "Tahoe Resources", ticker: "TAHO", project: "Escobal", location: "Guatemala", commodity: "Silver", stage: "Suspended" },
  { company: "Fortuna Silver", ticker: "FSM", project: "San Jose", location: "Mexico", commodity: "Silver", stage: "Operating" },
  { company: "Fortuna Silver", ticker: "FSM", project: "Lindero", location: "Argentina", commodity: "Gold", stage: "Operating" },
  { company: "Minera Alamos", ticker: "MAI", project: "Santana", location: "Mexico", commodity: "Gold", stage: "Development" },
  { company: "Minera Alamos", ticker: "MAI", project: "Cerro de Oro", location: "Mexico", commodity: "Gold", stage: "Exploration" },
  { company: "Argonaut Gold", ticker: "AR", project: "El Castillo", location: "Mexico", commodity: "Gold", stage: "Operating" },
  { company: "Argonaut Gold", ticker: "AR", project: "La Colorada", location: "Mexico", commodity: "Gold", stage: "Operating" },
  { company: "Argonaut Gold", ticker: "AR", project: "Magino", location: "Canada", commodity: "Gold", stage: "Development" },
  { company: "Premier Gold", ticker: "PG", project: "McCoy-Cove", location: "United States", commodity: "Gold", stage: "Exploration" },
  { company: "Premier Gold", ticker: "PG", project: "South Arturo", location: "United States", commodity: "Gold", stage: "Operating" },
  { company: "Premier Gold", ticker: "PG", project: "Hardrock", location: "Canada", commodity: "Gold", stage: "Development" },
  { company: "Bonterra Resources", ticker: "BTR", project: "Gladiator", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Bonterra Resources", ticker: "BTR", project: "Barry", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Maple Gold Mines", ticker: "MGM", project: "Douay", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Victoria Gold", ticker: "VGCX", project: "Eagle", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Nighthawk Gold", ticker: "NHK", project: "Indin Lake", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Sabina Gold & Silver", ticker: "SBB", project: "Back River", location: "Canada", commodity: "Gold", stage: "Development" },
  { company: "TMAC Resources", ticker: "TMR", project: "Hope Bay", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Agnico Eagle Mines", ticker: "AEM", project: "Hope Bay", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Agnico Eagle Mines", ticker: "AEM", project: "Amaruq", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Agnico Eagle Mines", ticker: "AEM", project: "Hammond Reef", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Wesdome Gold", ticker: "WDO", project: "Eagle River", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Wesdome Gold", ticker: "WDO", project: "Mishi", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Wesdome Gold", ticker: "WDO", project: "Kiena", location: "Canada", commodity: "Gold", stage: "Development" },
  { company: "Kirkland Lake Gold", ticker: "KL", project: "Fosterville", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Kirkland Lake Gold", ticker: "KL", project: "Macassa", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Kirkland Lake Gold", ticker: "KL", project: "Detour Lake", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Kirkland Lake Gold", ticker: "KL", project: "Holt", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Kirkland Lake Gold", ticker: "KL", project: "Taylor", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "St Barbara", ticker: "SBM", project: "Gwalia", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "St Barbara", ticker: "SBM", project: "Leonora", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "St Barbara", ticker: "SBM", project: "Atlantic", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "St Barbara", ticker: "SBM", project: "Touquoy", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Saracen Mineral", ticker: "SAR", project: "Carosue Dam", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Saracen Mineral", ticker: "SAR", project: "Thunderbox", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Saracen Mineral", ticker: "SAR", project: "Deep South", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "Red 5", ticker: "RED", project: "King of the Hills", location: "Australia", commodity: "Gold", stage: "Development" },
  { company: "Red 5", ticker: "RED", project: "Darlot", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Gascoyne Resources", ticker: "GCY", project: "Dalgaranga", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Gascoyne Resources", ticker: "GCY", project: "Glenburgh", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "Ora Banda Mining", ticker: "OBM", project: "Davyhurst", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Ora Banda Mining", ticker: "OBM", project: "Mount Ida", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Calidus Resources", ticker: "CAI", project: "Warrawoona", location: "Australia", commodity: "Gold", stage: "Development" },
  { company: "Calidus Resources", ticker: "CAI", project: "Blue Spec", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "Bardoc Gold", ticker: "BDC", project: "Aphrodite", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "Bardoc Gold", ticker: "BDC", project: "Zoroastrian", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "Emmerson Resources", ticker: "ERM", project: "Tennant Creek", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "Tennant Minerals", ticker: "TMS", project: "Barkly", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "White Rock Minerals", ticker: "WRM", project: "Red Mountain", location: "United States", commodity: "Zinc", stage: "Exploration" },
  { company: "White Rock Minerals", ticker: "WRM", project: "Woods Point", location: "Australia", commodity: "Gold", stage: "Exploration" },
  { company: "Azimut Exploration", ticker: "AZM", project: "Eleonore South", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Azimut Exploration", ticker: "AZM", project: "Patwon", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Probe Metals", ticker: "PRB", project: "Val-d'Or East", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Probe Metals", ticker: "PRB", project: "Detour East", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Wallbridge Mining", ticker: "WM", project: "Fenelon", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Wallbridge Mining", ticker: "WM", project: "Martiniere", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Osisko Mining", ticker: "OSK", project: "Windfall Lake", location: "Canada", commodity: "Gold", stage: "Development" },
  { company: "Osisko Mining", ticker: "OSK", project: "Urban Barry", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Eldorado Gold", ticker: "EGO", project: "Lamaque", location: "Canada", commodity: "Gold", stage: "Operating" },
  { company: "Eldorado Gold", ticker: "EGO", project: "Efemcukuru", location: "Turkey", commodity: "Gold", stage: "Operating" },
  { company: "Eldorado Gold", ticker: "EGO", project: "Olympias", location: "Greece", commodity: "Gold", stage: "Operating" },
  { company: "Eldorado Gold", ticker: "EGO", project: "Stratoni", location: "Greece", commodity: "Lead", stage: "Operating" },
  { company: "Eldorado Gold", ticker: "EGO", project: "Skouries", location: "Greece", commodity: "Copper", stage: "Development" },
  { company: "Eldorado Gold", ticker: "EGO", project: "Kisladag", location: "Turkey", commodity: "Gold", stage: "Operating" },
  { company: "Medgold Resources", ticker: "MED", project: "Tlamino", location: "Serbia", commodity: "Gold", stage: "Exploration" },
  { company: "Medgold Resources", ticker: "MED", project: "Barje", location: "Serbia", commodity: "Gold", stage: "Exploration" },
  { company: "Adriatic Metals", ticker: "ADT", project: "Vares", location: "Bosnia", commodity: "Silver", stage: "Development" },
  { company: "Adriatic Metals", ticker: "ADT", project: "Rupice", location: "Bosnia", commodity: "Silver", stage: "Development" },
  { company: "Adriatic Metals", ticker: "ADT", project: "Veovaca", location: "Bosnia", commodity: "Copper", stage: "Exploration" },
  { company: "European Metals", ticker: "EMH", project: "Cinovec", location: "Czech Republic", commodity: "Lithium", stage: "Development" },
  { company: "Bacanora Lithium", ticker: "BCN", project: "Sonora", location: "Mexico", commodity: "Lithium", stage: "Development" },
  { company: "Bacanora Lithium", ticker: "BCN", project: "Zinnwald", location: "Germany", commodity: "Lithium", stage: "Development" },
  { company: "Infinity Lithium", ticker: "INF", project: "San José", location: "Spain", commodity: "Lithium", stage: "Development" },
  { company: "Vulcan Energy", ticker: "VUL", project: "Upper Rhine Valley", location: "Germany", commodity: "Lithium", stage: "Development" },
  { company: "Neo Lithium", ticker: "NLC", project: "Tres Quebradas", location: "Argentina", commodity: "Lithium", stage: "Development" },
  { company: "Arena Minerals", ticker: "AN", project: "Antofalla", location: "Argentina", commodity: "Lithium", stage: "Exploration" },
  { company: "Millennial Lithium", ticker: "ML", project: "Pastos Grandes", location: "Argentina", commodity: "Lithium", stage: "Development" },
  { company: "LSC Lithium", ticker: "LSC", project: "Pozuelos-Pastos Grandes", location: "Argentina", commodity: "Lithium", stage: "Development" },
  { company: "AVZ Minerals", ticker: "AVZ", project: "Manono", location: "Democratic Republic of Congo", commodity: "Lithium", stage: "Development" },
  { company: "Prospect Resources", ticker: "PSC", project: "Arcadia", location: "Zimbabwe", commodity: "Lithium", stage: "Development" },
  { company: "Bikita Minerals", ticker: "BM", project: "Bikita", location: "Zimbabwe", commodity: "Lithium", stage: "Operating" },
  { company: "Premier African Minerals", ticker: "PREM", project: "Zulu", location: "Zimbabwe", commodity: "Lithium", stage: "Development" },
  { company: "Kodal Minerals", ticker: "KOD", project: "Bougouni", location: "Mali", commodity: "Lithium", stage: "Development" },
];

// Final batch of news articles
const MORE_NEWS = [
  { title: "Newmont maintains position as top gold producer globally", source: "Mining.com", published: "2025-01-15", summary: "Newmont reports strong annual production exceeding 6 million ounces.", commodities: ["Gold"], sentiment: "Positive" },
  { title: "Three-month copper futures reach $10,600 on LME", source: "Bloomberg", published: "2025-01-28", summary: "Copper futures surge amid supply concerns.", commodities: ["Copper"], sentiment: "Positive" },
  { title: "Critical mineral demand to triple by 2030", source: "IEA", published: "2025-01-03", summary: "IEA projects massive increase in demand for transition minerals.", commodities: ["Diversified"], sentiment: "Positive" },
  { title: "Exploration activity plateaus in 2024", source: "S&P Global", published: "2025-01-08", summary: "Global exploration spending shows mixed results by commodity.", commodities: ["Diversified"], sentiment: "Neutral" },
  { title: "754 mining acquisitions total $99.73B in 2024", source: "GlobalData", published: "2025-01-05", summary: "M&A activity increases 4% year-over-year.", commodities: ["Diversified"], sentiment: "Positive" },
  { title: "No new copper discoveries in 2022-2023", source: "S&P Global", published: "2024-12-10", summary: "Exploration challenges mount despite higher budgets.", commodities: ["Copper"], sentiment: "Negative" },
  { title: "Record gold prices drive EBITDA up 32%", source: "PwC", published: "2025-01-02", summary: "Gold miners see profit surge despite production costs.", commodities: ["Gold"], sentiment: "Positive" },
  { title: "Resource depletion emerges as top-5 industry risk", source: "EY", published: "2025-01-01", summary: "Lack of new discoveries concerns mining executives.", commodities: ["Diversified"], sentiment: "Negative" },
];

async function populateFinal() {
  console.log(`🎯 FINAL POPULATION BATCH\n`);
  let pAdded = 0, cAdded = 0, nAdded = 0;

  // Add projects
  for (const item of MORE_PROJECTS) {
    let companyId: string;
    const { data: existingCompany } = await supabase.from('companies').select('id').eq('name', item.company).single();
    
    if (existingCompany) {
      companyId = existingCompany.id;
    } else {
      const { data: newCompany } = await supabase.from('companies').insert({
        name: item.company,
        ticker: item.ticker,
        exchange: 'ASX',
        description: `Mining company focused on ${item.commodity}.`,
      }).select('id').single();
      if (newCompany) {
        companyId = newCompany.id;
        cAdded++;
      } else continue;
    }

    const { data: existingProject } = await supabase.from('projects').select('id').eq('name', item.project).eq('company_id', companyId).single();
    if (!existingProject) {
      await supabase.from('projects').insert({
        company_id: companyId,
        name: item.project,
        location: item.location,
        stage: item.stage,
        commodities: [item.commodity],
        status: item.stage === 'Operating' ? 'Active' : 'In Development',
        description: `${item.project} - ${item.commodity} project in ${item.location}.`,
        urls: [],
      });
      pAdded++;
      if (pAdded % 25 === 0) console.log(`📊 ${pAdded} projects added...`);
    }
  }

  // Add news
  for (const article of MORE_NEWS) {
    const { data: existing } = await supabase.from('news').select('id').eq('title', article.title).single();
    if (!existing) {
      await supabase.from('news').insert({
        title: article.title,
        urls: [],
        source: article.source,
        published_at: new Date(article.published).toISOString(),
        summary: article.summary,
        commodities: article.commodities,
        sentiment: article.sentiment,
        project_ids: [],
      });
      nAdded++;
    }
  }

  console.log(`\n✅ Final batch complete:`);
  console.log(`   Companies added: ${cAdded}`);
  console.log(`   Projects added: ${pAdded}`);
  console.log(`   News added: ${nAdded}\n`);

  const { count: companies } = await supabase.from('companies').select('*', { count: 'exact', head: true });
  const { count: projects } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  const { count: news } = await supabase.from('news').select('*', { count: 'exact', head: true });
  
  console.log(`📊 FINAL DATABASE TOTALS:`);
  console.log(`   Companies: ${companies}`);
  console.log(`   Projects: ${projects}`);
  console.log(`   News: ${news}\n`);
}

populateFinal();
