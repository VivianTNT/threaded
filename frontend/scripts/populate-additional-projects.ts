import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

// Additional mining projects from web research (2024-2025)
const ADDITIONAL_PROJECTS = [
  // Africa - DRC Projects
  { company: "Ivanhoe Mines", ticker: "IVN", project: "Kamoa North", location: "Democratic Republic of Congo", commodity: "Copper", stage: "Development" },
  { company: "Ivanhoe Mines", ticker: "IVN", project: "Kakula West", location: "Democratic Republic of Congo", commodity: "Copper", stage: "Development" },
  { company: "CMOC Group", ticker: "CMOC", project: "Tenke Fungurume", location: "Democratic Republic of Congo", commodity: "Copper", stage: "Operating" },
  { company: "Zijin Mining", ticker: "2899", project: "Kolwezi Copper Mine", location: "Democratic Republic of Congo", commodity: "Copper", stage: "Operating" },
  { company: "ERG", ticker: "ERG", project: "Boss Mining", location: "Democratic Republic of Congo", commodity: "Copper", stage: "Operating" },
  { company: "ERG", ticker: "ERG", project: "Frontier Mine", location: "Democratic Republic of Congo", commodity: "Copper", stage: "Operating" },

  // Zambia Projects
  { company: "First Quantum Minerals", ticker: "FM", project: "Mopani Copper Mines", location: "Zambia", commodity: "Copper", stage: "Operating" },
  { company: "Vedanta Resources", ticker: "VED", project: "Konkola Copper Mines", location: "Zambia", commodity: "Copper", stage: "Operating" },
  { company: "Barrick Gold", ticker: "GOLD", project: "Lumwana", location: "Zambia", commodity: "Copper", stage: "Operating" },

  // Ghana Projects
  { company: "Cardinal Resources", ticker: "CDV", project: "Namdini", location: "Ghana", commodity: "Gold", stage: "Development" },
  { company: "Newmont Corporation", ticker: "NEM", project: "Akyem", location: "Ghana", commodity: "Gold", stage: "Operating" },
  { company: "Asanko Gold", ticker: "AKG", project: "Asanko", location: "Ghana", commodity: "Gold", stage: "Operating" },
  { company: "Chirano Gold", ticker: "CG", project: "Chirano", location: "Ghana", commodity: "Gold", stage: "Operating" },

  // Tanzania Projects
  { company: "AngloGold Ashanti", ticker: "AU", project: "Nyanzaga", location: "Tanzania", commodity: "Gold", stage: "Development" },
  { company: "Barrick Gold", ticker: "GOLD", project: "Bulyanhulu", location: "Tanzania", commodity: "Gold", stage: "Operating" },
  { company: "Kabanga Nickel", ticker: "KNI", project: "Kabanga", location: "Tanzania", commodity: "Nickel", stage: "Development" },

  // Burkina Faso Projects
  { company: "SEMAFO", ticker: "SMF", project: "Mana", location: "Burkina Faso", commodity: "Gold", stage: "Operating" },
  { company: "Orezone Gold", ticker: "ORE", project: "Bomboré", location: "Burkina Faso", commodity: "Gold", stage: "Operating" },
  { company: "West African Resources", ticker: "WAF", project: "Sanbrado", location: "Burkina Faso", commodity: "Gold", stage: "Operating" },
  { company: "Teranga Gold", ticker: "TGZ", project: "Wahgnion", location: "Burkina Faso", commodity: "Gold", stage: "Operating" },

  // Mali Projects
  { company: "Hummingbird Resources", ticker: "HUM", project: "Yanfolila", location: "Mali", commodity: "Gold", stage: "Operating" },
  { company: "Barrick Gold", ticker: "GOLD", project: "Tongon", location: "Mali", commodity: "Gold", stage: "Operating" },
  { company: "Resolute Mining", ticker: "RSG", project: "Tabakoroni", location: "Mali", commodity: "Gold", stage: "Development" },
  { company: "Robex Resources", ticker: "RBX", project: "Nampala", location: "Mali", commodity: "Gold", stage: "Operating" },

  // Chile Projects
  { company: "CODELCO", ticker: "CODELCO", project: "Chuquicamata", location: "Chile", commodity: "Copper", stage: "Operating" },
  { company: "CODELCO", ticker: "CODELCO", project: "El Teniente", location: "Chile", commodity: "Copper", stage: "Operating" },
  { company: "CODELCO", ticker: "CODELCO", project: "Andina", location: "Chile", commodity: "Copper", stage: "Operating" },
  { company: "CODELCO", ticker: "CODELCO", project: "Salvador", location: "Chile", commodity: "Copper", stage: "Operating" },
  { company: "CODELCO", ticker: "CODELCO", project: "Radomiro Tomic", location: "Chile", commodity: "Copper", stage: "Operating" },
  { company: "Lundin Mining", ticker: "LUN", project: "Caserones", location: "Chile", commodity: "Copper", stage: "Operating" },
  { company: "Capstone Copper", ticker: "CS", project: "Mantoverde", location: "Chile", commodity: "Copper", stage: "Operating" },
  { company: "Teck Resources", ticker: "TECK", project: "Quebrada Blanca Phase 2", location: "Chile", commodity: "Copper", stage: "Development" },
  { company: "Kinross Gold", ticker: "KGC", project: "La Coipa", location: "Chile", commodity: "Gold", stage: "Operating" },

  // Peru Projects
  { company: "MMG Limited", ticker: "1208", project: "Las Bambas", location: "Peru", commodity: "Copper", stage: "Operating" },
  { company: "Hudbay Minerals", ticker: "HBM", project: "Pampacancha", location: "Peru", commodity: "Copper", stage: "Development" },
  { company: "Bear Creek Mining", ticker: "BCM", project: "Corani", location: "Peru", commodity: "Silver", stage: "Development" },
  { company: "Hochschild Mining", ticker: "HOC", project: "Inmaculada", location: "Peru", commodity: "Gold", stage: "Operating" },
  { company: "Hochschild Mining", ticker: "HOC", project: "Pallancata", location: "Peru", commodity: "Silver", stage: "Operating" },
  { company: "Fortuna Silver", ticker: "FSM", project: "Caylloma", location: "Peru", commodity: "Silver", stage: "Operating" },
  { company: "Nexa Resources", ticker: "NEXA", project: "Atacocha", location: "Peru", commodity: "Zinc", stage: "Operating" },
  { company: "Nexa Resources", ticker: "NEXA", project: "Cerro Lindo", location: "Peru", commodity: "Zinc", stage: "Operating" },

  // Brazil Projects
  { company: "Vale", ticker: "VALE", project: "Salobo", location: "Brazil", commodity: "Copper", stage: "Operating" },
  { company: "Vale", ticker: "VALE", project: "Carajás", location: "Brazil", commodity: "Iron Ore", stage: "Operating" },
  { company: "AngloGold Ashanti", ticker: "AU", project: "Serra Grande", location: "Brazil", commodity: "Gold", stage: "Operating" },
  { company: "Equinox Gold", ticker: "EQX", project: "Aurizona", location: "Brazil", commodity: "Gold", stage: "Operating" },
  { company: "Equinox Gold", ticker: "EQX", project: "Fazenda", location: "Brazil", commodity: "Gold", stage: "Operating" },
  { company: "Jaguar Mining", ticker: "JAG", project: "Turmalina", location: "Brazil", commodity: "Gold", stage: "Operating" },
  { company: "Lundin Mining", ticker: "LUN", project: "Chapada", location: "Brazil", commodity: "Copper", stage: "Operating" },

  // Argentina Projects
  { company: "Lundin Mining", ticker: "LUN", project: "Josemaria", location: "Argentina", commodity: "Copper", stage: "Development" },
  { company: "Galan Lithium", ticker: "GLN", project: "Hombre Muerto West", location: "Argentina", commodity: "Lithium", stage: "Development" },
  { company: "Allkem", ticker: "AKE", project: "Olaroz", location: "Argentina", commodity: "Lithium", stage: "Operating" },
  { company: "Livent Corporation", ticker: "LTHM", project: "Fenix", location: "Argentina", commodity: "Lithium", stage: "Operating" },
  { company: "Yamana Gold", ticker: "YRI", project: "Cerro Moro", location: "Argentina", commodity: "Gold", stage: "Operating" },
  { company: "Barrick Gold", ticker: "GOLD", project: "Veladero", location: "Argentina", commodity: "Gold", stage: "Operating" },
  { company: "McEwen Mining", ticker: "MUX", project: "San José", location: "Argentina", commodity: "Gold", stage: "Operating" },

  // Canada Junior Mining Projects
  { company: "Eros Resources", ticker: "ERC", project: "Bell Creek", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Prospector Metals", ticker: "PPP", project: "Savant Lake", location: "Canada", commodity: "Gold", stage: "Exploration" },
  { company: "Blue Star Gold", ticker: "BAU", project: "Ulu", location: "Canada", commodity: "Gold", stage: "Development" },
  { company: "Lahontan Gold", ticker: "LG", project: "Santa Fe", location: "United States", commodity: "Gold", stage: "Exploration" },
  { company: "Goldgroup Mining", ticker: "GGA", project: "Cerro Prieto", location: "Mexico", commodity: "Gold", stage: "Operating" },
  { company: "San Lorenzo Gold", ticker: "SLG", project: "San Lorenzo", location: "Mexico", commodity: "Gold", stage: "Development" },
  { company: "Camino Minerals", ticker: "COR", project: "Chapitos", location: "Peru", commodity: "Copper", stage: "Exploration" },
  { company: "C3 Metals", ticker: "CCCM", project: "Bellas Gate", location: "Jamaica", commodity: "Copper", stage: "Exploration" },
  { company: "Turmalina Metals", ticker: "TRML", project: "Colquemayo", location: "Peru", commodity: "Silver", stage: "Exploration" },
  { company: "New Age Metals", ticker: "NAM", project: "River Valley", location: "Canada", commodity: "Platinum", stage: "Exploration" },
  { company: "Surge Battery Metals", ticker: "NILI", project: "Nevada North Lithium", location: "United States", commodity: "Lithium", stage: "Exploration" },

  // Australia Projects
  { company: "Newcrest Mining", ticker: "NCM", project: "Cadia East", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Newcrest Mining", ticker: "NCM", project: "Telfer", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Newcrest Mining", ticker: "NCM", project: "Lihir", location: "Papua New Guinea", commodity: "Gold", stage: "Operating" },
  { company: "OZ Minerals", ticker: "OZL", project: "West Musgrave", location: "Australia", commodity: "Nickel", stage: "Development" },
  { company: "Northern Minerals", ticker: "NTU", project: "Browns Range", location: "Australia", commodity: "Rare Earths", stage: "Development" },
  { company: "Syrah Resources", ticker: "SYR", project: "Balama", location: "Mozambique", commodity: "Graphite", stage: "Operating" },
  { company: "Hastings Technology Metals", ticker: "HAS", project: "Yangibana", location: "Australia", commodity: "Rare Earths", stage: "Development" },
  { company: "Arafura Resources", ticker: "ARU", project: "Nolans", location: "Australia", commodity: "Rare Earths", stage: "Development" },
  { company: "Ionic Rare Earths", ticker: "IXR", project: "Makuutu", location: "Uganda", commodity: "Rare Earths", stage: "Development" },
  { company: "Iluka Resources", ticker: "ILU", project: "Eneabba", location: "Australia", commodity: "Rare Earths", stage: "Development" },
  { company: "Alkane Resources", ticker: "ALK", project: "Tomingley", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Alkane Resources", ticker: "ALK", project: "Boda", location: "Australia", commodity: "Copper", stage: "Exploration" },
  { company: "Capricorn Metals", ticker: "CMM", project: "Karlawinda", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Dacian Gold", ticker: "DCN", project: "Mt Morgans", location: "Australia", commodity: "Gold", stage: "Operating" },
  { company: "Genesis Minerals", ticker: "GMD", project: "Ulysses", location: "Australia", commodity: "Gold", stage: "Development" },

  // Indonesia & SE Asia Projects
  { company: "Nickel Industries", ticker: "NIC", project: "Hengjaya Mine", location: "Indonesia", commodity: "Nickel", stage: "Operating" },
  { company: "Nickel Industries", ticker: "NIC", project: "Ranger", location: "Indonesia", commodity: "Nickel", stage: "Operating" },
  { company: "PT Aneka Tambang", ticker: "ANTM", project: "Pongkor", location: "Indonesia", commodity: "Gold", stage: "Operating" },
  { company: "PT Freeport Indonesia", ticker: "FCX", project: "Deep MLZ", location: "Indonesia", commodity: "Copper", stage: "Development" },
  { company: "Philex Mining", ticker: "PX", project: "Padcal", location: "Philippines", commodity: "Copper", stage: "Operating" },

  // Central Asia Projects
  { company: "Polymetal", ticker: "POLY", project: "Kyzyl", location: "Kazakhstan", commodity: "Gold", stage: "Operating" },
  { company: "Polymetal", ticker: "POLY", project: "Varvara", location: "Kazakhstan", commodity: "Gold", stage: "Operating" },
  { company: "Centerra Gold", ticker: "CG", project: "Kumtor", location: "Kyrgyzstan", commodity: "Gold", stage: "Operating" },
  { company: "Nordgold", ticker: "NORD", project: "Gross", location: "Russia", commodity: "Gold", stage: "Operating" },
  { company: "Nordgold", ticker: "NORD", project: "Suzdal", location: "Kazakhstan", commodity: "Gold", stage: "Operating" },

  // Mexico Projects
  { company: "Agnico Eagle Mines", ticker: "AEM", project: "Pinos Altos", location: "Mexico", commodity: "Gold", stage: "Operating" },
  { company: "Agnico Eagle Mines", ticker: "AEM", project: "La India", location: "Mexico", commodity: "Gold", stage: "Operating" },
  { company: "Torex Gold", ticker: "TXG", project: "ELG Complex", location: "Mexico", commodity: "Gold", stage: "Operating" },
  { company: "MAG Silver", ticker: "MAG", project: "Juanicipio", location: "Mexico", commodity: "Silver", stage: "Operating" },
  { company: "First Majestic Silver", ticker: "AG", project: "San Dimas", location: "Mexico", commodity: "Silver", stage: "Operating" },
  { company: "First Majestic Silver", ticker: "AG", project: "Santa Elena", location: "Mexico", commodity: "Silver", stage: "Operating" },
  { company: "First Majestic Silver", ticker: "AG", project: "La Encantada", location: "Mexico", commodity: "Silver", stage: "Operating" },
  { company: "Endeavour Silver", ticker: "EXK", project: "Guanaceví", location: "Mexico", commodity: "Silver", stage: "Operating" },
  { company: "Endeavour Silver", ticker: "EXK", project: "Bolanitos", location: "Mexico", commodity: "Silver", stage: "Operating" },
  { company: "Alamos Gold", ticker: "AGI", project: "Mulatos", location: "Mexico", commodity: "Gold", stage: "Operating" },

  // USA Projects
  { company: "Nevada Gold Mines", ticker: "NEM", project: "Phoenix", location: "United States", commodity: "Gold", stage: "Development" },
  { company: "Nevada Gold Mines", ticker: "NEM", project: "Long Canyon", location: "United States", commodity: "Gold", stage: "Operating" },
  { company: "Coeur Mining", ticker: "CDE", project: "Rochester", location: "United States", commodity: "Silver", stage: "Operating" },
  { company: "Coeur Mining", ticker: "CDE", project: "Palmarejo", location: "Mexico", commodity: "Gold", stage: "Operating" },
  { company: "Royal Gold", ticker: "RGLD", project: "Andacollo", location: "Chile", commodity: "Gold", stage: "Operating" },
  { company: "Taseko Mines", ticker: "TKO", project: "Florence", location: "United States", commodity: "Copper", stage: "Development" },
  { company: "McEwen Mining", ticker: "MUX", project: "Gold Bar", location: "United States", commodity: "Gold", stage: "Operating" },
  { company: "i-80 Gold", ticker: "IAUX", project: "Ruby Hill", location: "United States", commodity: "Gold", stage: "Operating" },
  { company: "i-80 Gold", ticker: "IAUX", project: "Lone Tree", location: "United States", commodity: "Gold", stage: "Development" },

  // Rare Earth & Battery Metals
  { company: "Rimbal", ticker: "RIM", project: "Tanbreez", location: "Greenland", commodity: "Rare Earths", stage: "Exploration" },
  { company: "Magris Resources", ticker: "MGR", project: "St Honoré", location: "Canada", commodity: "Rare Earths", stage: "Development" },
  { company: "Rainbow Rare Earths", ticker: "RBW", project: "Phalaborwa", location: "South Africa", commodity: "Rare Earths", stage: "Operating" },
  { company: "Northern Graphite", ticker: "NGC", project: "Bissett Creek", location: "Canada", commodity: "Graphite", stage: "Development" },
  { company: "Titan Mining", ticker: "TI", project: "Empire State", location: "United States", commodity: "Zinc", stage: "Operating" },
  { company: "American Lithium", ticker: "LI", project: "Falchani", location: "Peru", commodity: "Lithium", stage: "Exploration" },
  { company: "Frontier Lithium", ticker: "FL", project: "PAK", location: "Canada", commodity: "Lithium", stage: "Development" },
  { company: "Patriot Battery Metals", ticker: "PMET", project: "Corvette", location: "Canada", commodity: "Lithium", stage: "Exploration" },
  { company: "Lake Resources", ticker: "LKE", project: "Kachi", location: "Argentina", commodity: "Lithium", stage: "Development" },
  { company: "Critical Elements", ticker: "CRE", project: "Rose", location: "Canada", commodity: "Lithium", stage: "Development" },
  { company: "Brunswick Exploration", ticker: "BRW", project: "Mirage", location: "Canada", commodity: "Lithium", stage: "Exploration" },

  // Additional Global Projects
  { company: "Greatland Gold", ticker: "GGP", project: "Havieron", location: "Australia", commodity: "Gold", stage: "Development" },
  { company: "Dundee Precious Metals", ticker: "DPM", project: "Ada Tepe", location: "Bulgaria", commodity: "Gold", stage: "Operating" },
  { company: "Dundee Precious Metals", ticker: "DPM", project: "Chelopech", location: "Bulgaria", commodity: "Gold", stage: "Operating" },
  { company: "Centamin", ticker: "CEY", project: "Sukari", location: "Egypt", commodity: "Gold", stage: "Operating" },
  { company: "Shanta Gold", ticker: "SHG", project: "New Luika", location: "Tanzania", commodity: "Gold", stage: "Operating" },
  { company: "Acacia Mining", ticker: "ACA", project: "Buzwagi", location: "Tanzania", commodity: "Gold", stage: "Operating" },
  { company: "Endeavour Mining", ticker: "EDV", project: "Sabodala-Massawa", location: "Senegal", commodity: "Gold", stage: "Operating" },
  { company: "Teranga Gold", ticker: "TGZ", project: "Massawa", location: "Senegal", commodity: "Gold", stage: "Operating" },
  { company: "West African Resources", ticker: "WAF", project: "M1 South", location: "Burkina Faso", commodity: "Gold", stage: "Development" },
  { company: "Roxgold", ticker: "ROXG", project: "Yaramoko", location: "Burkina Faso", commodity: "Gold", stage: "Operating" },
];

let companiesAdded = 0;
let projectsAdded = 0;

async function populateAdditionalProjects() {
  console.log(`🚀 POPULATING ADDITIONAL MINING PROJECTS (2024-2025 Research)`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);
  console.log(`📊 Additional projects to add: ${ADDITIONAL_PROJECTS.length}`);
  console.log(`📋 Source: Web research - Africa, South America, Asia, Australia\n`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);

  for (const item of ADDITIONAL_PROJECTS) {
    // Check if company exists, add if not
    let companyId: string;

    const { data: existingCompany } = await supabase
      .from('companies')
      .select('id')
      .eq('name', item.company)
      .single();

    if (existingCompany) {
      companyId = existingCompany.id;
    } else {
      // Add new company
      const { data: newCompany, error: companyError } = await supabase
        .from('companies')
        .insert({
          name: item.company,
          ticker: item.ticker,
          exchange: item.ticker?.includes('X') || item.ticker?.includes('M') || item.ticker?.includes('G') ? 'NYSE' :
                    item.ticker === 'CODELCO' ? 'Private' :
                    item.ticker?.length === 4 ? 'HKEX' : 'ASX',
          description: `${item.company} operates mining projects focused on ${item.commodity}.`,
        })
        .select('id')
        .single();

      if (companyError) {
        console.error(`❌ Error adding company ${item.company}:`, companyError.message);
        continue;
      }

      companyId = newCompany.id;
      companiesAdded++;
      console.log(`✓ Added company: ${item.company}`);
    }

    // Check if project already exists
    const { data: existingProject } = await supabase
      .from('projects')
      .select('id')
      .eq('name', item.project)
      .eq('company_id', companyId)
      .single();

    if (existingProject) {
      continue; // Skip duplicates
    }

    // Add project
    const { error: projectError } = await supabase
      .from('projects')
      .insert({
        company_id: companyId,
        name: item.project,
        location: item.location,
        stage: item.stage,
        commodities: [item.commodity],
        status: item.stage === 'Operating' ? 'Active' :
                item.stage === 'Closed' ? 'Closed' : 'In Development',
        description: `${item.project} is a ${item.commodity} mining project in ${item.location}, ${item.stage.toLowerCase()} by ${item.company}.`,
        urls: [],
      });

    if (projectError) {
      console.error(`  ❌ Error adding ${item.project}:`, projectError.message);
    } else {
      projectsAdded++;
      if (projectsAdded % 25 === 0) {
        console.log(`📊 Progress: ${projectsAdded} projects added...`);
      }
    }
  }

  console.log(`\n═══════════════════════════════════════════════════════════════`);
  console.log(`✅ ADDITIONAL PROJECTS POPULATED!`);
  console.log(`═══════════════════════════════════════════════════════════════`);
  console.log(`📊 Session Summary:`);
  console.log(`   New companies added: ${companiesAdded}`);
  console.log(`   New projects added: ${projectsAdded}`);

  const { count: finalProjects } = await supabase.from('projects').select('*', { count: 'exact', head: true });
  const { count: finalCompanies } = await supabase.from('companies').select('*', { count: 'exact', head: true });
  console.log(`\n📊 Current Database Totals:`);
  console.log(`   Companies: ${finalCompanies}`);
  console.log(`   Projects: ${finalProjects}`);
  console.log(`═══════════════════════════════════════════════════════════════\n`);
}

populateAdditionalProjects();
