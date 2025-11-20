import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
);

const testProjects = [
  {
    name: 'Cortez Hills Gold Project',
    location: 'Nevada, USA',
    latitude: 40.1833,
    longitude: -116.6167,
    stage: 'Production',
    status: 'Active',
    commodities: ['Gold'],
    description: 'Large-scale open pit and underground gold mine in the Cortez district. Mine life: 15 years. Annual production: 350,000 oz Au. Payback: 3.2 years. Mining: Open pit and underground. Processing: Heap leach and mill.',
    npv: 485.3,
    irr: 22.4,
    capex: 420.5,
    aisc: 850,
    resource: '5.2M oz Au M&I @ 1.4 g/t',
    reserve: '3.8M oz Au P&P @ 1.2 g/t',
    qualified_persons: [
      {
        name: 'Dr. Sarah Chen',
        credentials: 'Ph.D., P.Eng., MAusIMM',
        company: 'SRK Consulting'
      },
      {
        name: 'Michael Roberts',
        credentials: 'M.Sc., P.Geo.',
        company: 'Barrick Gold Corporation'
      }
    ]
  },
  {
    name: 'Kamoa-Kakula Copper Project',
    location: 'Democratic Republic of Congo',
    latitude: -10.4667,
    longitude: 25.7833,
    stage: 'Production',
    status: 'Active',
    commodities: ['Copper'],
    description: 'World-class high-grade copper discovery in the Central African Copperbelt. Joint venture between Ivanhoe Mines and Zijin Mining. Mine life: 25 years. Annual production: 450,000 tonnes Cu. Payback: 2.8 years. Mining: Underground. Processing: Conventional flotation.',
    npv: 3250.8,
    irr: 31.7,
    capex: 1850.2,
    aisc: 1.45, // $/lb copper
    resource: '42.4Mt @ 2.74% Cu Indicated',
    reserve: '31.5Mt @ 3.12% Cu Proven + Probable',
    qualified_persons: [
      {
        name: 'Gordon Seibel',
        credentials: 'M.Sc., MAusIMM, Senior Mining Engineer',
        company: 'Ivanhoe Mines'
      },
      {
        name: 'Dr. David Korsch',
        credentials: 'Ph.D., P.Geo., MAIG',
        company: 'Independent Qualified Person'
      },
      {
        name: 'Bernard Peters',
        credentials: 'B.Eng., P.Eng.',
        company: 'Amec Foster Wheeler'
      }
    ]
  },
  {
    name: 'Pilbara Lithium Project',
    location: 'Western Australia',
    latitude: -21.8644,
    longitude: 118.4844,
    stage: 'Feasibility',
    status: 'Active',
    commodities: ['Lithium'],
    description: 'Hard rock spodumene lithium project in the Pilbara region. Pre-production development targeting battery-grade lithium carbonate production. Mine life: 20 years. Annual production: 180,000 tpa LCE. Payback: 3.5 years. Mining: Open pit. Processing: Dense media separation + flotation.',
    npv: 892.5,
    irr: 28.3,
    capex: 680.0,
    aisc: 425, // $/tonne LCE
    resource: '25.8Mt @ 1.3% Li2O Indicated',
    reserve: '18.4Mt @ 1.25% Li2O Probable',
    qualified_persons: [
      {
        name: 'Jennifer Wu',
        credentials: 'B.Sc., FAusIMM, RPEQ',
        company: 'Lycopodium Minerals'
      },
      {
        name: 'Dr. Anthony Eggers',
        credentials: 'Ph.D., MAIG',
        company: 'CSA Global'
      }
    ]
  },
  {
    name: 'Kitimat Molybdenum-Silver Project',
    location: 'British Columbia, Canada',
    latitude: 54.2833,
    longitude: -128.6500,
    stage: 'Pre-Feasibility',
    status: 'Active',
    commodities: ['Molybdenum', 'Silver'],
    description: 'Large-scale porphyry molybdenum deposit with significant silver by-product credits in northwestern BC. Mine life: 22 years. Annual production: 18M lbs Mo + 1.2M oz Ag. Payback: 5.2 years. Mining: Open pit. Processing: Conventional flotation.',
    npv: 245.7,
    irr: 15.8,
    capex: 825.3,
    aisc: 6.50, // $/lb Mo
    resource: '450Mt @ 0.048% Mo, 3.2 g/t Ag M&I',
    reserve: 'Not yet determined',
    qualified_persons: [
      {
        name: 'James McDonald',
        credentials: 'P.Eng., CEM',
        company: 'Tetra Tech'
      },
      {
        name: 'Dr. Robert Morrison',
        credentials: 'Ph.D., P.Geo.',
        company: 'Morrison Geological Services'
      }
    ]
  },
  {
    name: 'Pueblo Viejo Gold-Silver Mine',
    location: 'Dominican Republic',
    latitude: 18.9333,
    longitude: -70.1667,
    stage: 'Production',
    status: 'Active',
    commodities: ['Gold', 'Silver'],
    description: 'High-sulfidation epithermal gold-silver deposit. One of the largest gold mines in Latin America, operated by Barrick Gold. Mine life: 18 years. Annual production: 625,000 oz Au + 1.8M oz Ag. Payback: 3.8 years. Mining: Open pit. Processing: Pressure oxidation + CIL.',
    npv: 1250.4,
    irr: 26.9,
    capex: 920.8,
    aisc: 720, // $/oz Au
    resource: '13.7M oz Au + 98M oz Ag M&I',
    reserve: '9.2M oz Au + 64M oz Ag P&P',
    qualified_persons: [
      {
        name: 'Craig Fiddes',
        credentials: 'P.Eng., SME-RM',
        company: 'Barrick Gold Corporation'
      },
      {
        name: 'Dr. Simon Bottoms',
        credentials: 'Ph.D., CEng, FIMMM',
        company: 'SRK Consulting (UK)'
      },
      {
        name: 'Chad Yuhasz',
        credentials: 'P.Geo., M.Sc.',
        company: 'Barrick Gold Corporation'
      }
    ]
  }
];

async function addTestProjects() {
  console.log('🚀 Adding 5 test projects with full data...\n');

  for (const project of testProjects) {
    try {
      // Check if project already exists
      const { data: existing } = await supabase
        .from('projects')
        .select('id, name')
        .eq('name', project.name)
        .single();

      if (existing) {
        console.log(`⚠️  Project "${project.name}" already exists. Updating...`);

        const { error } = await supabase
          .from('projects')
          .update({
            location: project.location,
            latitude: project.latitude,
            longitude: project.longitude,
            stage: project.stage,
            status: project.status,
            commodities: project.commodities,
            description: project.description,
            npv: project.npv,
            irr: project.irr,
            capex: project.capex,
            aisc: project.aisc,
            resource: project.resource,
            reserve: project.reserve,
            qualified_persons: project.qualified_persons
          })
          .eq('id', existing.id);

        if (error) {
          console.error(`   ❌ Error updating: ${error.message}`);
        } else {
          console.log(`   ✅ Updated successfully`);
          printProjectSummary(project);
        }
      } else {
        const { data, error } = await supabase
          .from('projects')
          .insert({
            name: project.name,
            location: project.location,
            latitude: project.latitude,
            longitude: project.longitude,
            stage: project.stage,
            status: project.status,
            commodities: project.commodities,
            description: project.description,
            npv: project.npv,
            irr: project.irr,
            capex: project.capex,
            aisc: project.aisc,
            resource: project.resource,
            reserve: project.reserve,
            qualified_persons: project.qualified_persons
          })
          .select()
          .single();

        if (error) {
          console.error(`❌ Error creating "${project.name}": ${error.message}`);
        } else {
          console.log(`✅ Created "${project.name}"`);
          printProjectSummary(project);
        }
      }
    } catch (error: any) {
      console.error(`❌ Unexpected error for "${project.name}":`, error.message);
    }
  }

  console.log('\n🎉 Test project creation complete!');
  console.log('\n📊 Summary:');
  console.log(`   - Projects created/updated: ${testProjects.length}`);
  console.log(`   - Commodities covered: Gold, Copper, Lithium, Molybdenum, Silver`);
  console.log(`   - Locations: Nevada, DRC, Australia, Canada, Dominican Republic`);
  console.log(`   - All projects have: NPV, IRR, CAPEX, AISC, Qualified Persons`);

  console.log('\n🧪 Test These Features:');
  console.log('   1. View projects in /global-projects table');
  console.log('   2. Click any project to see AISC and Qualified Persons displayed');
  console.log('   3. Generate AI Insights - should quote exact NPV/IRR/AISC values');
  console.log('   4. Run Sensitivity Analysis - should quote base case and new values');
  console.log('   5. Verify no generic AI responses ("strong economics", "experienced team")');
}

function printProjectSummary(project: any) {
  console.log(`   📍 Location: ${project.location}`);
  console.log(`   💰 NPV: $${project.npv}M | IRR: ${project.irr}% | AISC: $${project.aisc}/unit`);
  console.log(`   👥 Qualified Persons: ${project.qualified_persons.length}`);
  project.qualified_persons.forEach((qp: any) => {
    console.log(`      - ${qp.name}, ${qp.credentials}`);
  });
  console.log('');
}

addTestProjects().catch(console.error);
