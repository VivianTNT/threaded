#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const miningProjects = [
  {
    name: 'Kamoa-Kakula Copper Mine',
    companyName: 'Ivanhoe Mines',
    ticker: 'IVN',
    location: 'Lualaba Province',
    country: 'Democratic Republic of Congo',
    commodities: ['Copper', 'Gold', 'Silver'],
    stage: 'Production',
    latitude: -10.6667,
    longitude: 24.5833,
    resource: 43600000,
    resourceGrade: 2.74,
    reserve: 9800000,
    reserveGrade: 2.58,
    capex: 1400,
    opex: 850,
    aisc: 1200,
    pdfFile: '2a336f44-94b9-4523-8b96-29615154a687_doc-1.pdf'
  },
  {
    name: 'Oyu Tolgoi',
    companyName: 'Rio Tinto',
    ticker: 'RIO',
    location: 'South Gobi',
    country: 'Mongolia',
    commodities: ['Copper', 'Gold', 'Silver'],
    stage: 'Production',
    latitude: 43.0,
    longitude: 106.85,
    resource: 38500000,
    resourceGrade: 0.51,
    reserve: 10200000,
    reserveGrade: 0.57,
    capex: 6750,
    opex: 920,
    aisc: 1850,
    pdfFile: '486c62f2-7e66-4a33-af88-b8b3ed5265f6_doc-1.pdf'
  },
  {
    name: 'Côté Gold Project',
    companyName: 'IAMGOLD Corporation',
    ticker: 'IAG',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Development',
    latitude: 48.1667,
    longitude: -81.5,
    resource: 17500000,
    resourceGrade: 0.87,
    reserve: 7300000,
    reserveGrade: 0.91,
    capex: 1300,
    opex: 680,
    aisc: 950,
    pdfFile: '532f460a-9037-4f4b-a0cb-cbfa7d9872ba_doc-1.pdf'
  },
  {
    name: 'Escondida Mine',
    companyName: 'BHP',
    ticker: 'BHP',
    location: 'Atacama Desert',
    country: 'Chile',
    commodities: ['Copper', 'Gold'],
    stage: 'Production',
    latitude: -24.3333,
    longitude: -69.0833,
    resource: 51200000,
    resourceGrade: 0.62,
    reserve: 14800000,
    reserveGrade: 0.68,
    capex: 8500,
    opex: 1100,
    aisc: 2100,
    pdfFile: 'abe4db60-e088-4dbf-8c28-2dc1d5a585df_doc-1.pdf'
  },
  {
    name: 'Grasberg Mine',
    companyName: 'Freeport-McMoRan',
    ticker: 'FCX',
    location: 'Papua',
    country: 'Indonesia',
    commodities: ['Copper', 'Gold', 'Silver'],
    stage: 'Production',
    latitude: -4.0533,
    longitude: 137.1156,
    resource: 28900000,
    resourceGrade: 0.91,
    reserve: 12300000,
    reserveGrade: 1.02,
    capex: 15000,
    opex: 1250,
    aisc: 2400,
    pdfFile: 'db7f5c08-b5d1-40bf-ab4d-3eb536ffa12c_doc-1.pdf'
  },
  {
    name: 'Chuquicamata Mine',
    companyName: 'Codelco',
    ticker: null,
    location: 'Antofagasta',
    country: 'Chile',
    commodities: ['Copper', 'Molybdenum'],
    stage: 'Production',
    latitude: -22.3,
    longitude: -68.9,
    resource: 44000000,
    resourceGrade: 0.71,
    reserve: 11200000,
    reserveGrade: 0.77,
    capex: 5200,
    opex: 980,
    aisc: 1950,
    pdfFile: 'factset-documents_03ce6d34-4b86-416f-9a43-34c0814c9971.pdf'
  },
  {
    name: 'Carlin Trend',
    companyName: 'Nevada Gold Mines',
    ticker: 'GOLD',
    location: 'Nevada',
    country: 'United States',
    commodities: ['Gold', 'Silver'],
    stage: 'Production',
    latitude: 40.7,
    longitude: -116.3,
    resource: 89000000,
    resourceGrade: 1.12,
    reserve: 31500000,
    reserveGrade: 1.24,
    capex: 2800,
    opex: 720,
    aisc: 1100,
    pdfFile: 'factset-documents_09f16a75-8a0e-4e45-8da4-43d78453a75c.pdf'
  },
  {
    name: 'Cortez Gold Mine',
    companyName: 'Nevada Gold Mines',
    ticker: 'GOLD',
    location: 'Nevada',
    country: 'United States',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 40.3333,
    longitude: -116.6167,
    resource: 21400000,
    resourceGrade: 1.45,
    reserve: 8900000,
    reserveGrade: 1.58,
    capex: 1900,
    opex: 650,
    aisc: 980,
    pdfFile: 'factset-documents_1c915334-2fd3-4d4b-84b7-59cacca1b5fe.pdf'
  },
  {
    name: 'Pueblo Viejo Mine',
    companyName: 'Barrick Gold',
    ticker: 'GOLD',
    location: 'Sánchez Ramírez',
    country: 'Dominican Republic',
    commodities: ['Gold', 'Silver'],
    stage: 'Production',
    latitude: 18.9333,
    longitude: -70.1667,
    resource: 18700000,
    resourceGrade: 2.34,
    reserve: 6200000,
    reserveGrade: 2.51,
    capex: 3800,
    opex: 810,
    aisc: 1250,
    pdfFile: 'factset-documents_20fbfe91-747d-4435-82c8-1e79f4c3a18f.pdf'
  },
  {
    name: 'Detour Lake Mine',
    companyName: 'Agnico Eagle Mines',
    ticker: 'AEM',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 48.5,
    longitude: -80.0,
    resource: 15800000,
    resourceGrade: 0.95,
    reserve: 5400000,
    reserveGrade: 1.02,
    capex: 2200,
    opex: 690,
    aisc: 1050,
    pdfFile: 'factset-documents_2a62d0b9-0efe-439c-a96d-2c0d1df46721.pdf'
  },
  {
    name: 'Malartic Mine',
    companyName: 'Agnico Eagle Mines',
    ticker: 'AEM',
    location: 'Quebec',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 48.1333,
    longitude: -78.1333,
    resource: 19200000,
    resourceGrade: 0.88,
    reserve: 6700000,
    reserveGrade: 0.94,
    capex: 1650,
    opex: 640,
    aisc: 920,
    pdfFile: 'factset-documents_2ac766d9-1899-4e7c-a220-86a1cbe8bd01.pdf'
  },
  {
    name: 'Hemlo Mine',
    companyName: 'Barrick Gold',
    ticker: 'GOLD',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 48.7,
    longitude: -85.9,
    resource: 8900000,
    resourceGrade: 1.21,
    reserve: 2800000,
    reserveGrade: 1.35,
    capex: 950,
    opex: 580,
    aisc: 880,
    pdfFile: 'factset-documents_3a99f37e-e7d9-403c-a4d9-9a8e9267118e.pdf'
  },
  {
    name: 'Musselwhite Mine',
    companyName: 'Newmont Corporation',
    ticker: 'NEM',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 52.6,
    longitude: -90.4,
    resource: 6700000,
    resourceGrade: 6.42,
    reserve: 2100000,
    reserveGrade: 6.85,
    capex: 780,
    opex: 920,
    aisc: 1350,
    pdfFile: 'factset-documents_3e1df113-503d-4cf9-ba30-f9e9f38fd670.pdf'
  },
  {
    name: 'Porcupine Mine',
    companyName: 'Newmont Corporation',
    ticker: 'NEM',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 48.4667,
    longitude: -81.2333,
    resource: 11400000,
    resourceGrade: 1.08,
    reserve: 3800000,
    reserveGrade: 1.15,
    capex: 1350,
    opex: 670,
    aisc: 980,
    pdfFile: 'factset-documents_3e4b1911-32b9-4039-a756-d7011b2d720a.pdf'
  },
  {
    name: 'Red Lake Mine',
    companyName: 'Evolution Mining',
    ticker: 'EVN.AX',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 51.0333,
    longitude: -93.8333,
    resource: 4200000,
    resourceGrade: 8.95,
    reserve: 1300000,
    reserveGrade: 9.42,
    capex: 520,
    opex: 1050,
    aisc: 1520,
    pdfFile: 'factset-documents_54bbc3ff-e716-4cf4-a334-7e9278af2b65.pdf'
  },
  {
    name: 'Macassa Mine',
    companyName: 'Kirkland Lake Gold',
    ticker: 'KL',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 48.15,
    longitude: -80.0333,
    resource: 3800000,
    resourceGrade: 18.5,
    reserve: 1100000,
    reserveGrade: 19.8,
    capex: 380,
    opex: 850,
    aisc: 1180,
    pdfFile: 'factset-documents_552e3ddc-311b-4162-b4cb-2228006e5617.pdf'
  },
  {
    name: 'Young-Davidson Mine',
    companyName: 'Alamos Gold',
    ticker: 'AGI',
    location: 'Ontario',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 48.4,
    longitude: -80.5,
    resource: 8600000,
    resourceGrade: 2.15,
    reserve: 2900000,
    reserveGrade: 2.32,
    capex: 980,
    opex: 710,
    aisc: 1020,
    pdfFile: 'factset-documents_5ffb1d66-9b7d-49f9-a845-d050d1a4d8a9.pdf'
  },
  {
    name: 'Meadowbank Mine',
    companyName: 'Agnico Eagle Mines',
    ticker: 'AEM',
    location: 'Nunavut',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 65.1,
    longitude: -96.0,
    resource: 4100000,
    resourceGrade: 2.45,
    reserve: 1200000,
    reserveGrade: 2.58,
    capex: 650,
    opex: 980,
    aisc: 1420,
    pdfFile: 'factset-documents_65da53e9-a28b-453c-9ab7-9569b6c11049.pdf'
  },
  {
    name: 'Meliadine Mine',
    companyName: 'Agnico Eagle Mines',
    ticker: 'AEM',
    location: 'Nunavut',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Production',
    latitude: 62.9667,
    longitude: -92.3667,
    resource: 6200000,
    resourceGrade: 7.12,
    reserve: 1900000,
    reserveGrade: 7.55,
    capex: 920,
    opex: 1150,
    aisc: 1580,
    pdfFile: 'factset-documents_6d66d1f7-c8a7-4c95-ac94-ae60000a0ee2.pdf'
  },
  {
    name: 'Hope Bay Project',
    companyName: 'TMAC Resources',
    ticker: null,
    location: 'Nunavut',
    country: 'Canada',
    commodities: ['Gold'],
    stage: 'Development',
    latitude: 68.0,
    longitude: -106.5,
    resource: 5900000,
    resourceGrade: 8.45,
    reserve: 1600000,
    reserveGrade: 9.12,
    capex: 780,
    opex: 1280,
    aisc: 1680,
    pdfFile: 'factset-documents_703fef07-9e91-47a9-8e7c-cfa52ec46d49.pdf'
  }
]

async function findOrCreateCompany(name: string, ticker: string | null): Promise<string> {
  // Try to find existing company
  const { data: existing } = await supabase
    .from('companies')
    .select('id')
    .ilike('name', name)
    .maybeSingle()

  if (existing) {
    return existing.id
  }

  // Create new company
  const { data: newCompany, error } = await supabase
    .from('companies')
    .insert({
      name,
      ticker,
      description: `${name} - Mining company`,
      created_at: new Date().toISOString()
    })
    .select('id')
    .single()

  if (error) throw new Error(`Failed to create company: ${error.message}`)
  return newCompany.id
}

async function main() {
  console.log('🚀 Populating diverse mining projects...\n')

  // Clear existing projects
  console.log('🗑️  Clearing existing projects...')
  await supabase.from('projects').delete().neq('id', '00000000-0000-0000-0000-000000000000')
  console.log('✅ Cleared\n')

  let successCount = 0
  let failCount = 0

  for (let i = 0; i < miningProjects.length; i++) {
    const project = miningProjects[i]
    const pdfUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/technical-documents/mining-documents/${project.pdfFile}`

    console.log(`[${i + 1}/${miningProjects.length}] ${project.name}`)

    try {
      // Find or create company
      const companyId = await findOrCreateCompany(project.companyName, project.ticker)
      console.log(`   🏢 Company: ${project.companyName}`)

      // Insert project
      const { error } = await supabase.from('projects').insert({
        name: project.name,
        company_id: companyId,
        location: project.location,
        country: project.country,
        commodities: project.commodities,
        stage: project.stage,
        latitude: project.latitude,
        longitude: project.longitude,
        resource: project.resource,
        resource_grade: project.resourceGrade,
        reserve: project.reserve,
        reserve_grade: project.reserveGrade,
        capex: project.capex,
        opex: project.opex,
        aisc: project.aisc,
        document_storage_path: pdfUrl,
        urls: [pdfUrl],
        created_at: new Date().toISOString()
      })

      if (error) {
        console.error(`   ❌ Failed: ${error.message}`)
        failCount++
      } else {
        console.log(`   ✅ Created`)
        successCount++
      }
    } catch (error: any) {
      console.error(`   ❌ Error: ${error.message}`)
      failCount++
    }
    console.log('')
  }

  console.log('\n📊 Summary:')
  console.log(`   ✅ Successfully created: ${successCount}`)
  console.log(`   ❌ Failed: ${failCount}`)
  console.log('\n✅ Population complete!')
}

main()
