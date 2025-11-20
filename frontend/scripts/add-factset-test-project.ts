import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function addFactSetTestProject() {
  console.log('🔍 Adding FactSet HTML test project...\n')

  // First, check if we have any existing Freeport-McMoRan projects
  const { data: existingProjects, error: fetchError } = await supabase
    .from('projects')
    .select('*')
    .ilike('name', '%freeport%')
    .limit(5)

  if (fetchError) {
    console.error('❌ Error fetching existing projects:', fetchError)
    return
  }

  console.log(`📊 Found ${existingProjects?.length || 0} existing Freeport projects`)

  // Get or create company
  let companyId: string | null = null

  const { data: existingCompany } = await supabase
    .from('companies')
    .select('id')
    .eq('name', 'Freeport-McMoRan Inc.')
    .single()

  if (existingCompany) {
    companyId = existingCompany.id
    console.log('✅ Using existing company:', companyId)
  } else {
    const { data: newCompany, error: companyError } = await supabase
      .from('companies')
      .insert({
        name: 'Freeport-McMoRan Inc.',
        ticker: 'FCX',
        description: 'Leading international mining company with headquarters in Phoenix, Arizona.',
        country: 'United States'
      })
      .select()
      .single()

    if (companyError) {
      console.error('❌ Error creating company:', companyError)
      return
    }

    companyId = newCompany.id
    console.log('✅ Created new company:', companyId)
  }

  // Create test project with FactSet HTML document
  const testProject = {
    name: 'Freeport-McMoRan Copper & Gold - Q3 2025 10-Q',
    company_id: companyId,
    location: 'Arizona, United States',
    stage: 'Production',
    commodities: ['Copper', 'Gold', 'Molybdenum'],
    status: 'Active',
    description: 'Freeport-McMoRan Inc. operates large, long-lived geographically diverse assets with significant proven and probable reserves of copper, gold and molybdenum. This project represents their Q3 2025 quarterly report with comprehensive operational and financial data.',
    urls: ['https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/refinitiv/factset/edg/freeport-mcmoran/2025/0000831259-25-000031-1.html'],
    npv: 15000, // Example values
    irr: 22.5,
    capex: 4500,
    resource: '24.2 billion pounds of copper, 11.5 million ounces of gold',
    reserve: '118 billion pounds of copper, 34 million ounces of gold, 3.2 billion pounds of molybdenum',
    watchlist: false,
    is_private: false
  }

  const { data: newProject, error: insertError } = await supabase
    .from('projects')
    .insert(testProject)
    .select()
    .single()

  if (insertError) {
    console.error('❌ Error creating project:', insertError)
    return
  }

  console.log('\n✅ Successfully created test project!')
  console.log('📋 Project Details:')
  console.log('   ID:', newProject.id)
  console.log('   Name:', newProject.name)
  console.log('   Company ID:', newProject.company_id)
  console.log('   Document URL:', newProject.urls[0])
  console.log('   Stage:', newProject.stage)
  console.log('   Commodities:', newProject.commodities.join(', '))
  console.log('\n🎯 Test this by:')
  console.log('   1. Run: npm run dev')
  console.log('   2. Navigate to the dashboard')
  console.log('   3. Find "Freeport-McMoRan Copper & Gold - Q3 2025 10-Q"')
  console.log('   4. Click on it to open the project details')
  console.log('   5. Click on the technical document link to view the HTML document')
  console.log('\n📄 The HTML viewer should display the FactSet SEC filing in an iframe with controls.')
}

addFactSetTestProject().catch(console.error)
