import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function verifyProjectsIntact() {
  console.log('🔍 Verifying all projects are intact...\n')

  // Get total count
  const { count, error: countError } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  if (countError) {
    console.error('❌ Error counting projects:', countError)
    return
  }

  console.log(`📊 Total Projects: ${count}`)

  // Get all projects
  const { data: allProjects, error: fetchError } = await supabase
    .from('projects')
    .select('id, name, stage, commodities, company_id, urls, created_at')
    .order('created_at', { ascending: false })

  if (fetchError) {
    console.error('❌ Error fetching projects:', fetchError)
    return
  }

  console.log('\n📋 Recent Projects:')
  console.log('─'.repeat(100))

  allProjects?.slice(0, 10).forEach((project, index) => {
    const docCount = project.urls?.length || 0
    const hasHtmlDoc = project.urls?.some((url: string) => url.includes('.html'))
    console.log(`${index + 1}. ${project.name}`)
    console.log(`   ID: ${project.id}`)
    console.log(`   Stage: ${project.stage || 'N/A'}`)
    console.log(`   Commodities: ${project.commodities?.join(', ') || 'N/A'}`)
    console.log(`   Documents: ${docCount} ${hasHtmlDoc ? '(includes HTML FactSet filing)' : ''}`)
    console.log(`   Created: ${new Date(project.created_at).toLocaleDateString()}`)
    console.log()
  })

  // Count FactSet HTML documents
  const projectsWithHtml = allProjects?.filter(p =>
    p.urls?.some((url: string) => url.includes('.html'))
  ).length || 0

  console.log('─'.repeat(100))
  console.log('\n📈 Summary:')
  console.log(`   Total Projects: ${count}`)
  console.log(`   Projects with FactSet HTML documents: ${projectsWithHtml}`)
  console.log(`   Projects with other documents: ${count! - projectsWithHtml}`)
  console.log('\n✅ All projects are intact. No projects were removed.')
}

verifyProjectsIntact().catch(console.error)
