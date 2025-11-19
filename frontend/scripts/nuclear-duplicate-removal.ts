import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function nuclearDuplicateRemoval() {
  console.log('💥 NUCLEAR DUPLICATE REMOVAL - This will keep only ONE of each project name\n')

  // Get ALL projects
  const { data: allProjects } = await supabase
    .from('projects')
    .select('*')
    .order('created_at', { ascending: true })

  if (!allProjects) {
    console.log('No projects found')
    return
  }

  console.log(`Total projects before: ${allProjects.length}\n`)

  // Group by name
  const uniqueProjects = new Map<string, any>()
  const duplicatesToDelete: string[] = []

  for (const project of allProjects) {
    if (!uniqueProjects.has(project.name)) {
      // Keep the first one
      uniqueProjects.set(project.name, project)
    } else {
      // Mark this one for deletion
      duplicatesToDelete.push(project.id)
    }
  }

  console.log(`Unique project names: ${uniqueProjects.size}`)
  console.log(`Duplicates to delete: ${duplicatesToDelete.length}\n`)

  // Delete ALL duplicates in batches
  const batchSize = 100
  for (let i = 0; i < duplicatesToDelete.length; i += batchSize) {
    const batch = duplicatesToDelete.slice(i, i + batchSize)
    const { error } = await supabase
      .from('projects')
      .delete()
      .in('id', batch)

    if (error) {
      console.error(`Error deleting batch ${i}-${i + batchSize}:`, error)
    } else {
      console.log(`✓ Deleted batch ${i + 1}-${Math.min(i + batchSize, duplicatesToDelete.length)} of ${duplicatesToDelete.length}`)
    }
  }

  // Verify final count
  const { count } = await supabase
    .from('projects')
    .select('*', { count: 'exact', head: true })

  console.log(`\n✅ Done! Total unique projects remaining: ${count}`)

  // Show top 10
  const { data: topProjects } = await supabase
    .from('projects')
    .select('name, companies(name)')
    .order('updated_at', { ascending: false })
    .limit(10)

  console.log('\nTop 10 projects:\n')
  topProjects?.forEach((p, i) => {
    console.log(`${i + 1}. ${p.name} - ${(p as any).companies?.name || 'No company'}`)
  })
}

nuclearDuplicateRemoval()
