import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function deleteSpecificDuplicates() {
  console.log('🗑️  Deleting specific duplicate IDs...\n')

  const duplicateIds = [
    'f89de4e9-fedd-42c4-9487-794ba34df3b3',  // Sorowako duplicate
  ]

  for (const id of duplicateIds) {
    const { error } = await supabase
      .from('projects')
      .delete()
      .eq('id', id)

    if (error) {
      console.error(`✗ Failed to delete ${id}:`, error)
    } else {
      console.log(`✓ Deleted ${id}`)
    }
  }

  // Now find all remaining duplicates by name
  const { data: allProjects } = await supabase
    .from('projects')
    .select('id, name, created_at')
    .order('created_at', { ascending: true })

  if (!allProjects) return

  const nameMap = new Map<string, any[]>()
  allProjects.forEach(project => {
    const existing = nameMap.get(project.name) || []
    existing.push(project)
    nameMap.set(project.name, existing)
  })

  console.log('\n🔍 Finding any remaining duplicates...\n')

  for (const [name, projects] of nameMap.entries()) {
    if (projects.length > 1) {
      console.log(`Found ${projects.length} copies of "${name}"`)
      // Delete all but the first one
      const toDelete = projects.slice(1)
      for (const dup of toDelete) {
        const { error } = await supabase
          .from('projects')
          .delete()
          .eq('id', dup.id)

        if (error) {
          console.error(`  ✗ Failed to delete duplicate:`, error)
        } else {
          console.log(`  ✓ Deleted duplicate (id: ${dup.id})`)
        }
      }
    }
  }

  console.log('\n✅ All duplicates removed!')
}

deleteSpecificDuplicates()
