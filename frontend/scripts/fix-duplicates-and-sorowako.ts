import { createClient } from '@supabase/supabase-js'
import * as dotenv from 'dotenv'
import * as path from 'path'

// Load environment variables
dotenv.config({ path: path.resolve(process.cwd(), '.env.local') })

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!

const supabase = createClient(supabaseUrl, supabaseServiceKey)

async function fixDuplicatesAndSorowako() {
  console.log('🔍 Finding and fixing issues...\n')

  // 1. Find all projects
  const { data: allProjects, error: fetchError } = await supabase
    .from('projects')
    .select('id, name, created_at, company_id')
    .order('created_at', { ascending: true })

  if (fetchError) {
    console.error('Error fetching projects:', fetchError)
    return
  }

  // 2. Find duplicates
  const nameMap = new Map<string, any[]>()

  allProjects?.forEach(project => {
    const existing = nameMap.get(project.name) || []
    existing.push(project)
    nameMap.set(project.name, existing)
  })

  // 3. Delete duplicate entries (keep the first one)
  console.log('🗑️  Removing duplicates...\n')
  let deletedCount = 0

  for (const [name, projects] of nameMap.entries()) {
    if (projects.length > 1) {
      console.log(`Found ${projects.length} copies of "${name}"`)

      // Keep the first one, delete the rest
      const toDelete = projects.slice(1)

      for (const dup of toDelete) {
        const { error: deleteError } = await supabase
          .from('projects')
          .delete()
          .eq('id', dup.id)

        if (deleteError) {
          console.error(`  ✗ Failed to delete duplicate ${dup.id}:`, deleteError)
        } else {
          console.log(`  ✓ Deleted duplicate (id: ${dup.id})`)
          deletedCount++
        }
      }
      console.log()
    }
  }

  console.log(`✅ Removed ${deletedCount} duplicate projects\n`)

  // 4. Fix Sorowako Nickel Mine company link
  console.log('🔧 Fixing Sorowako Nickel Mine company link...\n')

  // Find Vale SA company
  const { data: valeCompany } = await supabase
    .from('companies')
    .select('id, name')
    .ilike('name', '%Vale%')
    .single()

  if (!valeCompany) {
    console.log('⚠️  Vale SA company not found, creating it...')

    const { data: newCompany, error: createError } = await supabase
      .from('companies')
      .insert({
        name: 'Vale SA',
        description: 'Brazilian multinational mining company, one of the largest producers of nickel and iron ore'
      })
      .select()
      .single()

    if (createError) {
      console.error('Failed to create Vale SA:', createError)
      return
    }

    console.log('✓ Created Vale SA company\n')

    // Update Sorowako project
    const { error: updateError } = await supabase
      .from('projects')
      .update({ company_id: newCompany.id })
      .ilike('name', '%Sorowako%')

    if (updateError) {
      console.error('Failed to link Sorowako:', updateError)
    } else {
      console.log('✓ Linked Sorowako Nickel Mine to Vale SA')
    }
  } else {
    console.log(`Found company: ${valeCompany.name} (id: ${valeCompany.id})`)

    // Update Sorowako project
    const { error: updateError } = await supabase
      .from('projects')
      .update({ company_id: valeCompany.id })
      .ilike('name', '%Sorowako%')

    if (updateError) {
      console.error('Failed to link Sorowako:', updateError)
    } else {
      console.log('✓ Linked Sorowako Nickel Mine to Vale SA')
    }
  }

  console.log('\n✅ All fixes completed!')
}

fixDuplicatesAndSorowako()
