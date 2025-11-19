#!/usr/bin/env npx tsx

import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function checkDuplicates() {
  const { data: projects } = await supabase
    .from('projects')
    .select('id, name, created_at')
    .order('name')

  if (!projects) {
    console.log('No projects found')
    return
  }

  // Group by name
  const grouped = new Map<string, any[]>()
  for (const project of projects) {
    if (!grouped.has(project.name)) {
      grouped.set(project.name, [])
    }
    grouped.get(project.name)!.push(project)
  }

  // Find duplicates
  console.log('🔍 Duplicate Projects:\n')
  let totalDuplicates = 0

  for (const [name, instances] of grouped.entries()) {
    if (instances.length > 1) {
      console.log(`📊 "${name}" - ${instances.length} instances`)
      totalDuplicates += instances.length - 1

      // Keep newest, delete others
      const sorted = instances.sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )

      const toDelete = sorted.slice(1)

      for (const dup of toDelete) {
        console.log(`   ❌ Deleting duplicate: ${dup.id}`)
        await supabase.from('projects').delete().eq('id', dup.id)
      }
    }
  }

  console.log(`\n✅ Removed ${totalDuplicates} duplicates`)
}

checkDuplicates().catch(console.error)
