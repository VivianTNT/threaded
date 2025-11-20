#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

async function main() {
  console.log('🔍 Checking projects with document_storage_path or URLs...\n')

  const { data: projects, error } = await supabase
    .from('projects')
    .select('id, name, document_storage_path, urls')
    .or('document_storage_path.not.is.null,urls.not.is.null')
    .limit(10)

  if (error) {
    console.error('Error fetching projects:', error)
    return
  }

  if (!projects || projects.length === 0) {
    console.log('❌ No projects found with document_storage_path or URLs')
    return
  }

  console.log(`Found ${projects.length} projects with documents:\n`)

  projects.forEach((project, index) => {
    console.log(`${index + 1}. ${project.name}`)
    console.log(`   ID: ${project.id}`)

    if (project.document_storage_path) {
      console.log(`   📁 Storage Path: ${project.document_storage_path}`)

      // Show what the full URL would be
      if (project.document_storage_path.startsWith('http')) {
        console.log(`   🔗 Full URL: ${project.document_storage_path}`)
      } else {
        const fullUrl = `${process.env.NEXT_PUBLIC_SUPABASE_URL}/storage/v1/object/public/${project.document_storage_path}`
        console.log(`   🔗 Full URL: ${fullUrl}`)
      }
    }

    if (project.urls && project.urls.length > 0) {
      console.log(`   📄 URLs: ${project.urls.length} document(s)`)
      project.urls.forEach((url: string, i: number) => {
        console.log(`      - URL ${i + 1}: ${url.substring(0, 100)}${url.length > 100 ? '...' : ''}`)
      })
    }

    console.log('')
  })
}

main()