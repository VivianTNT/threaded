#!/usr/bin/env node

import { config } from 'dotenv'
import { createClient } from '@supabase/supabase-js'
import path from 'path'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const PROJECTS = [
  {
    name: 'Vicuña Copper Project',
    company: 'Lundin Mining',
    location: 'Chile',
    stage: 'Exploration',
    commodities: ['Copper', 'Gold'],
    description: 'Large copper-gold project in Chile. The Vicuña Project represents a significant mineral resource discovery.',
    pdfUrl: 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/LUN/fbd61da6226f0d6410e3f20b81a903b7.pdf',
  },
  {
    name: 'Chapada Mine and Saúva Copper-Gold Project',
    company: 'Lundin Mining',
    location: 'Brazil',
    stage: 'Production',
    commodities: ['Copper', 'Gold'],
    description: 'Operating copper-gold mine in Brazil with significant resources and reserves.',
    pdfUrl: 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/LUN/535c9c8a6fc00af672ecc787d2b07cab.pdf',
  },
  {
    name: 'Caserones Copper-Molybdenum Project',
    company: 'Lundin Mining',
    location: 'Chile',
    stage: 'Production',
    commodities: ['Copper', 'Molybdenum'],
    description: 'Large-scale copper-molybdenum mining operation in northern Chile.',
    pdfUrl: 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/LUN/1e5f68faeb29dc0e1e6f9e08b0cab447-filer.pdf',
  },
  {
    name: 'Kansanshi Copper Mine',
    company: 'First Quantum Minerals',
    location: 'Zambia',
    stage: 'Production',
    commodities: ['Copper', 'Gold'],
    description: 'One of Africa\'s largest copper mines with significant gold production.',
    pdfUrl: 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/FM/21aa219db185f147ad46b0554b8352e3-filer.pdf',
  },
  {
    name: 'Cobre Las Cruces Polymetallic Project',
    company: 'First Quantum Minerals',
    location: 'Spain',
    stage: 'Development',
    commodities: ['Copper', 'Zinc', 'Lead'],
    description: 'Polymetallic sulphide project in southern Spain.',
    pdfUrl: 'https://dfxauievbyqwcynwtvib.supabase.co/storage/v1/object/public/factset-documents/FM/660109c487f5ab25a3def9e9c34ac9bf-filer.pdf',
  },
]

async function createProjects() {
  console.log('📋 Creating projects for downloaded FactSet technical reports...\n')

  for (const project of PROJECTS) {
    console.log(`\n📊 ${project.name}`)
    console.log('='.repeat(80))

    // Find or create company
    let companyId: string | null = null

    const { data: existingCompany } = await supabase
      .from('companies')
      .select('id, name')
      .ilike('name', `%${project.company}%`)
      .single()

    if (existingCompany) {
      companyId = existingCompany.id
      console.log(`   ✅ Found company: ${existingCompany.name} (${companyId})`)
    } else {
      const { data: newCompany, error: companyError } = await supabase
        .from('companies')
        .insert({
          name: project.company,
          ticker: project.company === 'Lundin Mining' ? 'LUN-CA' : 'FM-CA',
        })
        .select()
        .single()

      if (companyError) {
        console.log(`   ❌ Error creating company: ${companyError.message}`)
        continue
      }

      companyId = newCompany!.id
      console.log(`   ✅ Created company: ${project.company} (${companyId})`)
    }

    // Check if project already exists
    const { data: existing } = await supabase
      .from('projects')
      .select('id, name, urls')
      .eq('name', project.name)
      .single()

    if (existing) {
      // Update to add PDF URL if not already there
      const urls = existing.urls || []
      if (!urls.includes(project.pdfUrl)) {
        const { error: updateError } = await supabase
          .from('projects')
          .update({
            urls: [...urls, project.pdfUrl],
          })
          .eq('id', existing.id)

        if (updateError) {
          console.log(`   ❌ Error updating project: ${updateError.message}`)
        } else {
          console.log(`   ✅ Updated existing project with PDF URL`)
        }
      } else {
        console.log(`   ℹ️  Project already exists with PDF URL`)
      }
    } else {
      // Create new project
      const { data: newProject, error: projectError } = await supabase
        .from('projects')
        .insert({
          company_id: companyId,
          name: project.name,
          location: project.location,
          stage: project.stage,
          commodities: project.commodities,
          description: project.description,
          urls: [project.pdfUrl],
          status: 'Active',
        })
        .select()
        .single()

      if (projectError) {
        console.log(`   ❌ Error creating project: ${projectError.message}`)
      } else {
        console.log(`   ✅ Created project: ${project.name}`)
        console.log(`      ID: ${newProject!.id}`)
        console.log(`      Location: ${project.location}`)
        console.log(`      Stage: ${project.stage}`)
        console.log(`      Commodities: ${project.commodities.join(', ')}`)
        console.log(`      PDF: ${project.pdfUrl}`)
      }
    }
  }

  console.log('\n\n✅ Done! Projects created/updated.')
  console.log('\nTo view these projects in the frontend:')
  console.log('1. Go to http://localhost:3001/dashboard')
  console.log('2. Click on "Global Projects" or "Project Screener"')
  console.log('3. Search for: Vicuña, Chapada, Caserones, Kansanshi, or Cobre Las Cruces')
  console.log('4. Click on a project to see details and the PDF viewer')
}

createProjects().catch(console.error)
