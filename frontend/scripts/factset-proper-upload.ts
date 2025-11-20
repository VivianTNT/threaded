#!/usr/bin/env npx tsx
import { createClient } from '@supabase/supabase-js'
import { config } from 'dotenv'
import path from 'path'
import crypto from 'crypto'
import fs from 'fs'
import https from 'https'
import http from 'http'
import { URL } from 'url'

config({ path: path.join(__dirname, '..', '.env.local') })

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const authHeader = 'Basic ' + Buffer.from(`${process.env.FACTSET_USERNAME}:${process.env.FACTSET_API_KEY}`).toString('base64')

async function downloadFile(url: string, filepath: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url)
    const client = parsedUrl.protocol === 'https:' ? https : http

    const file = fs.createWriteStream(filepath)

    const request = client.get(url, {
      headers: {
        'Authorization': authHeader,
        'Accept': 'application/pdf'
      }
    }, (response) => {
      if (response.statusCode !== 200) {
        reject(new Error(`Failed to download: ${response.statusCode}`))
        return
      }

      response.pipe(file)

      file.on('finish', () => {
        file.close()
        resolve()
      })
    })

    request.on('error', (err) => {
      fs.unlink(filepath, () => {})
      reject(err)
    })

    file.on('error', (err) => {
      fs.unlink(filepath, () => {})
      reject(err)
    })
  })
}

async function main() {
  console.log('🎯 FACTSET PROPER UPLOAD DEMO')
  console.log('='.repeat(80))

  // Use American Lithium which we know has documents
  const ticker = 'LI-CA'
  const companyName = 'American Lithium Corp.'

  console.log(`\n📊 Searching ${companyName} (${ticker})...`)

  // Search FactSet
  const searchUrl = `https://api.factset.com/content/global-filings/v2/search?ids=${ticker}&sources=SDR&searchText=technical report ni 43-101&sort=-filingsDate&paginationLimit=5`

  const searchRes = await fetch(searchUrl, { headers: { 'Authorization': authHeader } })
  const searchData = await searchRes.json()

  const docs = searchData.data?.[0]?.documents || []
  console.log(`   Found ${docs.length} documents`)

  if (docs.length === 0) {
    console.log('   ❌ No documents found')
    return
  }

  const results: Array<{ projectName: string, storageUrl: string }> = []

  // Process each document
  for (const doc of docs.slice(0, 5)) {
    const headline = doc.headline.toLowerCase()

    // Skip non-technical reports
    if (headline.includes('consent') || headline.includes('certificate')) {
      continue
    }

    if (!headline.includes('technical report') && !headline.includes('43-101')) {
      continue
    }

    console.log(`\n📄 ${doc.headline.substring(0, 100)}...`)
    console.log(`   Date: ${doc.filingsDate}`)
    console.log(`   Link: ${doc.filingsLink}`)

    try {
      // Download to temp file
      const tempPath = `/tmp/factset_${crypto.randomUUID()}.pdf`
      console.log(`   ⬇️  Downloading...`)

      await downloadFile(doc.filingsLink, tempPath)

      // Check size
      const stats = fs.statSync(tempPath)
      const sizeMB = stats.size / 1024 / 1024
      const estimatedPages = Math.floor((stats.size / 1024) / 30)

      console.log(`   📦 Downloaded: ${sizeMB.toFixed(2)} MB (~${estimatedPages} pages)`)

      if (sizeMB < 3) {
        console.log(`   ⚠️  Too small (need 100+ pages), skipping`)
        fs.unlinkSync(tempPath)
        continue
      }

      // Upload to Supabase - read file as buffer
      const hash = crypto.createHash('md5').update(doc.documentId).digest('hex')
      const storagePath = `${ticker}/${hash}.pdf`

      console.log(`   📤 Uploading to Supabase storage: ${storagePath}`)

      const fileBuffer = fs.readFileSync(tempPath)

      // Try upload with explicit timeout
      const { data: uploadData, error: uploadError } = await supabase.storage
        .from('factset-documents')
        .upload(storagePath, fileBuffer, {
          contentType: 'application/pdf',
          cacheControl: '3600',
          upsert: true
        })

      if (uploadError) {
        console.log(`   ❌ Upload error: ${uploadError.message}`)
        fs.unlinkSync(tempPath)
        continue
      }

      // Get public URL
      const { data: urlData } = supabase.storage
        .from('factset-documents')
        .getPublicUrl(storagePath)

      console.log(`   ✅ UPLOADED SUCCESSFULLY!`)
      console.log(`   🔗 Supabase URL: ${urlData.publicUrl}`)

      // Extract project name
      const projectName = doc.headline
        .replace(/technical report/gi, '')
        .replace(/ni 43-101/gi, '')
        .replace(/\(.*?\)/g, '')
        .replace(companyName, '')
        .replace(/-/g, '')
        .trim()
        .substring(0, 100) || `${companyName} Project`

      results.push({
        projectName: projectName.trim(),
        storageUrl: urlData.publicUrl
      })

      // Clean up
      fs.unlinkSync(tempPath)

      // Stop after 2 successful uploads
      if (results.length >= 2) {
        break
      }

    } catch (error: any) {
      console.log(`   ❌ Error: ${error.message}`)
    }
  }

  // Summary
  console.log('\n\n' + '='.repeat(80))
  console.log('✅ EXTRACTION COMPLETE')
  console.log('='.repeat(80))
  console.log(`\nSuccessfully extracted ${results.length} project(s):\n`)

  results.forEach((r, i) => {
    console.log(`${i + 1}. Project: ${r.projectName}`)
    console.log(`   Supabase Storage URL: ${r.storageUrl}`)
    console.log('')
  })
}

main().catch(console.error)
