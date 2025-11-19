import { createClient } from '@supabase/supabase-js'
import OpenAI from 'openai'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
})

async function generateSummary(title: string, content: string | null): Promise<string> {
  try {
    const textToSummarize = content || title

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        {
          role: 'system',
          content: 'You are a mining industry news analyst. Generate concise, informative summaries of mining news articles. Focus on key facts: company names, project names, commodities, locations, financial figures, and operational updates. Keep summaries to 2-3 sentences.'
        },
        {
          role: 'user',
          content: `Summarize this mining news article:\n\nTitle: ${title}\n\nContent: ${textToSummarize}`
        }
      ],
      temperature: 0.3,
      max_tokens: 150,
    })

    return response.choices[0].message.content?.trim() || 'Summary unavailable'
  } catch (error) {
    console.error('Error generating summary:', error)
    return 'Summary generation failed'
  }
}

async function processNewsWithoutSummaries() {
  console.log('🔍 Finding news items without summaries...\n')

  // Fetch news items without summaries
  const { data: newsItems, error } = await supabase
    .from('news')
    .select('id, title, summary, urls')
    .or('summary.is.null,summary.eq.')
    .order('created_at', { ascending: false })

  if (error) {
    console.error('❌ Error fetching news:', error)
    return
  }

  if (!newsItems || newsItems.length === 0) {
    console.log('✅ All news items already have summaries!')
    return
  }

  console.log(`📰 Found ${newsItems.length} news items without summaries\n`)

  let successCount = 0
  let failCount = 0

  for (let i = 0; i < newsItems.length; i++) {
    const item = newsItems[i]
    console.log(`\n[${i + 1}/${newsItems.length}] Processing: ${item.title?.substring(0, 60)}...`)

    // Generate summary using OpenAI
    const summary = await generateSummary(item.title || 'Untitled', null)

    if (summary && summary !== 'Summary generation failed') {
      // Update the database
      const { error: updateError } = await supabase
        .from('news')
        .update({ summary })
        .eq('id', item.id)

      if (updateError) {
        console.log(`   ❌ Failed to update: ${updateError.message}`)
        failCount++
      } else {
        console.log(`   ✅ Summary generated: ${summary.substring(0, 80)}...`)
        successCount++
      }
    } else {
      console.log(`   ⚠️  Summary generation failed`)
      failCount++
    }

    // Rate limiting: wait 1 second between requests
    if (i < newsItems.length - 1) {
      await new Promise(resolve => setTimeout(resolve, 1000))
    }
  }

  console.log('\n' + '='.repeat(60))
  console.log('📊 SUMMARY GENERATION COMPLETE')
  console.log('='.repeat(60))
  console.log(`✅ Successfully generated: ${successCount}`)
  console.log(`❌ Failed: ${failCount}`)
  console.log(`📰 Total processed: ${newsItems.length}`)
}

processNewsWithoutSummaries()
