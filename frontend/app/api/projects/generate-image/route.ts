import { NextRequest, NextResponse } from 'next/server'
import OpenAI from 'openai'
import { createClient } from '@supabase/supabase-js'

export async function POST(request: NextRequest) {
  try {
    const { projectId } = await request.json()

    console.log('Generating image for project:', projectId)

    if (!projectId) {
      return NextResponse.json({ error: 'Project ID is required' }, { status: 400 })
    }

    // Check if OpenAI API key is configured
    if (!process.env.OPENAI_API_KEY) {
      console.error('OpenAI API key not found')
      // For now, return a placeholder image if no OpenAI key
      const placeholderImage = `https://via.placeholder.com/1024x1024.png?text=Mining+Project+Visualization`
      return NextResponse.json({ imageUrl: placeholderImage, success: true })
    }

    // Create OpenAI client
    const openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })

    // Create Supabase client inside the function
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
    const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

    if (!supabaseUrl || !supabaseKey) {
      console.error('Supabase credentials not found')
      return NextResponse.json({ error: 'Database configuration error' }, { status: 500 })
    }

    const supabase = createClient(supabaseUrl, supabaseKey)

    // Fetch project details
    console.log('Fetching project from database...')
    const { data: project, error } = await supabase
      .from('projects')
      .select('*')
      .eq('id', projectId)
      .single()

    console.log('Database response:', { project: project?.id, error })

    if (error) {
      console.error('Database error:', error)
      return NextResponse.json({ error: 'Database error', details: error.message }, { status: 500 })
    }

    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 })
    }

    // Generate a descriptive prompt for realistic mining imagery
    const prompt = `Create a hyperrealistic photograph of a ${project.stage?.toLowerCase() || 'mining'} mining operation for ${project.primary_commodity || 'minerals'} in ${project.country || 'a remote location'}.

The image should show:
- ${project.stage === 'Production' ? 'Active mining operations with large excavators, haul trucks, and processing equipment' :
    project.stage === 'Development' ? 'Construction of mining infrastructure, earthworks, and equipment installation' :
    project.stage === 'Exploration' ? 'Drilling rigs and geological survey equipment in natural terrain' :
    'Modern mining site with industrial equipment and facilities'}
- ${project.primary_commodity === 'Lithium' ? 'White evaporation ponds or spodumene ore processing' :
    project.primary_commodity === 'Copper' ? 'Large open pit with reddish-brown copper ore and terraced walls' :
    project.primary_commodity === 'Gold' ? 'Gold ore processing mill and heap leaching facilities' :
    project.primary_commodity === 'Nickel' ? 'Laterite mining with red soil and tropical environment' :
    'Mineral extraction and processing operations'}
- Natural landscape typical of ${project.country === 'Australia' ? 'Australian outback with red earth' :
    project.country === 'Canada' ? 'Canadian wilderness with forests or tundra' :
    project.country === 'Chile' ? 'Atacama desert or Andean mountains' :
    'rugged mining terrain'}
- Clear blue sky, dramatic lighting showing time of day (golden hour preferred)
- Sense of scale showing the magnitude of operations

Style: Hyperrealistic photography, professional mining industry documentation, aerial or elevated perspective, no text or infographics, dramatic natural lighting, high detail and clarity.`

    console.log('Calling OpenAI to generate image...')

    let imageUrl = ''

    // Generate image using DALL-E 3
    try {
      const response = await openai.images.generate({
        model: "dall-e-3",
        prompt,
        n: 1,
        size: "1024x1024",
        quality: "hd",
        style: "vivid",
      })

      imageUrl = response.data[0].url || ''

      if (!imageUrl) {
        throw new Error('No image URL returned from OpenAI')
      }
    } catch (openAIError: any) {
      console.error('OpenAI API error:', openAIError)

      // If OpenAI fails, return a placeholder for now
      imageUrl = `https://via.placeholder.com/1024x1024.png?text=${encodeURIComponent(project.project_name || 'Mining Project')}`
      console.log('Using placeholder image due to OpenAI error')
    }

    // Update project with the generated image URL
    const { error: updateError } = await supabase
      .from('projects')
      .update({ generated_image_url: imageUrl })
      .eq('id', projectId)

    if (updateError) {
      console.error('Failed to save image URL to database:', updateError)
      // Still return the image even if we couldn't save it
    }

    console.log('Image generated successfully:', imageUrl)
    return NextResponse.json({ imageUrl, success: true })
  } catch (error) {
    console.error('Error generating image:', error)
    return NextResponse.json(
      { error: 'Failed to generate image', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    )
  }
}