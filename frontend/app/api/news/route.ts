import { NextResponse } from 'next/server';
import { createServerSupabaseClient } from '@/lib/supabase-server';

export async function GET(request: Request) {
  try {
    const supabase = await createServerSupabaseClient();

    // Get query parameters
    const url = new URL(request.url);
    const commodity = url.searchParams.get('commodity');
    const search = url.searchParams.get('search');
    const limit = parseInt(url.searchParams.get('limit') || '100');
    const source = url.searchParams.get('source');
    const category = url.searchParams.get('category');
    const importance = url.searchParams.get('importance');

    // Build query for unified_news table
    let query = supabase
      .from('unified_news')
      .select(`
        id,
        headline,
        summary,
        url,
        published_date,
        source_name,
        source_type,
        symbol,
        company_name,
        exchange,
        topics,
        primary_commodity,
        commodities,
        news_category,
        countries,
        project_names,
        is_mining_related,
        is_project_related,
        is_exploration_news,
        is_production_news,
        mentions_financials,
        mentions_technical_report,
        mentions_resource_estimate,
        mentions_feasibility_study,
        mentions_environmental,
        mentions_permits,
        mentions_acquisition,
        sentiment_score,
        relevance_score,
        importance_level,
        is_featured,
        created_at
      `)
      .eq('is_archived', false)
      .order('published_date', { ascending: false })
      .limit(limit);

    // Apply filters
    if (commodity) {
      query = query.eq('primary_commodity', commodity);
    }

    if (source) {
      query = query.eq('source_name', source);
    }

    if (category) {
      query = query.eq('news_category', category);
    }

    if (importance) {
      query = query.eq('importance_level', importance);
    }

    if (search) {
      // Use full-text search for better results
      query = query.textSearch(
        'headline, summary, company_name',
        search,
        {
          type: 'websearch',
          config: 'english'
        }
      );
    }

    const { data: news, error } = await query;

    if (error) {
      console.error('Error fetching news:', error);
      // Try simpler query if text search fails
      if (search && error.message.includes('textSearch')) {
        let simpleQuery = supabase
          .from('unified_news')
          .select('*')
          .eq('is_archived', false)
          .order('published_date', { ascending: false })
          .limit(limit);
        
        if (search) {
          simpleQuery = simpleQuery.or(
            `headline.ilike.%${search}%,company_name.ilike.%${search}%,summary.ilike.%${search}%`
          );
        }
        
        const { data: simpleNews, error: simpleError } = await simpleQuery;
        
        if (!simpleError) {
          return NextResponse.json({ news: simpleNews || [] });
        }
      }
      
      return NextResponse.json({ error: 'Failed to fetch news' }, { status: 500 });
    }

    // Transform data to match frontend expectations
    const transformedNews = (news || []).map((item: any) => ({
      ...item,
      // Map published_date to datetime for backward compatibility
      datetime: item.published_date,
      // Map url to story_url for backward compatibility  
      story_url: item.url,
      // Ensure news_id exists for compatibility
      news_id: item.id
    }));

    return NextResponse.json({ news: transformedNews });
  } catch (error) {
    console.error('Error in news API:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}