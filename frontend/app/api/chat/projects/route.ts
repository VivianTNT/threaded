import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

// Create Supabase client with service role for server-side
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
const supabase = createClient(supabaseUrl, supabaseServiceKey);

export async function GET() {
  try {
    console.log('[Projects API] Starting fetch...');
    console.log('[Projects API] Supabase URL:', supabaseUrl);
    console.log('[Projects API] Service key available:', !!supabaseServiceKey);

    // Fetch all projects - use * to get all available columns
    const { data: projects, error, count } = await supabase
      .from('projects')
      .select('*', { count: 'exact' })
      .order('updated_at', { ascending: false });

    console.log('[Projects API] Query complete. Error:', error);
    console.log('[Projects API] Projects found:', projects?.length || 0);
    console.log('[Projects API] Total count:', count);

    if (error) {
      console.error('[Projects API] Error fetching projects:', error);
      return NextResponse.json({ projects: [], stats: {}, error: error.message });
    }

    // Format projects data for AI consumption
    const formattedProjects = projects?.map(project => ({
      id: project.id,
      name: project.name,
      company_id: project.company_id,
      stage: project.stage || 'N/A',
      location: project.location || 'N/A',
      commodities: Array.isArray(project.commodities) ? project.commodities.join(', ') : 'N/A',
      status: project.status || 'N/A',
      npv: project.npv !== null && project.npv !== undefined ? `$${project.npv}M` : 'N/A',
      irr: project.irr !== null && project.irr !== undefined ? `${project.irr}%` : 'N/A',
      capex: project.capex !== null && project.capex !== undefined ? `$${project.capex}M` : 'N/A',
      description: project.description ? project.description.substring(0, 200) + '...' : 'N/A'
    })) || [];

    // Calculate summary statistics
    const stats = {
      totalProjects: projects?.length || 0,
      byStage: projects?.reduce((acc: any, p) => {
        const stage = p.stage || 'Unknown';
        acc[stage] = (acc[stage] || 0) + 1;
        return acc;
      }, {}),
      byCommodity: projects?.reduce((acc: any, p) => {
        const commodities = p.commodities || [];
        commodities.forEach((commodity: string) => {
          acc[commodity] = (acc[commodity] || 0) + 1;
        });
        return acc;
      }, {}),
      avgIRR: projects?.filter(p => p.irr !== null && p.irr !== undefined)
        .reduce((sum, p) => sum + (p.irr || 0), 0) /
        projects?.filter(p => p.irr !== null && p.irr !== undefined).length || 0,
      totalNPV: projects?.reduce((sum, p) => sum + (p.npv || 0), 0) || 0,
      avgCapex: projects?.filter(p => p.capex !== null && p.capex !== undefined)
        .reduce((sum, p) => sum + (p.capex || 0), 0) /
        projects?.filter(p => p.capex !== null && p.capex !== undefined).length || 0
    };

    return NextResponse.json({ 
      projects: formattedProjects,
      stats,
      lastUpdated: new Date().toISOString()
    });
  } catch (error) {
    console.error('Error in projects API:', error);
    return NextResponse.json({ 
      projects: [],
      stats: {},
      error: 'Failed to fetch projects' 
    });
  }
}