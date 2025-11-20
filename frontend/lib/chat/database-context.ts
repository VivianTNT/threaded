import { createClient } from '@supabase/supabase-js';

export interface ProjectContext {
  globalProjects: any[];
  watchlistedProjects: any[];
  summaryStats: {
    totalProjects: number;
    avgNPV: number;
    avgIRR: number;
    topCommodities: { commodity: string; count: number }[];
    topCountries: { country: string; count: number }[];
  };
  recentNews: any[];
}

/**
 * Fetch comprehensive database context for chat
 */
export async function fetchDatabaseContext(userId?: string): Promise<ProjectContext> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Fetch all global projects
    const { data: globalProjects, error: projectsError } = await supabase
      .from('mining_projects')
      .select('*')
      .order('npv_value', { ascending: false })
      .limit(100);

    if (projectsError) throw projectsError;

    // Fetch watchlisted projects if userId provided
    let watchlistedProjects: any[] = [];
    if (userId) {
      const { data: watchlistData, error: watchlistError } = await supabase
        .from('user_watchlist')
        .select(`
          mining_projects (*)
        `)
        .eq('user_id', userId);

      if (!watchlistError && watchlistData) {
        watchlistedProjects = watchlistData.map(item => item.mining_projects);
      }
    }

    // Fetch recent news
    const { data: recentNews, error: newsError } = await supabase
      .from('unified_news')
      .select('*')
      .order('published_date', { ascending: false })
      .limit(20);

    if (newsError) console.error('Error fetching news:', newsError);

    // Calculate summary statistics
    const projects = globalProjects || [];
    const summaryStats = {
      totalProjects: projects.length,
      avgNPV: projects.reduce((sum, p) => sum + (p.npv_value || 0), 0) / projects.length || 0,
      avgIRR: projects.reduce((sum, p) => sum + (p.irr_value || 0), 0) / projects.length || 0,
      topCommodities: calculateTopItems(projects, 'primary_commodity'),
      topCountries: calculateTopItems(projects, 'country')
    };

    return {
      globalProjects: projects,
      watchlistedProjects,
      summaryStats,
      recentNews: recentNews || []
    };
  } catch (error) {
    console.error('Error fetching database context:', error);
    return {
      globalProjects: [],
      watchlistedProjects: [],
      summaryStats: {
        totalProjects: 0,
        avgNPV: 0,
        avgIRR: 0,
        topCommodities: [],
        topCountries: []
      },
      recentNews: []
    };
  }
}

/**
 * Search for specific projects by query
 */
export async function searchProjectsForContext(query: string): Promise<any[]> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY!;
  const supabase = createClient(supabaseUrl, supabaseServiceKey);

  try {
    const searchTerm = query.toLowerCase();
    
    const { data: projects, error } = await supabase
      .from('mining_projects')
      .select('*')
      .or(`
        project_name.ilike.%${searchTerm}%,
        company.ilike.%${searchTerm}%,
        primary_commodity.ilike.%${searchTerm}%,
        country.ilike.%${searchTerm}%,
        region.ilike.%${searchTerm}%
      `)
      .limit(10);

    if (error) throw error;
    return projects || [];
  } catch (error) {
    console.error('Error searching projects:', error);
    return [];
  }
}

/**
 * Format context for AI consumption
 */
export function formatContextForAI(context: ProjectContext, query?: string): string {
  let formattedContext = `
DATABASE CONTEXT:

SUMMARY STATISTICS:
- Total Projects: ${context.summaryStats.totalProjects}
- Average NPV: $${(context.summaryStats.avgNPV / 1000000).toFixed(1)}M
- Average IRR: ${context.summaryStats.avgIRR.toFixed(1)}%
- Top Commodities: ${context.summaryStats.topCommodities.slice(0, 5).map(c => `${c.commodity} (${c.count})`).join(', ')}
- Top Countries: ${context.summaryStats.topCountries.slice(0, 5).map(c => `${c.country} (${c.count})`).join(', ')}

`;

  // Add watchlisted projects if any
  if (context.watchlistedProjects.length > 0) {
    formattedContext += `\nWATCHLISTED PROJECTS (${context.watchlistedProjects.length}):\n`;
    context.watchlistedProjects.forEach(p => {
      formattedContext += `- ${p.project_name} (${p.company}): ${p.primary_commodity} in ${p.country}, NPV: $${(p.npv_value / 1000000).toFixed(1)}M, IRR: ${p.irr_value}%\n`;
    });
  }

  // Add top global projects
  formattedContext += `\nTOP GLOBAL PROJECTS BY NPV:\n`;
  context.globalProjects.slice(0, 10).forEach((p, i) => {
    formattedContext += `${i + 1}. ${p.project_name} (${p.company}): ${p.primary_commodity} in ${p.country}, NPV: $${(p.npv_value / 1000000).toFixed(1)}M, IRR: ${p.irr_value}%, Stage: ${p.development_stage}\n`;
  });

  // Add recent news if relevant
  if (context.recentNews.length > 0) {
    formattedContext += `\nRECENT NEWS:\n`;
    context.recentNews.slice(0, 5).forEach(n => {
      formattedContext += `- ${n.headline} (${n.source_name}, ${new Date(n.published_date).toLocaleDateString()})\n`;
    });
  }

  return formattedContext;
}

// Helper function to calculate top items
function calculateTopItems(projects: any[], field: string): { [key: string]: number; count: number }[] {
  const counts: { [key: string]: number } = {};
  
  projects.forEach(project => {
    const value = project[field];
    if (value) {
      counts[value] = (counts[value] || 0) + 1;
    }
  });

  return Object.entries(counts)
    .map(([key, count]) => ({ [field]: key, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
}
