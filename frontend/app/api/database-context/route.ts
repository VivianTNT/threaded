import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

export async function GET(request: NextRequest) {
  console.log("Database context API called");
  
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
    
    if (!supabaseUrl || !supabaseServiceKey) {
      console.error('Supabase configuration missing');
      return NextResponse.json({ error: 'Configuration error' }, { status: 500 });
    }
    
    const supabase = createClient(supabaseUrl, supabaseServiceKey);
    
    // Fetch ALL projects without any filtering
    const { data: projects, error: projectsError } = await supabase
      .from('projects')
      .select('*')
      .order('created_at', { ascending: false });

    if (projectsError) {
      console.error('Error fetching projects:', projectsError);
      return NextResponse.json({ error: 'Failed to fetch projects' }, { status: 500 });
    }

    // Fetch companies
    const { data: companies, error: companiesError } = await supabase
      .from('companies')
      .select('*')
      .order('created_at', { ascending: false });

    if (companiesError) {
      console.error('Error fetching companies:', companiesError);
    }

    // Fetch news
    const { data: news, error: newsError } = await supabase
      .from('news')
      .select('*')
      .order('published_at', { ascending: false })
      .limit(200);

    if (newsError) {
      console.error('Error fetching news:', newsError);
    }
    
    // Extract commodities from array for statistics
    const allCommodities: string[] = [];
    projects?.forEach(p => {
      if (p.commodities && Array.isArray(p.commodities)) {
        allCommodities.push(...p.commodities);
      }
    });

    const commodityCounts: { [key: string]: number } = {};
    allCommodities.forEach(c => {
      commodityCounts[c] = (commodityCounts[c] || 0) + 1;
    });

    const topCommodities = Object.entries(commodityCounts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    // Calculate summary statistics (matching dashboard display)
    const summaryStats = {
      totalProjects: projects?.length || 0,
      totalCompanies: companies?.length || 0,
      totalNews: news?.length || 0,
      topCommodities: topCommodities,
      topCountries: getTopItems(projects || [], 'location', 5),
      topStages: getTopItems(projects || [], 'stage', 5),
      topExchanges: getTopItems(companies || [], 'exchange', 5)
    };

    // Format context for easy consumption
    const formattedContext = {
      projects: projects || [],
      companies: companies || [],
      news: news || [],
      stats: summaryStats,
      formattedText: formatForChat(projects || [], companies || [], news || [], summaryStats)
    };

    console.log(`Returning database context: ${projects?.length || 0} projects, ${companies?.length || 0} companies, ${news?.length || 0} news`);
    return NextResponse.json(formattedContext);
    
  } catch (error) {
    console.error('Database context error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

function getTopItems(projects: any[], field: string, limit: number) {
  const counts: { [key: string]: number } = {};
  
  projects.forEach(project => {
    const value = project[field];
    if (value) {
      counts[value] = (counts[value] || 0) + 1;
    }
  });
  
  return Object.entries(counts)
    .map(([key, count]) => ({ name: key, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, limit);
}

function groupBy(items: any[], field: string) {
  const groups: { [key: string]: any[] } = {};
  
  items.forEach(item => {
    const value = item[field] || 'Unknown';
    if (!groups[value]) {
      groups[value] = [];
    }
    groups[value].push(item);
  });
  
  return groups;
}

function formatForChat(projects: any[], companies: any[], news: any[], stats: any): string {
  // Group data by key attributes for better searchability
  const projectsByLocation = groupBy(projects, 'location');
  const projectsByStage = groupBy(projects, 'stage');
  const companiesByCountry = groupBy(companies, 'country');
  const companiesByExchange = groupBy(companies, 'exchange');

  let context = `═══════════════════════════════════════════════════════════════
COMPREHENSIVE DATABASE CONTEXT - MINING INDUSTRY INTELLIGENCE
═══════════════════════════════════════════════════════════════

📊 SUMMARY STATISTICS:
────────────────────────────────────────────────────────────────
• Total Mining Projects: ${stats.totalProjects}
• Total Companies: ${stats.totalCompanies}
• Total News Articles: ${stats.totalNews}

🔬 PROJECT BREAKDOWN:
────────────────────────────────────────────────────────────────
${stats.topStages && stats.topStages.length > 0 ? stats.topStages.map((s: any) => `   • ${s.name}: ${s.count} projects`).join('\n') : '   • No stage data available'}

🌍 TOP COMMODITIES:
${stats.topCommodities && stats.topCommodities.length > 0 ? stats.topCommodities.map((c: any) => `   • ${c.name}: ${c.count} projects`).join('\n') : '   • No commodity data available'}

🗺️  TOP LOCATIONS (Projects):
${stats.topCountries && stats.topCountries.length > 0 ? stats.topCountries.map((c: any) => `   • ${c.name}: ${c.count} projects`).join('\n') : '   • No location data available'}

📈 TOP EXCHANGES (Companies):
${stats.topExchanges && stats.topExchanges.length > 0 ? stats.topExchanges.map((e: any) => `   • ${e.name}: ${e.count} companies`).join('\n') : '   • No exchange data available'}

════════════════════════════════════════════════════════════════
📋 MINING PROJECTS (${projects.length} total)
════════════════════════════════════════════════════════════════
${projects.slice(0, 30).map((p: any, i: number) =>
  `${(i+1).toString().padStart(2, '0')}. ${p.name || 'N/A'}
    Commodities: ${p.commodities?.join(', ') || 'N/A'} | Location: ${p.location || 'N/A'}
    Stage: ${p.stage || 'N/A'} | Status: ${p.status || 'N/A'}
    Ownership: ${p.ownership_percentage ? p.ownership_percentage + '%' : 'N/A'}
    ${p.description ? p.description.substring(0, 150) + '...' : 'No description'}`
).join('\n\n')}

════════════════════════════════════════════════════════════════
🏢 COMPANIES DATABASE (${companies.length} total)
════════════════════════════════════════════════════════════════
${companies.slice(0, 30).map((c: any, i: number) =>
  `${(i+1).toString().padStart(2, '0')}. ${c.name || 'N/A'} (${c.ticker || 'N/A'})
    Exchange: ${c.exchange || 'N/A'} | Country: ${c.country || 'N/A'}
    Market Cap: ${c.market_cap ? '$' + (c.market_cap / 1000000).toFixed(1) + 'M' : 'N/A'}
    ${c.description ? c.description.substring(0, 150) + '...' : 'No description'}`
).join('\n\n')}

════════════════════════════════════════════════════════════════
📰 RECENT NEWS & ANNOUNCEMENTS (Latest ${Math.min(news.length, 30)})
════════════════════════════════════════════════════════════════
${news.slice(0, 30).map((n: any, i: number) =>
  `${(i+1).toString().padStart(2, '0')}. ${n.title || 'N/A'}
    Source: ${n.source || 'Unknown'} | Date: ${n.published_at ? new Date(n.published_at).toLocaleDateString() : 'N/A'}
    Commodities: ${n.commodities?.join(', ') || 'N/A'} | Sentiment: ${n.sentiment || 'N/A'}
    Summary: ${n.summary?.substring(0, 150) || 'No summary'}...`
).join('\n\n') || 'No news available'}

════════════════════════════════════════════════════════════════
🔍 COMPLETE PROJECT CATALOG (All ${projects.length} Projects)
════════════════════════════════════════════════════════════════
${projects.map((p: any) =>
  `• ${p.name || 'N/A'} | Commodities: ${p.commodities?.join(', ') || 'N/A'} | Location: ${p.location || 'Unknown'} | Stage: ${p.stage || 'N/A'} | Status: ${p.status || 'N/A'} | Ownership: ${p.ownership_percentage ? p.ownership_percentage + '%' : 'N/A'} | Resources: ${p.resource_estimate || 'N/A'} | Reserves: ${p.reserve_estimate || 'N/A'}`
).join('\n')}

════════════════════════════════════════════════════════════════
🔍 COMPLETE COMPANIES CATALOG (All ${companies.length} Companies)
════════════════════════════════════════════════════════════════
${companies.map((c: any) =>
  `• ${c.name || 'N/A'} (${c.ticker || 'N/A'}) | Exchange: ${c.exchange || 'N/A'} | Country: ${c.country || 'N/A'} | Market Cap: ${c.market_cap ? '$' + (c.market_cap / 1000000).toFixed(1) + 'M' : 'N/A'} | Sector: ${c.sector || 'N/A'} | Website: ${c.website || 'N/A'}`
).join('\n')}

═══════════════════════════════════════════════════════════════
END OF DATABASE CONTEXT
═══════════════════════════════════════════════════════════════

INSTRUCTIONS FOR USING THIS DATA:
• You have complete access to ${stats.totalProjects} mining projects, ${stats.totalCompanies} companies, and ${stats.totalNews} news articles
• When asked about specific projects, companies, or trends, search through the complete catalogs above
• Compare projects by their financial metrics (NPV, IRR, CAPEX), commodity types, countries, or development stages
• Provide data-driven insights using exact figures from the database
• Reference recent news when discussing market trends or specific companies
• Always cite specific projects, companies, or news articles when making claims`;

  return context;
}
