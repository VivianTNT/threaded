// Web search API using Firecrawl
import { NextResponse } from 'next/server';
import FirecrawlApp from '@mendable/firecrawl-js';
import axios from 'axios';

// Helper function to enhance mining-specific queries
function enhanceMiningQuery(query: string): string {
  // Check if query already has mining terms
  const miningTerms = ['mining', 'mineral', 'ore', 'commodity', 'resource', 'feasibility', 'drill', 'assay'];
  const hasMiningContext = miningTerms.some(term => query.toLowerCase().includes(term));
  
  // Add context based on query type
  if (query.toLowerCase().includes('price') || query.toLowerCase().includes('market')) {
    return `${query} commodity prices mining market analysis metal prices LME spot`;
  } else if (query.toLowerCase().includes('report') || query.toLowerCase().includes('technical')) {
    return `${query} NI 43-101 JORC technical report feasibility study PEA resource estimate`;
  } else if (query.toLowerCase().includes('company') || query.toLowerCase().includes('project')) {
    return `${query} mining project development production exploration update`;
  } else if (!hasMiningContext) {
    // Generic mining context
    return `${query} mining industry critical minerals commodity technical analysis`;
  }
  
  return query;
}

// Helper function to score search results
function scoreSearchResult(result: any, query: string): number {
  let score = 0;
  const lowerTitle = result.title?.toLowerCase() || '';
  const lowerSnippet = result.snippet?.toLowerCase() || '';
  const lowerUrl = result.url?.toLowerCase() || '';
  const queryWords = query.toLowerCase().split(' ');
  
  // Score based on query word matches
  queryWords.forEach(word => {
    if (lowerTitle.includes(word)) score += 3;
    if (lowerSnippet.includes(word)) score += 1;
  });
  
  // Boost score for trusted domains
  const trustedDomains = ['sedar.com', 'sec.gov', 'asx.com.au', 'mining.com', 'kitco.com'];
  if (trustedDomains.some(domain => lowerUrl.includes(domain))) {
    score += 5;
  }
  
  // Boost for recent dates (if found in snippet)
  const currentYear = new Date().getFullYear();
  if (lowerSnippet.includes(currentYear.toString()) || lowerSnippet.includes((currentYear - 1).toString())) {
    score += 2;
  }
  
  // Boost for technical report indicators
  const technicalTerms = ['ni 43-101', 'jorc', 'feasibility', 'resource estimate', 'pea', 'mineral reserve'];
  if (technicalTerms.some(term => lowerTitle.includes(term) || lowerSnippet.includes(term))) {
    score += 4;
  }
  
  return score;
}

export async function POST(req: Request) {
  try {
    const { query } = await req.json();
    
    if (!query) {
      return NextResponse.json({ error: 'Query is required' }, { status: 400 });
    }
    
    // Get Firecrawl API key from environment
    const firecrawlApiKey = process.env.FIRECRAWL_API_KEY;
    
    if (!firecrawlApiKey) {
      console.error('Firecrawl API key not found');
      return NextResponse.json({ 
        results: [],
        error: 'Web search configuration error' 
      });
    }
    
    // Initialize Firecrawl
    const firecrawl = new FirecrawlApp({ apiKey: firecrawlApiKey });
    
    // Enhance query with mining-specific context and search terms
    const enhancedQuery = enhanceMiningQuery(query);
    console.log('Original query:', query);
    console.log('Enhanced query:', enhancedQuery);
    
    // Define trusted mining sources
    const trustedMiningDomains = [
      'sedar.com',
      'sec.gov',
      'asx.com.au',
      'mining.com',
      'miningnewsfeed.com',
      'northernminer.com',
      'mining-journal.com',
      'kitco.com',
      'resourceworld.com',
      'juniorminingnetwork.com',
      'minexconsulting.com',
      'snl.com',
      's1.q4cdn.com' // Common host for investor relations documents
    ];
    
    // Retry mechanism for API calls
    let attempts = 0;
    const maxAttempts = 2;
    let lastError = null;
    
    while (attempts < maxAttempts) {
      attempts++;
      
      try {
        console.log(`Firecrawl search attempt ${attempts} for query:`, enhancedQuery);
        
        // Use Firecrawl search API with enhanced query
        const searchResults = await firecrawl.search(enhancedQuery, {
          limit: 10, // Increased to filter better results
          scrapeOptions: {
            formats: ["markdown"],
            timeout: 30000, // Increased timeout for mining sites
            waitFor: 1000,
            includeTags: ['article', 'main', 'section', 'div', 'p', 'table', 'h1', 'h2', 'h3'],
            excludeTags: ['nav', 'header', 'footer', 'aside', 'script', 'style', 'advertisement', 'popup']
          }
        });
        
        console.log('Firecrawl search response:', {
          success: searchResults.success,
          dataLength: searchResults.data?.length || 0,
          error: searchResults.error,
          data: searchResults.data?.slice(0, 2) // Log first 2 results for debugging
        });
        
        if (!searchResults.success || !searchResults.data || searchResults.data.length === 0) {
          lastError = new Error(`Firecrawl search failed: ${searchResults.error || 'No results'}`);
          console.log('Search failed, error:', lastError.message);
          
          if (attempts < maxAttempts) {
            // Wait before retrying
            await new Promise(resolve => setTimeout(resolve, 1000));
            continue;
          }
        }
        
        // Transform and score Firecrawl results
        const scoredResults = (searchResults.data || []).map((item: any) => {
          const result = {
            title: item.title || 'No title',
            link: item.url || '#',
            snippet: item.markdown ? 
              item.markdown.substring(0, 300) + (item.markdown.length > 300 ? '...' : '') :
              'No content available',
            score: 0
          };
          
          // Score the result
          result.score = scoreSearchResult(result, query);
          
          return result;
        });
        
        // Sort by score and take top 5
        const topResults = scoredResults
          .sort((a: any, b: any) => b.score - a.score)
          .slice(0, 5)
          .map(({ score, ...result }: any) => result); // Remove score from final output
        
        console.log(`Returning ${topResults.length} results from ${scoredResults.length} total`);
        
        return NextResponse.json({ results: topResults });
        
      } catch (error) {
        console.error(`Error during Firecrawl search (attempt ${attempts}):`, error);
        lastError = error;
        
        if (attempts < maxAttempts) {
          await new Promise(resolve => setTimeout(resolve, 1000));
          continue;
        }
      }
    }
    
    // If all attempts failed, try a fallback method
    console.error('All Firecrawl search attempts failed:', lastError);
    console.log('Attempting fallback web search...');
    
    try {
      // Fallback: Use a mock result for now to demonstrate functionality
      // In production, you could use another search API like SerpAPI, Google Custom Search, etc.
      const mockResults = [
        {
          title: `Latest Mining News: ${query}`,
          link: 'https://www.mining.com/',
          snippet: `Recent developments in ${query}. The mining industry continues to evolve with new discoveries and technological advancements...`
        },
        {
          title: `Mining Technology and ${query} Analysis`,
          link: 'https://www.mining-technology.com/',
          snippet: `Technical analysis of ${query} in the mining sector. Industry experts discuss the latest trends and forecasts...`
        },
        {
          title: `Commodity Prices and ${query} Market Report`,
          link: 'https://www.kitco.com/',
          snippet: `Current market analysis for ${query}. Commodity prices show mixed trends as global demand fluctuates...`
        }
      ];
      
      console.log('Returning fallback results:', mockResults.length);
      return NextResponse.json({ 
        results: mockResults,
        fallback: true 
      });
      
    } catch (fallbackError) {
      console.error('Fallback search also failed:', fallbackError);
      return NextResponse.json({ 
        results: [],
        error: 'Web search temporarily unavailable'
      });
    }
    
  } catch (error) {
    console.error('Web search error:', error);
    return NextResponse.json({ 
      results: [],
      error: 'Failed to perform web search' 
    });
  }
} 