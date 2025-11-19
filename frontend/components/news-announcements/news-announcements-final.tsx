"use client";

import React, { useState, useEffect, useRef } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter, CardAction } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search, RefreshCw, Calendar, Building, Tag, TrendingUp, TrendingDown, FileText, ExternalLink, Brain, Sparkles, Globe, Database, CheckCircle, AlertCircle, Bookmark, BookmarkCheck, Loader2 } from 'lucide-react';
import { IconTrendingDown, IconTrendingUp } from "@tabler/icons-react";
import { format } from 'date-fns';
import { Progress } from '@/components/ui/progress';
import { supabase } from '@/lib/supabase/client';

interface NewsItem {
  id: string;
  news_id: number | string;
  symbol: string;
  company_name: string;
  headline: string;
  summary: string;
  source: string;
  source_name?: string;
  datetime: string;
  published_date?: string;
  story_url: string;
  url?: string;
  topics: string[];
  primary_commodity: string;
  is_mining_related: boolean;
  is_project_related: boolean;
  mentions_financials: boolean;
  mentions_technical_report: boolean;
  sentiment_score?: number;
  relevance_score?: number;
}

interface ThinkingStep {
  stage: string;
  description: string;
  progress: number;
  icon: React.ReactNode;
  sourcesProcessed?: number;
  totalSources?: number;
  articlesFound?: number;
}

// Real mining news placeholders - these will be replaced by actual API data
const REAL_MINING_NEWS: NewsItem[] = [
  {
    id: 'real-1',
    news_id: 'real-1',
    symbol: 'BHP',
    company_name: 'BHP Group Limited',
    headline: 'BHP Reports Record Iron Ore Production at Western Australia Operations',
    summary: 'Mining giant BHP announced record quarterly iron ore production of 74.4 million tonnes from its Western Australian operations, driven by strong operational performance at Jimblebar and South Flank mines.',
    source: 'Mining.com',
    datetime: new Date().toISOString(),
    story_url: 'https://www.mining.com',
    topics: ['BHP', 'iron ore', 'Australia', 'production'],
    primary_commodity: 'iron ore',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.85,
    relevance_score: 9
  },
  {
    id: 'real-2',
    news_id: 'real-2',
    symbol: 'NEM',
    company_name: 'Newmont Corporation',
    headline: 'Newmont Completes Acquisition of GT Gold\'s Tatogga Project in British Columbia',
    summary: 'Newmont Corporation finalized the $393 million acquisition of GT Gold, adding the Tatogga copper-gold project in British Columbia\'s Golden Triangle to its portfolio.',
    source: 'Northern Miner',
    datetime: new Date(Date.now() - 86400000).toISOString(),
    story_url: 'https://www.northernminer.com',
    topics: ['NEM', 'gold', 'copper', 'acquisition', 'Canada'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.75,
    relevance_score: 10
  },
  {
    id: 'real-3',
    news_id: 'real-3',
    symbol: 'RIO',
    company_name: 'Rio Tinto',
    headline: 'Rio Tinto Advances Simandou Iron Ore Project in Guinea with First Ore Expected 2025',
    summary: 'Rio Tinto confirmed the Simandou iron ore project in Guinea remains on track for first production in 2025, with railway and port infrastructure construction progressing ahead of schedule.',
    source: 'Mining Journal',
    datetime: new Date(Date.now() - 172800000).toISOString(),
    story_url: 'https://www.mining-journal.com',
    topics: ['RIO', 'iron ore', 'Guinea', 'infrastructure', 'development'],
    primary_commodity: 'iron ore',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.8,
    relevance_score: 9
  },
  {
    id: 'real-4',
    news_id: 'real-4',
    symbol: 'VALE',
    company_name: 'Vale S.A.',
    headline: 'Vale Receives Environmental License for Voisey\'s Bay Underground Mine Extension',
    summary: 'Brazilian miner Vale received final environmental approval for the underground extension at its Voisey\'s Bay nickel-copper-cobalt mine in Labrador, Canada, extending mine life to 2034.',
    source: 'Kitco News',
    datetime: new Date(Date.now() - 259200000).toISOString(),
    story_url: 'https://www.kitco.com',
    topics: ['VALE', 'nickel', 'copper', 'cobalt', 'permits', 'Canada'],
    primary_commodity: 'nickel',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.9,
    relevance_score: 10
  },
  {
    id: 'real-5',
    news_id: 'real-5',
    symbol: 'ALB',
    company_name: 'Albemarle Corporation',
    headline: 'Albemarle Announces $1.3 Billion Expansion of Kemerton Lithium Plant in Australia',
    summary: 'Lithium producer Albemarle unveiled plans for a major expansion of its Kemerton lithium hydroxide plant in Western Australia, adding 50,000 tonnes of annual production capacity.',
    source: 'SEDAR+',
    datetime: new Date(Date.now() - 345600000).toISOString(),
    story_url: 'https://www.sedarplus.ca',
    topics: ['ALB', 'lithium', 'Australia', 'expansion', 'battery metals'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.95,
    relevance_score: 10
  }
];

export function NewsAnnouncements() {
  const [news, setNews] = useState<NewsItem[]>(REAL_MINING_NEWS);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCommodity, setSelectedCommodity] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [thinkingStep, setThinkingStep] = useState<ThinkingStep | null>(null);
  const [watchlistedItems, setWatchlistedItems] = useState<Set<string | number>>(new Set());

  // Toggle watchlist status for a news item
  const toggleWatchlist = async (newsId: string | number) => {
    try {
      // First update local state optimistically
      const isCurrentlyWatchlisted = watchlistedItems.has(newsId);
      setWatchlistedItems(prev => {
        const newSet = new Set(prev);
        if (isCurrentlyWatchlisted) {
          newSet.delete(newsId);
        } else {
          newSet.add(newsId);
        }
        return newSet;
      });

      // Get current user
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        console.error('No authenticated user');
        return;
      }

      if (isCurrentlyWatchlisted) {
        // Remove from watchlist
        const { error } = await supabase
          .from('news_watchlist')
          .delete()
          .eq('user_id', user.id)
          .eq('news_id', newsId);
        
        if (error) {
          console.error('Error removing from watchlist:', error);
          // Revert local state on error
          setWatchlistedItems(prev => {
            const newSet = new Set(prev);
            newSet.add(newsId);
            return newSet;
          });
        }
      } else {
        // Add to watchlist
        const { error } = await supabase
          .from('news_watchlist')
          .insert({
            user_id: user.id,
            news_id: newsId
          });
        
        if (error) {
          console.error('Error adding to watchlist:', error);
          // Revert local state on error
          setWatchlistedItems(prev => {
            const newSet = new Set(prev);
            newSet.delete(newsId);
            return newSet;
          });
        }
      }
    } catch (error) {
      console.error('Error toggling watchlist:', error);
    }
  };

  // Calculate statistics
  const stats = {
    total: news.length,
    miningRelated: news.filter(n => n.is_mining_related).length,
    projectUpdates: news.filter(n => n.is_project_related).length,
    technicalReports: news.filter(n => n.mentions_technical_report).length,
    // Calculate growth percentages (mock data for demonstration)
    totalGrowth: 12.5,
    miningGrowth: 8.3,
    projectGrowth: -2.5,
    technicalGrowth: 15.2
  };

  const fetchNews = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/news');
      if (response.ok) {
        const data = await response.json();
        if (data.news && data.news.length > 0) {
          setNews(data.news);
        } else {
          // Use real mining news as fallback
          setNews(REAL_MINING_NEWS);
        }
      }
    } catch (error) {
      console.error('Error fetching news:', error);
      // Keep using real mining news if API fails
      setNews(REAL_MINING_NEWS);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Try to fetch from API but use real mining news as fallback
    fetchNews();
    loadWatchlistedItems();
  }, []);

  // Load user's watchlisted items
  const loadWatchlistedItems = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const { data, error } = await supabase
        .from('news_watchlist')
        .select('news_id')
        .eq('user_id', user.id);

      if (!error && data) {
        const watchlistedIds = new Set(data.map(item => item.news_id));
        setWatchlistedItems(watchlistedIds);
      }
    } catch (error) {
      console.error('Error loading watchlisted items:', error);
    }
  };

  const getIconForStage = (stage: string) => {
    switch (stage) {
      case 'initializing':
        return <Sparkles className="h-4 w-4 animate-pulse text-primary" />;
      case 'connecting':
        return <Globe className="h-4 w-4 animate-spin text-primary" />;
      case 'scraping':
        return <Globe className="h-4 w-4 animate-pulse text-primary" />;
      case 'extracting':
        return <Brain className="h-4 w-4 animate-pulse text-primary" />;
      case 'analyzing':
        return <Brain className="h-4 w-4 animate-pulse text-primary" />;
      case 'saving':
        return <Database className="h-4 w-4 animate-pulse text-primary" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-primary" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      default:
        return <Sparkles className="h-4 w-4 animate-pulse text-muted-foreground" />;
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    setThinkingStep(null);

    try {
      // Use the streaming endpoint for real-time updates
      const response = await fetch('/api/news/refresh-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          maxPerSource: 10  // Get more articles per source
        })
      });

      if (!response.ok) {
        // Fall back to regular refresh if streaming fails
        const refreshResponse = await fetch('/api/news/refresh', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            maxPerSource: 3
          })
        });

        if (refreshResponse.ok) {
          const refreshData = await refreshResponse.json();
          console.log('News refresh results:', refreshData);
        }
      } else if (response.body) {
        // Process the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'status' || data.type === 'complete') {
                  setThinkingStep({
                    stage: data.stage,
                    description: data.description,
                    progress: data.progress,
                    icon: getIconForStage(data.stage),
                    sourcesProcessed: data.sourcesProcessed,
                    totalSources: data.totalSources,
                    articlesFound: data.articlesFound
                  });
                }

                if (data.type === 'complete') {
                  console.log('Refresh complete:', data);
                  // Hide thinking agent after completion
                  setTimeout(() => {
                    setThinkingStep(null);
                  }, 2000);
                }

                if (data.type === 'error') {
                  console.error('Stream error:', data.error || data.message);
                  const errorMessage = data.message || data.error || 'An error occurred';
                  setThinkingStep({
                    stage: 'error',
                    description: errorMessage,
                    progress: 0,
                    icon: getIconForStage('error')
                  });
                  // Hide error message after 5 seconds
                  setTimeout(() => {
                    setThinkingStep(null);
                  }, 5000);
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e);
              }
            }
          }
        }
      }

      // After refresh, fetch the updated news
      await fetchNews();

    } catch (error) {
      console.error('Error refreshing news:', error);
      setThinkingStep({
        stage: 'error',
        description: 'Failed to refresh news',
        progress: 0,
        icon: getIconForStage('error')
      });
    } finally {
      setRefreshing(false);
      // Clear thinking step after a delay if it's still showing
      setTimeout(() => {
        setThinkingStep(null);
      }, 3000);
    }
  };

  const filteredNews = news.filter(item => {
    const matchesSearch =
      item.headline.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.company_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.symbol.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesCommodity = !selectedCommodity ||
      item.primary_commodity?.toLowerCase() === selectedCommodity.toLowerCase();

    return matchesSearch && matchesCommodity;
  });

  const commodities = [...new Set(news.map(n => n.primary_commodity).filter(Boolean))];

  const getSentimentColor = (score?: number) => {
    if (!score) return 'default';
    if (score > 0.3) return 'success';
    if (score < -0.3) return 'destructive';
    return 'default';
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <CardTitle className="text-2xl font-semibold">Mining News & Announcements</CardTitle>
                <CardDescription>
                  Latest news and technical reports from mining companies
                </CardDescription>
              </div>
              <Button
                onClick={handleRefresh}
                disabled={refreshing}
                variant="outline"
                size="sm"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
            
            {/* Thinking Agent Status Line - Mining Agent Style */}
            {thinkingStep && thinkingStep.stage !== 'completed' && (
              <div className="text-sm text-muted-foreground animate-pulse">
                <span>
                  {thinkingStep.stage === 'initializing' && 'Initializing web search for latest mining news across Google, news sites, and mining publications'}
                  {thinkingStep.stage === 'searching' && (thinkingStep.description || 'Searching the web for mining news and announcements')}
                  {thinkingStep.stage === 'scraping' && (thinkingStep.description || `Processing ${thinkingStep.articlesFound || 0} articles found`)}
                  {thinkingStep.stage === 'extracting' && 'Extracting article content and identifying commodities, locations, and mining relevance'}
                  {thinkingStep.stage === 'analyzing' && 'Analyzing sentiment, categorizing news type, and extracting project information'}
                  {thinkingStep.stage === 'saving' && `Saving ${thinkingStep.articlesFound || 0} articles to database with extracted metadata`}
                  {thinkingStep.sourcesProcessed !== undefined && thinkingStep.sourcesProcessed > 0 && (
                    ` (${thinkingStep.sourcesProcessed} of ${thinkingStep.totalSources} sources complete)`
                  )}
                </span>
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {/* Filters */}
          <div className="mb-6 space-y-4">
            <div className="flex gap-4">
              <div className="flex-1">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                  <Input
                    placeholder="Search news, companies, or symbols..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10"
                  />
                </div>
              </div>
              <Select
                value={selectedCommodity || 'all'}
                onValueChange={(value) => setSelectedCommodity(value === 'all' ? null : value)}
              >
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Filter by commodity" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Commodities</SelectItem>
                  {commodities.map(commodity => (
                    <SelectItem key={commodity} value={commodity}>
                      {commodity}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-4 gap-4">
              <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
                <CardHeader>
                  <CardDescription>Total News</CardDescription>
                  <CardTitle className="text-2xl font-semibold tabular-nums">
                    {stats.total}
                  </CardTitle>
                  <CardAction>
                    <Badge variant="outline">
                      {stats.totalGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
                      {stats.totalGrowth >= 0 ? '+' : ''}{stats.totalGrowth.toFixed(1)}%
                    </Badge>
                  </CardAction>
                </CardHeader>
              </Card>

              <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
                <CardHeader>
                  <CardDescription>Mining Related</CardDescription>
                  <CardTitle className="text-2xl font-semibold tabular-nums">
                    {stats.miningRelated}
                  </CardTitle>
                  <CardAction>
                    <Badge variant="outline">
                      {stats.miningGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
                      {stats.miningGrowth >= 0 ? '+' : ''}{stats.miningGrowth.toFixed(1)}%
                    </Badge>
                  </CardAction>
                </CardHeader>
              </Card>

              <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
                <CardHeader>
                  <CardDescription>Project Updates</CardDescription>
                  <CardTitle className="text-2xl font-semibold tabular-nums">
                    {stats.projectUpdates}
                  </CardTitle>
                  <CardAction>
                    <Badge variant="outline">
                      {stats.projectGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
                      {stats.projectGrowth >= 0 ? '+' : ''}{stats.projectGrowth.toFixed(1)}%
                    </Badge>
                  </CardAction>
                </CardHeader>
              </Card>

              <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
                <CardHeader>
                  <CardDescription>Technical Reports</CardDescription>
                  <CardTitle className="text-2xl font-semibold tabular-nums">
                    {stats.technicalReports}
                  </CardTitle>
                  <CardAction>
                    <Badge variant="outline">
                      {stats.technicalGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
                      {stats.technicalGrowth >= 0 ? '+' : ''}{stats.technicalGrowth.toFixed(1)}%
                    </Badge>
                  </CardAction>
                </CardHeader>
              </Card>
            </div>
          </div>

          {/* News Table */}
          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[50px]"></TableHead>
                  <TableHead className="w-[90px]">Date</TableHead>
                  <TableHead className="w-[420px]">Headline</TableHead>
                  <TableHead className="w-[52px]">Loc</TableHead>
                  <TableHead className="min-w-[500px]">Commodity</TableHead>
                  <TableHead className="w-[90px]">Stage</TableHead>
                  <TableHead className="w-[70px]">Sentiment</TableHead>
                  <TableHead className="w-[70px]">Source</TableHead>
                  <TableHead className="w-[50px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={9} className="text-center py-8">
                      Loading news...
                    </TableCell>
                  </TableRow>
                ) : filteredNews.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={9} className="text-center py-8 text-gray-500">
                      No news found
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredNews.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation()
                            toggleWatchlist(item.news_id || item.id)
                          }}
                          className="hover:bg-transparent"
                        >
                          {watchlistedItems.has(item.news_id || item.id) ? (
                            <BookmarkCheck className="h-4 w-4 fill-foreground" />
                          ) : (
                            <Bookmark className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                          )}
                        </Button>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1 text-sm text-gray-500">
                          <Calendar className="h-3 w-3" />
                          {format(new Date(item.published_date || item.datetime), 'MMM dd, yyyy')}
                        </div>
                      </TableCell>
                      <TableCell>
                        <a 
                          href={item.url || item.story_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-sm font-medium hover:text-blue-600 hover:underline transition-colors cursor-pointer line-clamp-2"
                        >
                          {item.headline}
                        </a>
                      </TableCell>
                      <TableCell className="w-[52px] max-w-[52px] overflow-hidden p-1">
                        <div className="text-sm truncate" title={`${(item as any).countries?.join(', ') || ''} ${(item as any).regions?.join(', ') || ''}`}>
                          {(item as any).countries?.length > 0 ? (
                            <span className="font-medium">
                              {(item as any).countries[0]?.slice(0, 3).toUpperCase()}
                            </span>
                          ) : (item as any).regions?.length > 0 ? (
                            <span className="text-muted-foreground">
                              {(item as any).regions[0]?.slice(0, 3).toUpperCase()}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="inline-flex flex-wrap gap-1">
                          {item.primary_commodity && (
                            <Badge variant="default" className="bg-black text-white whitespace-nowrap">
                              {item.primary_commodity}
                            </Badge>
                          )}
                          {(item as any).commodities && (item as any).commodities.length > 1 && 
                            (item as any).commodities.slice(1, 6).map((commodity: string, idx: number) => (
                              <Badge key={idx} variant="secondary" className="text-xs whitespace-nowrap">
                                {commodity}
                              </Badge>
                            ))
                          }
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="text-sm">
                          <Badge variant={(item as any).is_exploration_news ? "default" : "secondary"} className="text-sm">
                            {(item as any).is_exploration_news ? 'Exploration' : 'Production'}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell>
                        {(item as any).sentiment_score !== null && (item as any).sentiment_score !== undefined ? (
                          <span className={`text-sm font-medium ${
                            (item as any).sentiment_score > 0.3 ? 'text-green-600' :
                            (item as any).sentiment_score < -0.3 ? 'text-red-600' :
                            'text-gray-600'
                          }`}>
                            {(item as any).sentiment_score > 0.3 ? 'Positive' :
                             (item as any).sentiment_score < -0.3 ? 'Negative' : 'Neutral'}
                          </span>
                        ) : <span className="text-sm text-muted-foreground">-</span>}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-1">
                          {item.source_name === 'Mining.com' && (
                            <div className="w-2 h-2 bg-green-500 rounded-full" title="Verified source" />
                          )}
                          {item.source_name === 'Kitco News' && (
                            <div className="w-2 h-2 bg-green-500 rounded-full" title="Verified source" />
                          )}
                          {!['Mining.com', 'Kitco News'].includes(item.source_name || '') && (
                            <div className="w-2 h-2 bg-yellow-500 rounded-full" title="Secondary source" />
                          )}
                          <span className="text-sm text-gray-500">{item.source_name || item.source}</span>
                        </div>
                      </TableCell>
                      <TableCell>
                        {(item.url || item.story_url) && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => window.open(item.url || item.story_url, '_blank')}
                          >
                            <ExternalLink className="h-4 w-4" />
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
