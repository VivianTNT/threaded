"use client";

import React, { useState, useEffect } from 'react';
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
import { Search, RefreshCw, Calendar, Building, Tag, TrendingUp, TrendingDown, FileText, ExternalLink } from 'lucide-react';
import { IconTrendingDown, IconTrendingUp } from "@tabler/icons-react";
import { format } from 'date-fns';

interface NewsItem {
  id: string;
  news_id: number;
  symbol: string;
  company_name: string;
  headline: string;
  summary: string;
  source: string;
  datetime: string;
  story_url: string;
  topics: string[];
  primary_commodity: string;
  is_mining_related: boolean;
  is_project_related: boolean;
  mentions_financials: boolean;
  mentions_technical_report: boolean;
  sentiment_score?: number;
  relevance_score?: number;
}

// Extended sample news data with 50 items
const SAMPLE_NEWS: NewsItem[] = [
  // Today's news
  {
    id: '3001',
    news_id: 3001,
    symbol: 'LAC',
    company_name: 'Lithium Americas Corp',
    headline: 'Lithium Americas Receives Final Federal Permit for Thacker Pass Project',
    summary: 'The U.S. Bureau of Land Management issued the Record of Decision for Thacker Pass, clearing the way for construction of the largest known lithium resource in the United States.',
    source: 'Reuters',
    datetime: '2025-01-26T14:30:00Z',
    story_url: 'https://www.lithiumamericas.com/news/lithium-americas-receives-final-federal-permit-for-thacker-pass',
    topics: ['LAC', 'lithium', 'Nevada', 'permits'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.9,
    relevance_score: 10
  },
  {
    id: '3002',
    news_id: 3002,
    symbol: 'ALB',
    company_name: 'Albemarle Corporation',
    headline: 'Albemarle Reports 45% Increase in Lithium Hydroxide Production',
    summary: 'Q4 results show record production at Kemerton facility in Australia, with battery-grade lithium hydroxide output reaching 15,000 tonnes.',
    source: 'Mining Weekly',
    datetime: '2025-01-26T09:15:00Z',
    story_url: 'https://investors.albemarle.com/news-and-events/news/news-details',
    topics: ['ALB', 'lithium', 'Australia', 'production'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.85,
    relevance_score: 9
  },
  {
    id: '3004',
    news_id: 3004,
    symbol: 'FCX',
    company_name: 'Freeport-McMoRan Inc',
    headline: 'Freeport Discovers New High-Grade Copper Zone at Grasberg',
    summary: 'Drilling results reveal significant copper-gold mineralization extending 500 meters below current workings with grades averaging 2.1% Cu and 1.2 g/t Au.',
    source: 'Mining.com',
    datetime: '2025-01-26T11:00:00Z',
    story_url: 'https://www.fcx.com/news/news-releases',
    topics: ['FCX', 'copper', 'Indonesia', 'exploration'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.9,
    relevance_score: 10
  },
  {
    id: '3021',
    news_id: 3021,
    symbol: 'NEM',
    company_name: 'Newmont Corporation',
    headline: 'Newmont Completes $2.1 Billion Acquisition of Australian Gold Assets',
    summary: 'Newmont finalizes acquisition of three operating gold mines in Western Australia, adding 450,000 ounces to annual production.',
    source: 'Bloomberg',
    datetime: '2025-01-26T08:00:00Z',
    story_url: 'https://www.newmont.com/investors/news-release',
    topics: ['NEM', 'gold', 'Australia', 'acquisition'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.87,
    relevance_score: 9
  },
  {
    id: '3022',
    news_id: 3022,
    symbol: 'SLI',
    company_name: 'Standard Lithium',
    headline: 'Standard Lithium Achieves First Lithium Carbonate from Arkansas Brine',
    summary: 'Company produces battery-grade lithium carbonate from Smackover Formation brines using proprietary direct extraction technology.',
    source: 'Company Release',
    datetime: '2025-01-26T07:30:00Z',
    story_url: 'https://www.standardlithium.com/investors/news',
    topics: ['SLI', 'lithium', 'Arkansas', 'production'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.91,
    relevance_score: 10
  },
  {
    id: '3023',
    news_id: 3023,
    symbol: 'TECK',
    company_name: 'Teck Resources',
    headline: 'Teck Announces $3.5 Billion Copper Growth Strategy',
    summary: 'Company unveils plans to double copper production by 2030 through expansion of existing operations and new developments.',
    source: 'Reuters',
    datetime: '2025-01-25T16:00:00Z',
    story_url: 'https://www.teck.com/news/news-releases/',
    topics: ['TECK', 'copper', 'growth', 'strategy'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.88,
    relevance_score: 9
  },
  {
    id: '3024',
    news_id: 3024,
    symbol: 'SGML',
    company_name: 'Sigma Lithium',
    headline: 'Sigma Lithium Commissions Second Production Line in Brazil',
    summary: 'Grota do Cirilo project doubles production capacity to 520,000 tonnes per annum of lithium concentrate.',
    source: 'Mining.com',
    datetime: '2025-01-25T14:30:00Z',
    story_url: 'https://www.sigmalithiumresources.com/news',
    topics: ['SGML', 'lithium', 'Brazil', 'production'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.89,
    relevance_score: 10
  },
  {
    id: '3025',
    news_id: 3025,
    symbol: 'ERG',
    company_name: 'Eurasian Resources Group',
    headline: 'ERG Announces Discovery of Major Cobalt Deposit in DRC',
    summary: 'New discovery contains estimated 500,000 tonnes of cobalt resources, potentially one of largest deposits globally.',
    source: 'Financial Times',
    datetime: '2025-01-25T12:00:00Z',
    story_url: 'https://www.eurasianresources.lu/en/news',
    topics: ['ERG', 'cobalt', 'DRC', 'discovery'],
    primary_commodity: 'cobalt',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.92,
    relevance_score: 10
  },
  {
    id: '3026',
    news_id: 3026,
    symbol: 'AEM',
    company_name: 'Agnico Eagle Mines',
    headline: 'Agnico Eagle Reports Record Gold Production at Canadian Mines',
    summary: 'Company achieves 3.4 million ounce production milestone, beating guidance by 5% driven by higher grades.',
    source: 'Northern Miner',
    datetime: '2025-01-25T10:15:00Z',
    story_url: 'https://www.agnicoeagle.com/English/investor-relations',
    topics: ['AEM', 'gold', 'Canada', 'production'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.86,
    relevance_score: 8
  },
  {
    id: '3027',
    news_id: 3027,
    symbol: 'PALAF',
    company_name: 'Paladin Energy',
    headline: 'Paladin Energy Restarts Langer Heinrich Uranium Mine',
    summary: 'Namibian uranium mine resumes production after 7-year care and maintenance period, targeting 6 million pounds annually.',
    source: 'World Nuclear News',
    datetime: '2025-01-24T15:45:00Z',
    story_url: 'https://www.paladinenergy.com.au/news',
    topics: ['PALAF', 'uranium', 'Namibia', 'restart'],
    primary_commodity: 'uranium',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.88,
    relevance_score: 9
  },
  {
    id: '3028',
    news_id: 3028,
    symbol: 'CMP',
    company_name: 'Compass Minerals',
    headline: 'Compass Minerals Begins Lithium Production at Great Salt Lake',
    summary: 'First commercial lithium extraction from Utah brine operations produces battery-grade lithium carbonate.',
    source: 'Chemical Week',
    datetime: '2025-01-24T13:00:00Z',
    story_url: 'https://www.compassminerals.com/news',
    topics: ['CMP', 'lithium', 'Utah', 'production'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.87,
    relevance_score: 9
  },
  {
    id: '3029',
    news_id: 3029,
    symbol: 'AA',
    company_name: 'Alcoa Corporation',
    headline: 'Alcoa Invests $500 Million in Aluminum Smelter Upgrades',
    summary: 'Modernization program aims to reduce carbon emissions by 30% while increasing production efficiency.',
    source: 'Reuters',
    datetime: '2025-01-24T11:30:00Z',
    story_url: 'https://www.alcoa.com/global/en/news',
    topics: ['AA', 'aluminum', 'sustainability', 'investment'],
    primary_commodity: 'aluminum',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.82,
    relevance_score: 8
  },
  {
    id: '3030',
    news_id: 3030,
    symbol: 'GORO',
    company_name: 'Goro Nickel',
    headline: 'Tesla Signs Nickel Supply Agreement with New Caledonia Mine',
    summary: 'Five-year agreement secures 42,000 tonnes of nickel for Tesla battery production from Prony Resources.',
    source: 'Bloomberg',
    datetime: '2025-01-23T17:00:00Z',
    story_url: 'https://www.pronyresources.com/news',
    topics: ['GORO', 'nickel', 'Tesla', 'supply-agreement'],
    primary_commodity: 'nickel',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.9,
    relevance_score: 9
  },
  {
    id: '3031',
    news_id: 3031,
    symbol: 'LIT',
    company_name: 'Lithium ETF',
    headline: 'Global Lithium ETF Sees Record Inflows of $2.3 Billion',
    summary: 'Investor interest in lithium sector surges as EV adoption accelerates globally.',
    source: 'Financial Times',
    datetime: '2025-01-23T15:30:00Z',
    story_url: 'https://www.globalxetfs.com/funds/lit',
    topics: ['LIT', 'lithium', 'ETF', 'investment'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.85,
    relevance_score: 8
  },
  {
    id: '3032',
    news_id: 3032,
    symbol: 'CENX',
    company_name: 'Century Aluminum',
    headline: 'Century Aluminum Receives $500M DOE Loan for Green Smelter',
    summary: 'Department of Energy funding supports construction of first new U.S. aluminum smelter in 45 years.',
    source: 'Platts',
    datetime: '2025-01-23T13:15:00Z',
    story_url: 'https://www.centuryaluminum.com/news',
    topics: ['CENX', 'aluminum', 'DOE', 'green-energy'],
    primary_commodity: 'aluminum',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.91,
    relevance_score: 9
  },
  {
    id: '3033',
    news_id: 3033,
    symbol: 'LTHM',
    company_name: 'Livent Corporation',
    headline: 'Livent Expands Argentina Lithium Production by 40%',
    summary: 'Fenix project expansion increases lithium carbonate capacity to 40,000 tonnes annually.',
    source: 'S&P Global',
    datetime: '2025-01-23T11:00:00Z',
    story_url: 'https://livent.com/newsroom',
    topics: ['LTHM', 'lithium', 'Argentina', 'expansion'],
    primary_commodity: 'lithium',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.86,
    relevance_score: 9
  },
  {
    id: '3034',
    news_id: 3034,
    symbol: 'KGC',
    company_name: 'Kinross Gold',
    headline: 'Kinross Discovers Extension of Tasiast Gold Deposit in Mauritania',
    summary: 'New drilling confirms mineralization extends 1.2km south of current pit, adding potential 2 million ounces.',
    source: 'Mining Journal',
    datetime: '2025-01-22T16:30:00Z',
    story_url: 'https://www.kinross.com/news-and-investors',
    topics: ['KGC', 'gold', 'Mauritania', 'exploration'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.88,
    relevance_score: 9
  },
  {
    id: '3035',
    news_id: 3035,
    symbol: 'WPM',
    company_name: 'Wheaton Precious Metals',
    headline: 'Wheaton Signs $1.4 Billion Streaming Agreement for Chilean Copper',
    summary: 'Agreement with Antofagasta provides exposure to 100% of gold and 25% of copper from Centinela mine.',
    source: 'Reuters',
    datetime: '2025-01-22T14:00:00Z',
    story_url: 'https://www.wheatonpm.com/news',
    topics: ['WPM', 'copper', 'Chile', 'streaming'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.87,
    relevance_score: 9
  },
  {
    id: '3036',
    news_id: 3036,
    symbol: 'GLENCORE',
    company_name: 'Glencore',
    headline: 'Glencore Announces $7 Billion Investment in Congo Copper Operations',
    summary: 'Major expansion of Mutanda and Katanga mines to meet growing demand for electric vehicle batteries.',
    source: 'Financial Times',
    datetime: '2025-01-22T11:45:00Z',
    story_url: 'https://www.glencore.com/media-and-insights/news',
    topics: ['GLENCORE', 'copper', 'DRC', 'investment'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.85,
    relevance_score: 10
  },
  {
    id: '3037',
    news_id: 3037,
    symbol: 'EMX',
    company_name: 'EMX Royalty',
    headline: 'EMX Royalty Acquires Portfolio of 47 Battery Metal Royalties',
    summary: 'Acquisition adds lithium, nickel, and cobalt royalties across North America and Europe valued at $230 million.',
    source: 'Mining.com',
    datetime: '2025-01-21T15:15:00Z',
    story_url: 'https://www.emxroyalty.com/news',
    topics: ['EMX', 'royalties', 'battery-metals', 'acquisition'],
    primary_commodity: 'diversified',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.86,
    relevance_score: 8
  },
  {
    id: '3038',
    news_id: 3038,
    symbol: 'UUUU',
    company_name: 'Energy Fuels',
    headline: 'Energy Fuels Begins Rare Earth Production at White Mesa Mill',
    summary: 'First U.S. production of mixed rare earth carbonate from monazite sands marks milestone in domestic supply chain.',
    source: 'World Nuclear News',
    datetime: '2025-01-21T12:30:00Z',
    story_url: 'https://www.energyfuels.com/news',
    topics: ['UUUU', 'rare-earth', 'USA', 'production'],
    primary_commodity: 'rare_earth',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.9,
    relevance_score: 10
  },
  {
    id: '3039',
    news_id: 3039,
    symbol: 'TRQ',
    company_name: 'Turquoise Hill',
    headline: 'Oyu Tolgoi Underground Mine Reaches Commercial Production',
    summary: 'Mongolia mega-project achieves sustainable production levels, on track for 500,000 tonnes copper annually.',
    source: 'Bloomberg',
    datetime: '2025-01-21T09:00:00Z',
    story_url: 'https://www.turquoisehill.com/news',
    topics: ['TRQ', 'copper', 'Mongolia', 'production'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.91,
    relevance_score: 10
  },
  {
    id: '3040',
    news_id: 3040,
    symbol: 'ANGLO',
    company_name: 'Anglo American',
    headline: 'Anglo American Approves $3.9 Billion Quellaveco Expansion',
    summary: 'Peru copper mine expansion to add 100,000 tonnes annual production capacity by 2027.',
    source: 'Reuters',
    datetime: '2025-01-20T16:45:00Z',
    story_url: 'https://www.angloamerican.com/media/news',
    topics: ['ANGLO', 'copper', 'Peru', 'expansion'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.84,
    relevance_score: 9
  },
  {
    id: '3041',
    news_id: 3041,
    symbol: 'FNV',
    company_name: 'Franco-Nevada',
    headline: 'Franco-Nevada Reports Record Royalty Revenue of $1.2 Billion',
    summary: 'Precious metals royalty company benefits from higher gold prices and increased production from portfolio assets.',
    source: 'Globe and Mail',
    datetime: '2025-01-20T14:00:00Z',
    story_url: 'https://www.franco-nevada.com/news',
    topics: ['FNV', 'royalties', 'gold', 'revenue'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.88,
    relevance_score: 8
  },
  {
    id: '3042',
    news_id: 3042,
    symbol: 'CLF',
    company_name: 'Cleveland-Cliffs',
    headline: 'Cleveland-Cliffs Completes Acquisition of Ferrous Processing',
    summary: 'Acquisition strengthens position as largest flat-rolled steel producer in North America.',
    source: 'American Metal Market',
    datetime: '2025-01-20T11:15:00Z',
    story_url: 'https://www.clevelandcliffs.com/news',
    topics: ['CLF', 'steel', 'acquisition', 'processing'],
    primary_commodity: 'steel',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.82,
    relevance_score: 7
  },
  {
    id: '3043',
    news_id: 3043,
    symbol: 'SSRM',
    company_name: 'SSR Mining',
    headline: 'SSR Mining Discovers High-Grade Silver Zone at Çöpler Mine',
    summary: 'New discovery in Turkey contains estimated 50 million ounces of silver at grades exceeding 200 g/t.',
    source: 'Mining Weekly',
    datetime: '2025-01-19T15:30:00Z',
    story_url: 'https://www.ssrmining.com/news',
    topics: ['SSRM', 'silver', 'Turkey', 'discovery'],
    primary_commodity: 'silver',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.89,
    relevance_score: 9
  },
  {
    id: '3044',
    news_id: 3044,
    symbol: 'POLY',
    company_name: 'Polymet Mining',
    headline: 'Polymet Receives Final State Permits for NorthMet Project',
    summary: 'Minnesota copper-nickel-platinum project clears final regulatory hurdles after decade-long permitting process.',
    source: 'E&MJ',
    datetime: '2025-01-19T12:00:00Z',
    story_url: 'https://www.polymetmining.com/news',
    topics: ['POLY', 'copper', 'Minnesota', 'permits'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.87,
    relevance_score: 9
  },
  {
    id: '3045',
    news_id: 3045,
    symbol: 'HBM',
    company_name: 'Hudbay Minerals',
    headline: 'Hudbay Reports Breakthrough in Copper Recovery Technology',
    summary: 'New processing technology increases copper recovery rates by 15% while reducing operating costs by 20%.',
    source: 'Canadian Mining Journal',
    datetime: '2025-01-19T09:45:00Z',
    story_url: 'https://www.hudbayminerals.com/news',
    topics: ['HBM', 'copper', 'technology', 'innovation'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.88,
    relevance_score: 8
  },
  {
    id: '3046',
    news_id: 3046,
    symbol: 'PAAS',
    company_name: 'Pan American Silver',
    headline: 'Pan American Silver Expands La Colorada Mine Life by 8 Years',
    summary: 'Updated mine plan adds 100 million ounces of silver reserves through underground expansion in Mexico.',
    source: 'Silver Institute',
    datetime: '2025-01-18T16:00:00Z',
    story_url: 'https://www.panamericansilver.com/news',
    topics: ['PAAS', 'silver', 'Mexico', 'expansion'],
    primary_commodity: 'silver',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: true,
    sentiment_score: 0.86,
    relevance_score: 9
  },
  {
    id: '3047',
    news_id: 3047,
    symbol: 'BTG',
    company_name: 'B2Gold',
    headline: 'B2Gold Completes Construction of Fekola Solar Plant in Mali',
    summary: '36 MW solar plant reduces carbon emissions by 39,000 tonnes annually at flagship gold mine.',
    source: 'Mining Magazine',
    datetime: '2025-01-18T13:30:00Z',
    story_url: 'https://www.b2gold.com/news',
    topics: ['BTG', 'gold', 'Mali', 'sustainability'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.83,
    relevance_score: 8
  },
  {
    id: '3048',
    news_id: 3048,
    symbol: 'EQX',
    company_name: 'Equinox Gold',
    headline: 'Equinox Gold Announces Merger with i-80 Gold',
    summary: 'All-stock merger creates mid-tier gold producer with 1.2 million ounce annual production profile.',
    source: 'Globe Newswire',
    datetime: '2025-01-18T10:00:00Z',
    story_url: 'https://www.equinoxgold.com/news',
    topics: ['EQX', 'gold', 'merger', 'consolidation'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.85,
    relevance_score: 9
  },
  {
    id: '3049',
    news_id: 3049,
    symbol: 'OR',
    company_name: 'Osisko Gold Royalties',
    headline: 'Osisko Acquires 3% NSR Royalty on Windfall Gold Project',
    summary: 'C$300 million investment secures royalty on one of Canada highest-grade gold development projects.',
    source: 'Northern Miner',
    datetime: '2025-01-17T15:45:00Z',
    story_url: 'https://www.osiskogr.com/news',
    topics: ['OR', 'gold', 'royalty', 'Canada'],
    primary_commodity: 'gold',
    is_mining_related: true,
    is_project_related: false,
    mentions_financials: true,
    mentions_technical_report: false,
    sentiment_score: 0.84,
    relevance_score: 8
  },
  {
    id: '3050',
    news_id: 3050,
    symbol: 'CMMC',
    company_name: 'Copper Mountain Mining',
    headline: 'Copper Mountain Achieves Record Mill Throughput at Eva Project',
    summary: 'Australian copper project exceeds design capacity by 20% in first quarter of operations.',
    source: 'Australian Mining',
    datetime: '2025-01-17T12:15:00Z',
    story_url: 'https://www.cumtn.com/news',
    topics: ['CMMC', 'copper', 'Australia', 'production'],
    primary_commodity: 'copper',
    is_mining_related: true,
    is_project_related: true,
    mentions_financials: false,
    mentions_technical_report: false,
    sentiment_score: 0.87,
    relevance_score: 9
  }
];

export function NewsAnnouncements() {
  const [news, setNews] = useState<NewsItem[]>(SAMPLE_NEWS);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCommodity, setSelectedCommodity] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

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
        }
      }
    } catch (error) {
      console.error('Error fetching news:', error);
      // Keep using sample data if API fails
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Try to fetch from API but use sample data as fallback
    fetchNews();
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    await fetch('/api/news/refresh', { method: 'POST' });
    await fetchNews();
    setRefreshing(false);
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
          <div className="flex items-center justify-between">
            <div>
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
              <select
                value={selectedCommodity || ''}
                onChange={(e) => setSelectedCommodity(e.target.value || null)}
                className="px-4 py-2 border rounded-lg bg-white dark:bg-gray-800"
              >
                <option value="">All Commodities</option>
                {commodities.map(commodity => (
                  <option key={commodity} value={commodity}>
                    {commodity}
                  </option>
                ))}
              </select>
            </div>

            {/* Quick Stats with exact dashboard structure */}
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
                <CardFooter className="flex-col items-start gap-1.5 text-sm">
                  <div className="line-clamp-1 flex gap-2 font-medium">
                    Trending up this week <IconTrendingUp className="size-4" />
                  </div>
                  <div className="text-muted-foreground">
                    Latest mining headlines
                  </div>
                </CardFooter>
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
                <CardFooter className="flex-col items-start gap-1.5 text-sm">
                  <div className="line-clamp-1 flex gap-2 font-medium">
                    Strong sector coverage <IconTrendingUp className="size-4" />
                  </div>
                  <div className="text-muted-foreground">
                    Industry specific news
                  </div>
                </CardFooter>
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
                      {Math.abs(stats.projectGrowth).toFixed(1)}%
                    </Badge>
                  </CardAction>
                </CardHeader>
                <CardFooter className="flex-col items-start gap-1.5 text-sm">
                  <div className="line-clamp-1 flex gap-2 font-medium">
                    Down 2.5% this period <IconTrendingDown className="size-4" />
                  </div>
                  <div className="text-muted-foreground">
                    Development milestones
                  </div>
                </CardFooter>
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
                <CardFooter className="flex-col items-start gap-1.5 text-sm">
                  <div className="line-clamp-1 flex gap-2 font-medium">
                    Record filing activity <IconTrendingUp className="size-4" />
                  </div>
                  <div className="text-muted-foreground">
                    NI 43-101 & S-K 1300
                  </div>
                </CardFooter>
              </Card>
            </div>
          </div>

          {/* News Table */}
          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[100px]">Date</TableHead>
                  <TableHead className="w-[80px]">Symbol</TableHead>
                  <TableHead>Headline</TableHead>
                  <TableHead className="w-[120px]">Commodity</TableHead>
                  <TableHead className="w-[100px]">Tags</TableHead>
                  <TableHead className="w-[80px]">Source</TableHead>
                  <TableHead className="w-[50px]"></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-8">
                      Loading news...
                    </TableCell>
                  </TableRow>
                ) : filteredNews.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="text-center py-8 text-gray-500">
                      No news found
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredNews.map((item) => (
                    <TableRow key={item.id}>
                      <TableCell className="text-xs text-gray-500">
                        <div className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          {format(new Date(item.datetime), 'MMM dd, yyyy')}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{item.symbol}</Badge>
                      </TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <div className="text-sm">{item.headline}</div>
                          {item.summary && (
                            <div className="text-xs text-gray-500 line-clamp-2">
                              {item.summary}
                            </div>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        {item.primary_commodity && (
                          <Badge variant="secondary">
                            {item.primary_commodity}
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {item.is_project_related && (
                            <Badge variant="outline" className="text-xs">
                              <Building className="h-3 w-3 mr-1" />
                              Project
                            </Badge>
                          )}
                          {item.mentions_financials && (
                            <Badge variant="outline" className="text-xs">
                              <TrendingUp className="h-3 w-3 mr-1" />
                              Financials
                            </Badge>
                          )}
                          {item.mentions_technical_report && (
                            <Badge variant="outline" className="text-xs">
                              <FileText className="h-3 w-3 mr-1" />
                              Technical
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <span className="text-xs text-gray-500">{item.source}</span>
                      </TableCell>
                      <TableCell>
                        {item.story_url && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => window.open(item.story_url, '_blank')}
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