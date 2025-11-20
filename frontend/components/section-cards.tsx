"use client"

import { IconTrendingDown, IconTrendingUp } from "@tabler/icons-react"
import { useEffect, useState } from "react"
import { supabase } from "@/lib/supabase/client"

import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { InfoTooltip } from '@/components/ui/info-tooltip'
import { ContextMenuChat } from '@/components/ui/context-menu-chat'

interface DashboardStats {
  totalProjects: number
  projectsGrowth: number
  totalCompanies: number
  companiesGrowth: number
  totalNews: number
  newsGrowth: number
  watchlistedItems: number
  watchlistedGrowth: number
}

export function SectionCards() {
  const [stats, setStats] = useState<DashboardStats>({
    totalProjects: 0,
    projectsGrowth: 0,
    totalCompanies: 0,
    companiesGrowth: 0,
    totalNews: 0,
    newsGrowth: 0,
    watchlistedItems: 0,
    watchlistedGrowth: 0
  })

  useEffect(() => {
    fetchDashboardStats()
  }, [])

  const fetchDashboardStats = async () => {
    try {
      // Fetch total projects
      const { count: projectCount } = await supabase
        .from('projects')
        .select('*', { count: 'exact', head: true })

      // Fetch total companies from companies table
      const { count: companyCount } = await supabase
        .from('companies')
        .select('*', { count: 'exact', head: true })

      // Fetch total news
      const { count: newsCount } = await supabase
        .from('news')
        .select('*', { count: 'exact', head: true })

      // Fetch watchlisted items (projects + companies + news)
      const { count: watchlistedProjects } = await supabase
        .from('projects')
        .select('*', { count: 'exact', head: true })
        .eq('watchlist', true)

      const { count: watchlistedCompanies } = await supabase
        .from('companies')
        .select('*', { count: 'exact', head: true })
        .eq('watchlist', true)

      const { count: watchlistedNews } = await supabase
        .from('news')
        .select('*', { count: 'exact', head: true })
        .eq('watchlist', true)

      const totalWatchlisted = (watchlistedProjects || 0) + (watchlistedCompanies || 0) + (watchlistedNews || 0)

      // Calculate growth (mock data for now - in production you'd compare with last month)
      const projectsLastMonth = Math.floor((projectCount || 0) * 0.88)
      const projectsGrowth = projectCount ? ((projectCount - projectsLastMonth) / projectsLastMonth * 100) : 0

      const companiesLastMonth = Math.floor((companyCount || 0) * 0.95)
      const companiesGrowth = companyCount ? ((companyCount - companiesLastMonth) / companiesLastMonth * 100) : 0

      const newsLastWeek = Math.floor((newsCount || 0) * 0.85)
      const newsGrowth = newsCount ? ((newsCount - newsLastWeek) / newsLastWeek * 100) : 0

      setStats({
        totalProjects: projectCount || 0,
        projectsGrowth: projectsGrowth,
        totalCompanies: companyCount || 0,
        companiesGrowth: companiesGrowth,
        totalNews: newsCount || 0,
        newsGrowth: newsGrowth,
        watchlistedItems: totalWatchlisted,
        watchlistedGrowth: 0 // Could calculate based on change in watchlist over time
      })
    } catch (error) {
      console.error('Error fetching dashboard stats:', error)
    }
  }
  return (
    <div className="grid grid-cols-1 gap-4 px-4 lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
      <ContextMenuChat
        data={{
          totalProjects: stats.totalProjects,
          growth: stats.projectsGrowth,
          type: 'projects'
        }}
        dataType="metric"
        context="Total mining projects in database"
      >
        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
        <CardHeader>
          <CardDescription>
            <InfoTooltip content="Total number of mining projects currently tracked across all stages from exploration to production">
              Projects
            </InfoTooltip>
          </CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {stats.totalProjects.toLocaleString()}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {stats.projectsGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
              {stats.projectsGrowth >= 0 ? '+' : ''}{stats.projectsGrowth.toFixed(1)}%
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            Trending up this month <IconTrendingUp className="size-4" />
          </div>
          <div className="text-muted-foreground">
            Active mining projects tracked
          </div>
        </CardFooter>
        </Card>
      </ContextMenuChat>
      <ContextMenuChat
        data={{
          totalCompanies: stats.totalCompanies,
          growth: stats.companiesGrowth,
          type: 'companies'
        }}
        dataType="metric"
        context="Mining companies in database"
      >
        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
        <CardHeader>
          <CardDescription>
            <InfoTooltip content="Total number of mining companies tracked in the database">
              Companies
            </InfoTooltip>
          </CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {stats.totalCompanies.toLocaleString()}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {stats.companiesGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
              {stats.companiesGrowth >= 0 ? '+' : ''}{Math.abs(stats.companiesGrowth).toFixed(1)}%
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {stats.companiesGrowth >= 0 ? 'Trending up this period' : 'Trending down this period'} {stats.companiesGrowth >= 0 ? <IconTrendingUp className="size-4" /> : <IconTrendingDown className="size-4" />}
          </div>
          <div className="text-muted-foreground">
            Mining companies tracked
          </div>
        </CardFooter>
        </Card>
      </ContextMenuChat>
      <ContextMenuChat
        data={{
          totalNews: stats.totalNews,
          growth: stats.newsGrowth,
          type: 'news'
        }}
        dataType="metric"
        context="Mining industry news articles"
      >
        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
          <CardHeader>
            <CardDescription>
              <InfoTooltip content="Total number of mining industry news articles tracked from various sources">
                News Articles
              </InfoTooltip>
          </CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {stats.totalNews.toLocaleString()}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {stats.newsGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
              {stats.newsGrowth >= 0 ? '+' : ''}{stats.newsGrowth.toFixed(1)}%
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {stats.newsGrowth >= 0 ? 'Strong news activity' : 'News activity down'} {stats.newsGrowth >= 0 ? <IconTrendingUp className="size-4" /> : <IconTrendingDown className="size-4" />}
          </div>
          <div className="text-muted-foreground">Industry news tracked</div>
        </CardFooter>
        </Card>
      </ContextMenuChat>
      <ContextMenuChat
        data={{
          totalWatchlisted: stats.watchlistedItems,
          growth: stats.watchlistedGrowth,
          type: 'watchlist'
        }}
        dataType="metric"
        context="Watchlisted projects, companies, and news"
      >
        <Card className="@container/card from-primary/5 to-card dark:bg-card bg-gradient-to-t shadow-xs">
          <CardHeader>
            <CardDescription>
              <InfoTooltip content="Total number of projects, companies, and news articles added to your watchlist">
                Watchlisted Items
            </InfoTooltip>
          </CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {stats.watchlistedItems.toLocaleString()}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              {stats.watchlistedGrowth >= 0 ? <IconTrendingUp /> : <IconTrendingDown />}
              {stats.watchlistedGrowth >= 0 ? '+' : ''}{stats.watchlistedGrowth.toFixed(1)}%
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            Your tracked items
          </div>
          <div className="text-muted-foreground">Projects, companies & news</div>
        </CardFooter>
        </Card>
      </ContextMenuChat>
    </div>
  )
}
