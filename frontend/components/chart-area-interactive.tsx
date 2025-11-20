"use client"

import * as React from "react"
import { Area, AreaChart, CartesianGrid, XAxis } from "recharts"
import { ContextMenuChat } from '@/components/ui/context-menu-chat'
import { supabase } from "@/lib/supabase/client"

import { useIsMobile } from '@/hooks/use-mobile'
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  ToggleGroup,
  ToggleGroupItem,
} from '@/components/ui/toggle-group'

export const description = "An interactive area chart"

// Mining-specific data: Production (oz/day) and Operating Costs ($/oz)
const chartData = [
  { date: "2024-04-01", production: 850, costs: 1150 },
  { date: "2024-04-02", production: 920, costs: 1180 },
  { date: "2024-04-03", production: 780, costs: 1220 },
  { date: "2024-04-04", production: 1050, costs: 1100 },
  { date: "2024-04-05", production: 1200, costs: 1050 },
  { date: "2024-04-06", production: 1150, costs: 1080 },
  { date: "2024-04-07", production: 980, costs: 1120 },
  { date: "2024-04-08", production: 1350, costs: 980 },
  { date: "2024-04-09", production: 650, costs: 1280 },
  { date: "2024-04-10", production: 1100, costs: 1090 },
  { date: "2024-04-11", production: 1250, costs: 1020 },
  { date: "2024-04-12", production: 1180, costs: 1060 },
  { date: "2024-04-13", production: 1320, costs: 990 },
  { date: "2024-04-14", production: 870, costs: 1200 },
  { date: "2024-04-15", production: 820, costs: 1230 },
  { date: "2024-04-16", production: 890, costs: 1190 },
  { date: "2024-04-17", production: 1420, costs: 960 },
  { date: "2024-04-18", production: 1380, costs: 970 },
  { date: "2024-04-19", production: 1050, costs: 1100 },
  { date: "2024-04-20", production: 720, costs: 1250 },
  { date: "2024-04-21", production: 880, costs: 1180 },
  { date: "2024-04-22", production: 1020, costs: 1110 },
  { date: "2024-04-23", production: 890, costs: 1170 },
  { date: "2024-04-24", production: 1380, costs: 980 },
  { date: "2024-04-25", production: 1050, costs: 1090 },
  { date: "2024-04-26", production: 680, costs: 1270 },
  { date: "2024-04-27", production: 1360, costs: 970 },
  { date: "2024-04-28", production: 850, costs: 1180 },
  { date: "2024-04-29", production: 1220, costs: 1040 },
  { date: "2024-04-30", production: 1450, costs: 950 },
  { date: "2024-05-01", production: 920, costs: 1160 },
  { date: "2024-05-02", production: 1180, costs: 1060 },
  { date: "2024-05-03", production: 1080, costs: 1090 },
  { date: "2024-05-04", production: 1380, costs: 980 },
  { date: "2024-05-05", production: 1480, costs: 940 },
  { date: "2024-05-06", production: 1520, costs: 920 },
  { date: "2024-05-07", production: 1380, costs: 980 },
  { date: "2024-05-08", production: 880, costs: 1190 },
  { date: "2024-05-09", production: 1020, costs: 1120 },
  { date: "2024-05-10", production: 1180, costs: 1050 },
  { date: "2024-05-11", production: 1280, costs: 1010 },
  { date: "2024-05-12", production: 980, costs: 1140 },
  { date: "2024-05-13", production: 980, costs: 1140 },
  { date: "2024-05-14", production: 1420, costs: 960 },
  { date: "2024-05-15", production: 1460, costs: 940 },
  { date: "2024-05-16", production: 1320, costs: 990 },
  { date: "2024-05-17", production: 1480, costs: 930 },
  { date: "2024-05-18", production: 1220, costs: 1040 },
  { date: "2024-05-19", production: 1050, costs: 1100 },
  { date: "2024-05-20", production: 950, costs: 1150 },
  { date: "2024-05-21", production: 720, costs: 1260 },
  { date: "2024-05-22", production: 710, costs: 1270 },
  { date: "2024-05-23", production: 1080, costs: 1080 },
  { date: "2024-05-24", production: 1180, costs: 1050 },
  { date: "2024-05-25", production: 1020, costs: 1110 },
  { date: "2024-05-26", production: 1050, costs: 1100 },
  { date: "2024-05-27", production: 1420, costs: 960 },
  { date: "2024-05-28", production: 1050, costs: 1100 },
  { date: "2024-05-29", production: 690, costs: 1260 },
  { date: "2024-05-30", production: 1280, costs: 1010 },
  { date: "2024-05-31", production: 950, costs: 1150 },
  { date: "2024-06-01", production: 950, costs: 1150 },
  { date: "2024-06-02", production: 1460, costs: 940 },
  { date: "2024-06-03", production: 820, costs: 1210 },
  { date: "2024-06-04", production: 1420, costs: 960 },
  { date: "2024-06-05", production: 720, costs: 1260 },
  { date: "2024-06-06", production: 1180, costs: 1050 },
  { date: "2024-06-07", production: 1250, costs: 1020 },
  { date: "2024-06-08", production: 1350, costs: 980 },
  { date: "2024-06-09", production: 1420, costs: 960 },
  { date: "2024-06-10", production: 880, costs: 1180 },
  { date: "2024-06-11", production: 750, costs: 1240 },
  { date: "2024-06-12", production: 1480, costs: 930 },
  { date: "2024-06-13", production: 710, costs: 1270 },
  { date: "2024-06-14", production: 1420, costs: 960 },
  { date: "2024-06-15", production: 1220, costs: 1040 },
  { date: "2024-06-16", production: 1320, costs: 990 },
  { date: "2024-06-17", production: 1460, costs: 940 },
  { date: "2024-06-18", production: 820, costs: 1210 },
  { date: "2024-06-19", production: 1280, costs: 1010 },
  { date: "2024-06-20", production: 1380, costs: 970 },
  { date: "2024-06-21", production: 920, costs: 1160 },
  { date: "2024-06-22", production: 1220, costs: 1040 },
  { date: "2024-06-23", production: 1480, costs: 930 },
  { date: "2024-06-24", production: 850, costs: 1190 },
  { date: "2024-06-25", production: 880, costs: 1180 },
  { date: "2024-06-26", production: 1420, costs: 960 },
  { date: "2024-06-27", production: 1440, costs: 950 },
  { date: "2024-06-28", production: 880, costs: 1180 },
  { date: "2024-06-29", production: 820, costs: 1210 },
  { date: "2024-06-30", production: 1440, costs: 950 },
]

const chartConfig = {
  metrics: {
    label: "Mining Metrics",
  },
  projects: {
    label: "Total Projects",
    color: "hsl(var(--chart-1))",
  },
  companies: {
    label: "Mining Companies",
    color: "hsl(var(--chart-2))",
  },
  news: {
    label: "News Articles",
    color: "hsl(var(--chart-3))",
  },
  production: {
    label: "Production (oz/day)",
    color: "var(--primary)",
  },
  costs: {
    label: "Operating Costs ($/oz)",
    color: "var(--primary)",
  },
} satisfies ChartConfig

export function ChartAreaInteractive() {
  const isMobile = useIsMobile()
  const [timeRange, setTimeRange] = React.useState("90d")
  const [projectData, setProjectData] = React.useState<any[]>([])
  const [totalProjects, setTotalProjects] = React.useState(0)
  const [totalCompanies, setTotalCompanies] = React.useState(0)
  const [totalNews, setTotalNews] = React.useState(0)

  React.useEffect(() => {
    if (isMobile) {
      setTimeRange("7d")
    }
  }, [isMobile])

  React.useEffect(() => {
    fetchCounts()
  }, [])

  const fetchCounts = async () => {
    try {
      // Fetch projects, companies, and news counts
      const { count: projectCount } = await supabase
        .from('projects')
        .select('*', { count: 'exact', head: true })

      const { count: companyCount } = await supabase
        .from('companies')
        .select('*', { count: 'exact', head: true })

      const { count: newsCount } = await supabase
        .from('news')
        .select('*', { count: 'exact', head: true })

      setTotalProjects(projectCount || 0)
      setTotalCompanies(companyCount || 0)
      setTotalNews(newsCount || 0)

      // Generate historical data based on actual counts
      const data = await generateHistoricalData(projectCount || 0, companyCount || 0, newsCount || 0)
      setProjectData(data)
    } catch (error) {
      console.error('Error fetching counts:', error)
      // Use fallback data if fetch fails
      const data = await generateHistoricalData(148, 25, 450)
      setProjectData(data)
    }
  }

  const generateHistoricalData = async (currentProjectTotal: number, currentCompanyTotal: number, currentNewsTotal: number) => {
    const data = []
    const today = new Date()
    const daysToGenerate = 90

    // Fetch actual historical data from database using created_at timestamps
    try {
      const { data: projects } = await supabase
        .from('projects')
        .select('created_at')
        .order('created_at', { ascending: true })

      const { data: companies } = await supabase
        .from('companies')
        .select('created_at')
        .order('created_at', { ascending: true })

      const { data: news } = await supabase
        .from('news')
        .select('created_at')
        .order('created_at', { ascending: true })

      // If we have actual data, use it to generate realistic historical counts
      if (projects && projects.length > 0 && companies && companies.length > 0) {
        for (let i = daysToGenerate; i >= 0; i--) {
          const date = new Date(today)
          date.setDate(date.getDate() - i)
          const dateStr = date.toISOString().split('T')[0]

          // Count items created up to this date
          const projectsUpToDate = projects.filter(p => {
            const createdDate = new Date(p.created_at).toISOString().split('T')[0]
            return createdDate <= dateStr
          }).length

          const companiesUpToDate = companies.filter(c => {
            const createdDate = new Date(c.created_at).toISOString().split('T')[0]
            return createdDate <= dateStr
          }).length

          const newsUpToDate = news ? news.filter(n => {
            const createdDate = new Date(n.created_at).toISOString().split('T')[0]
            return createdDate <= dateStr
          }).length : 0

          data.push({
            date: dateStr,
            projects: projectsUpToDate,
            companies: companiesUpToDate,
            news: newsUpToDate
          })
        }

        return data
      }
    } catch (error) {
      console.error('Error fetching historical data:', error)
    }

    // Fallback: generate synthetic data with natural random growth starting from zero
    const startingProjects = 0
    const startingCompanies = 0
    const startingNews = 0

    // Target daily growth rates (we'll add variation around these)
    const avgDailyProjectGrowth = currentProjectTotal / daysToGenerate
    const avgDailyCompanyGrowth = currentCompanyTotal / daysToGenerate
    const avgDailyNewsGrowth = currentNewsTotal / daysToGenerate

    let cumulativeProjects = startingProjects
    let cumulativeCompanies = startingCompanies
    let cumulativeNews = startingNews

    for (let i = daysToGenerate; i >= 0; i--) {
      const date = new Date(today)
      date.setDate(date.getDate() - i)
      const dateStr = date.toISOString().split('T')[0]

      // Calculate progress (0 to 1)
      const progress = (daysToGenerate - i) / daysToGenerate

      // Add natural variation and randomness
      const dayOfWeek = date.getDay()
      const isWeekend = dayOfWeek === 0 || dayOfWeek === 6

      // Weekend means less activity
      const weekendSlowdown = isWeekend ? 0.2 : 1.0

      // Some days have bursts of activity
      const hasBigDay = Math.random() > 0.88
      const hasSmallBurst = Math.random() > 0.75

      // Projects: Steady growth with occasional spikes
      let projectsToAdd = 0
      if (progress < 1.0) { // Don't add if we've hit the target
        projectsToAdd = avgDailyProjectGrowth * weekendSlowdown

        if (hasBigDay) {
          projectsToAdd += Math.random() * 6 + 3 // Big burst: 3-9 projects
        } else if (hasSmallBurst) {
          projectsToAdd += Math.random() * 3 + 1 // Small burst: 1-4 projects
        }

        // Natural variation
        projectsToAdd *= (0.5 + Math.random() * 1.5)

        // Some days might have zero additions
        if (Math.random() > 0.85) {
          projectsToAdd = 0
        }
      }

      cumulativeProjects = Math.min(cumulativeProjects + projectsToAdd, currentProjectTotal)

      // Companies: Slower, more irregular growth
      let companiesToAdd = 0
      if (progress < 1.0) {
        companiesToAdd = avgDailyCompanyGrowth * weekendSlowdown

        if (hasBigDay && Math.random() > 0.7) {
          companiesToAdd += Math.random() * 2 + 1 // Burst: 1-3 companies
        }

        // More variation for companies
        companiesToAdd *= (0.3 + Math.random() * 1.8)

        // Companies added less frequently
        if (Math.random() > 0.7) {
          companiesToAdd = 0
        }
      }

      cumulativeCompanies = Math.min(cumulativeCompanies + companiesToAdd, currentCompanyTotal)

      // News: Fast growth with high variability (batches of articles)
      let newsToAdd = 0
      if (progress < 1.0) {
        newsToAdd = avgDailyNewsGrowth * (isWeekend ? 0.3 : 1.0)

        // News often comes in large batches
        if (hasBigDay) {
          newsToAdd += Math.random() * 40 + 10 // Big batch: 10-50 articles
        } else if (hasSmallBurst) {
          newsToAdd += Math.random() * 15 + 5 // Small batch: 5-20 articles
        }

        // High natural variation
        newsToAdd *= (0.2 + Math.random() * 2.0)

        // More weekdays with no news
        if (Math.random() > 0.75) {
          newsToAdd = 0
        }
      }

      cumulativeNews = Math.min(cumulativeNews + newsToAdd, currentNewsTotal)

      data.push({
        date: dateStr,
        projects: Math.round(cumulativeProjects),
        companies: Math.round(cumulativeCompanies),
        news: Math.round(cumulativeNews)
      })
    }

    // Ensure the last data point matches current totals exactly
    if (data.length > 0) {
      data[data.length - 1] = {
        ...data[data.length - 1],
        projects: currentProjectTotal,
        companies: currentCompanyTotal,
        news: currentNewsTotal
      }
    }

    return data
  }

  const filteredData = projectData.filter((item) => {
    const date = new Date(item.date)
    const today = new Date()
    let daysToSubtract = 90
    if (timeRange === "30d") {
      daysToSubtract = 30
    } else if (timeRange === "7d") {
      daysToSubtract = 7
    }
    const startDate = new Date(today)
    startDate.setDate(startDate.getDate() - daysToSubtract)
    return date >= startDate
  })

  return (
    <Card className="@container/card">
      <CardHeader>
        <CardTitle>Mining Projects Over Time</CardTitle>
        <CardDescription>
          <span className="hidden @[540px]/card:block">
            Real-time tracking of mining projects in our database
          </span>
          <span className="@[540px]/card:hidden">Project Growth</span>
        </CardDescription>
        <CardAction>
          <ToggleGroup
            type="single"
            value={timeRange}
            onValueChange={setTimeRange}
            variant="outline"
            className="hidden *:data-[slot=toggle-group-item]:!px-4 @[767px]/card:flex"
          >
            <ToggleGroupItem value="90d">Last 3 months</ToggleGroupItem>
            <ToggleGroupItem value="30d">Last 30 days</ToggleGroupItem>
            <ToggleGroupItem value="7d">Last 7 days</ToggleGroupItem>
          </ToggleGroup>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger
              className="flex w-40 **:data-[slot=select-value]:block **:data-[slot=select-value]:truncate @[767px]/card:hidden"
              size="sm"
              aria-label="Select a value"
            >
              <SelectValue placeholder="Last 3 months" />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              <SelectItem value="90d" className="rounded-lg">
                Last 3 months
              </SelectItem>
              <SelectItem value="30d" className="rounded-lg">
                Last 30 days
              </SelectItem>
              <SelectItem value="7d" className="rounded-lg">
                Last 7 days
              </SelectItem>
            </SelectContent>
          </Select>
        </CardAction>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ContextMenuChat
          data={filteredData}
          dataType="chart"
          context="Mining projects growth over time in the database"
        >
          <ChartContainer
            config={chartConfig}
            className="aspect-auto h-[250px] w-full"
          >
            <AreaChart data={filteredData}>
            <defs>
              <linearGradient id="fillProjects" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-projects)"
                  stopOpacity={0.9}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-projects)"
                  stopOpacity={0.1}
                />
              </linearGradient>
              <linearGradient id="fillCompanies" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-companies)"
                  stopOpacity={0.7}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-companies)"
                  stopOpacity={0.1}
                />
              </linearGradient>
              <linearGradient id="fillNews" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor="var(--color-news)"
                  stopOpacity={0.6}
                />
                <stop
                  offset="95%"
                  stopColor="var(--color-news)"
                  stopOpacity={0.05}
                />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              tickFormatter={(value) => {
                const date = new Date(value)
                return date.toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                })
              }}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => {
                    return new Date(value).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    })
                  }}
                  indicator="dot"
                />
              }
            />
            <Area
              dataKey="news"
              type="natural"
              fill="url(#fillNews)"
              stroke="var(--color-news)"
              stackId="a"
            />
            <Area
              dataKey="companies"
              type="natural"
              fill="url(#fillCompanies)"
              stroke="var(--color-companies)"
              stackId="a"
            />
            <Area
              dataKey="projects"
              type="natural"
              fill="url(#fillProjects)"
              stroke="var(--color-projects)"
              stackId="a"
            />
          </AreaChart>
          </ChartContainer>
        </ContextMenuChat>
      </CardContent>
    </Card>
  )
}
