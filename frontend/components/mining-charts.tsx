"use client"

import * as React from "react"
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { TrendingUp, BarChart3, PieChart, LineChart as LineChartIcon, Activity } from "lucide-react"

type ChartType = "commodity-prices" | "npv-by-stage" | "new-projects" | "irr-distribution" | "regional-production"
type TimePeriod = "7d" | "30d" | "90d" | "ytd" | "12m"

const chartConfig = {
  lithium: {
    label: "Lithium",
    color: "hsl(var(--chart-1))",
  },
  copper: {
    label: "Copper",
    color: "hsl(var(--chart-2))",
  },
  nickel: {
    label: "Nickel",
    color: "hsl(var(--chart-3))",
  },
  gold: {
    label: "Gold",
    color: "hsl(var(--chart-4))",
  },
  value: {
    label: "Value",
    color: "hsl(var(--chart-5))",
  },
}

// Generate realistic mining data
const generateCommodityPriceData = (period: TimePeriod) => {
  const days = period === "7d" ? 7 : period === "30d" ? 30 : period === "90d" ? 90 : period === "ytd" ? 180 : 365
  const data = []
  const baseDate = new Date()
  
  for (let i = days; i >= 0; i -= Math.ceil(days / 30)) {
    const date = new Date(baseDate)
    date.setDate(date.getDate() - i)
    
    data.push({
      date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      lithium: 12000 + Math.random() * 3000 + (days - i) * 20,
      copper: 8500 + Math.random() * 1000 + (days - i) * 5,
      nickel: 16000 + Math.random() * 2000 + (days - i) * 10,
      gold: 1800 + Math.random() * 100 + (days - i) * 0.5,
    })
  }
  
  return data
}

const npvByStageData = [
  { stage: "Exploration", value: 125, count: 3240 },
  { stage: "PEA", value: 340, count: 1820 },
  { stage: "PFS", value: 580, count: 980 },
  { stage: "DFS", value: 890, count: 540 },
  { stage: "Development", value: 1240, count: 320 },
  { stage: "Production", value: 2100, count: 280 },
]

const newProjectsData = [
  { month: "Jan", projects: 145 },
  { month: "Feb", projects: 132 },
  { month: "Mar", projects: 178 },
  { month: "Apr", projects: 156 },
  { month: "May", projects: 189 },
  { month: "Jun", projects: 203 },
  { month: "Jul", projects: 178 },
  { month: "Aug", projects: 165 },
  { month: "Sep", projects: 142 },
  { month: "Oct", projects: 198 },
  { month: "Nov", projects: 176 },
  { month: "Dec", projects: 234 },
]

const irrDistributionData = [
  { range: "0-10%", count: 120 },
  { range: "10-20%", count: 340 },
  { range: "20-30%", count: 580 },
  { range: "30-40%", count: 420 },
  { range: "40-50%", count: 280 },
  { range: "50%+", count: 160 },
]

const regionalProductionData = [
  { region: "Australia", production: 42.5 },
  { region: "Chile", production: 28.3 },
  { region: "China", production: 35.7 },
  { region: "Peru", production: 22.1 },
  { region: "Canada", production: 18.9 },
  { region: "USA", production: 15.2 },
  { region: "DRC", production: 12.8 },
  { region: "Others", production: 24.5 },
]

export function MiningCharts() {
  const [selectedChart, setSelectedChart] = React.useState<ChartType>("commodity-prices")
  const [timePeriod, setTimePeriod] = React.useState<TimePeriod>("30d")
  const [commodityData, setCommodityData] = React.useState(generateCommodityPriceData("30d"))

  React.useEffect(() => {
    setCommodityData(generateCommodityPriceData(timePeriod))
  }, [timePeriod])

  const chartButtons = [
    { id: "commodity-prices" as ChartType, label: "Commodity Prices", icon: LineChartIcon },
    { id: "npv-by-stage" as ChartType, label: "NPV by Stage", icon: BarChart3 },
    { id: "new-projects" as ChartType, label: "New Projects", icon: TrendingUp },
    { id: "irr-distribution" as ChartType, label: "IRR Distribution", icon: BarChart3 },
    { id: "regional-production" as ChartType, label: "Regional Production", icon: PieChart },
  ]

  const renderChart = () => {
    switch (selectedChart) {
      case "commodity-prices":
        return (
          <ChartContainer config={chartConfig} className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={commodityData} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="date" 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                />
                <YAxis 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                  label={{ value: 'Price ($/t)', angle: -90, position: 'insideLeft' }}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line 
                  type="monotone" 
                  dataKey="lithium" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2}
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="copper" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  dot={false}
                />
                <Line 
                  type="monotone" 
                  dataKey="nickel" 
                  stroke="hsl(var(--chart-3))" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "npv-by-stage":
        return (
          <ChartContainer config={chartConfig} className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={npvByStageData} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="stage" 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                />
                <YAxis 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                  label={{ value: 'Total NPV ($B)', angle: -90, position: 'insideLeft' }}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar 
                  dataKey="value" 
                  fill="hsl(var(--chart-1))"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "new-projects":
        return (
          <ChartContainer config={chartConfig} className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={newProjectsData} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="month" 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                />
                <YAxis 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                  label={{ value: 'New Projects', angle: -90, position: 'insideLeft' }}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Area 
                  type="monotone" 
                  dataKey="projects" 
                  stroke="hsl(var(--chart-2))"
                  fill="hsl(var(--chart-2))"
                  fillOpacity={0.2}
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "irr-distribution":
        return (
          <ChartContainer config={chartConfig} className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={irrDistributionData} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="range" 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                />
                <YAxis 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                  label={{ value: 'Number of Projects', angle: -90, position: 'insideLeft' }}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar 
                  dataKey="count" 
                  fill="hsl(var(--chart-3))"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      case "regional-production":
        return (
          <ChartContainer config={chartConfig} className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart 
                data={regionalProductionData} 
                layout="horizontal"
                margin={{ top: 5, right: 10, left: 60, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  type="number"
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                  label={{ value: 'Production (Mt)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  type="category"
                  dataKey="region" 
                  className="text-xs"
                  tick={{ fill: 'hsl(var(--foreground))' }}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar 
                  dataKey="production" 
                  fill="hsl(var(--chart-4))"
                  radius={[0, 4, 4, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </ChartContainer>
        )

      default:
        return null
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Mining Intelligence Analytics</CardTitle>
            <CardDescription>
              Real-time market data and project insights
            </CardDescription>
          </div>
          <Select value={timePeriod} onValueChange={(value) => setTimePeriod(value as TimePeriod)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
              <SelectItem value="ytd">Year to date</SelectItem>
              <SelectItem value="12m">Last 12 months</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-2">
          {chartButtons.map((button) => (
            <Button
              key={button.id}
              variant={selectedChart === button.id ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedChart(button.id)}
              className="gap-2"
            >
              <button.icon className="h-4 w-4" />
              {button.label}
            </Button>
          ))}
        </div>
        
        <div className="w-full">
          {renderChart()}
        </div>

        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <span>Source: Lithos Intelligence Platform</span>
          <span>Last updated: {new Date().toLocaleString()}</span>
        </div>
      </CardContent>
    </Card>
  )
} 