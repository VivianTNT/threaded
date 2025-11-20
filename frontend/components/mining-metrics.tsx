"use client"

import * as React from "react"
import { TrendingUp, TrendingDown, Minus, Building2, FileText, DollarSign, Mountain, Leaf, Activity, Users } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { InfoTooltip } from "@/components/ui/info-tooltip"

interface MetricCardProps {
  title: string
  value: string | number
  change?: number
  changeLabel?: string
  subtitle?: string
  icon?: React.ReactNode
  trend?: "up" | "down" | "neutral"
  tooltip?: string
}

function MetricCard({ title, value, change, changeLabel, subtitle, icon, trend, tooltip }: MetricCardProps) {
  const getTrendIcon = () => {
    if (!trend || trend === "neutral") return <Minus className="h-4 w-4" />
    return trend === "up" ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />
  }

  const getTrendColor = () => {
    if (!trend || trend === "neutral") return "text-gray-500"
    return trend === "up" ? "text-green-600" : "text-red-600"
  }

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardContent className="p-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <p className="text-sm font-medium text-muted-foreground">
              {tooltip ? (
                <InfoTooltip content={tooltip}>
                  {title}
                </InfoTooltip>
              ) : (
                title
              )}
            </p>
            <p className="text-2xl font-semibold tracking-tight">{value}</p>
            {change !== undefined && (
              <div className={cn("flex items-center gap-1 text-sm", getTrendColor())}>
                {getTrendIcon()}
                <span className="font-medium">
                  {change > 0 ? "+" : ""}{change}%
                </span>
                {changeLabel && <span className="text-muted-foreground">• {changeLabel}</span>}
              </div>
            )}
            {subtitle && (
              <p className="text-xs text-muted-foreground">{subtitle}</p>
            )}
          </div>
          {icon && (
            <div className="rounded-lg bg-muted p-2">
              {icon}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export function MiningMetrics() {
  // In a real app, these would come from an API
  const metrics = [
    {
      title: "Total Projects",
      value: "88,307",
      change: 12.5,
      changeLabel: "MoM",
      subtitle: "Active mining projects worldwide",
      icon: <Mountain className="h-5 w-5 text-blue-600" />,
      trend: "up" as const,
      tooltip: "Total number of mining projects tracked across all stages from exploration to production, including greenfield and brownfield developments"
    },
    {
      title: "Active Companies",
      value: "84,063",
      change: -2.3,
      changeLabel: "MoM",
      subtitle: "Companies with recent filings",
      icon: <Building2 className="h-5 w-5 text-purple-600" />,
      trend: "down" as const,
      tooltip: "Mining companies that have filed at least one regulatory report in the past 90 days, indicating active operations or development"
    },
    {
      title: "Recent Filings",
      value: "224,172",
      change: 8.7,
      changeLabel: "vs last 30d",
      subtitle: "Technical reports & updates",
      icon: <FileText className="h-5 w-5 text-orange-600" />,
      trend: "up" as const,
      tooltip: "Technical reports filed in the last 30 days including NI 43-101, JORC, SAMREC, and other regulatory standards"
    },
    {
      title: "Average IRR",
      value: "37.46%",
      change: 3.2,
      changeLabel: "QoQ",
      subtitle: "Across all active projects",
      icon: <TrendingUp className="h-5 w-5 text-green-600" />,
      trend: "up" as const,
      tooltip: "Internal Rate of Return - the discount rate that makes the net present value of all cash flows equal to zero, indicating project profitability"
    },
    {
      title: "Total NPV",
      value: "$1.25T",
      change: 15.8,
      changeLabel: "YoY",
      subtitle: "Aggregate project value",
      icon: <DollarSign className="h-5 w-5 text-emerald-600" />,
      trend: "up" as const,
      tooltip: "Net Present Value - the total present value of all future cash flows from all tracked projects, discounted at standard industry rates"
    },
    {
      title: "New Discoveries",
      value: "1,234",
      change: -20.1,
      changeLabel: "vs last month",
      subtitle: "Newly announced projects",
      icon: <Activity className="h-5 w-5 text-indigo-600" />,
      trend: "down" as const,
      tooltip: "Newly announced mining projects and discoveries reported in the past 30 days across all commodities and jurisdictions"
    },
    {
      title: "ESG Compliance",
      value: "45.67%",
      change: 12.5,
      changeLabel: "YoY",
      subtitle: "Projects rated A or B",
      icon: <Leaf className="h-5 w-5 text-green-600" />,
      trend: "up" as const,
      tooltip: "Percentage of projects meeting high Environmental, Social, and Governance standards with ratings of A or B from recognized ESG rating agencies"
    },
    {
      title: "Commodity Index",
      value: "104.5",
      change: 4.5,
      changeLabel: "WoW",
      subtitle: "Weighted basket price index",
      icon: <Activity className="h-5 w-5 text-cyan-600" />,
      trend: "up" as const,
      tooltip: "Weighted average price index of key mining commodities including gold, copper, lithium, and rare earth elements relative to baseline 100"
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      {metrics.map((metric, index) => (
        <MetricCard key={index} {...metric} />
      ))}
    </div>
  )
} 