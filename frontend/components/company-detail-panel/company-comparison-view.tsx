"use client"

import * as React from "react"
import { Download, FileText } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Company } from "@/lib/hooks/use-companies"
import { cn } from "@/lib/utils"

interface CompanyComparisonViewProps {
  companies: Company[]
}

interface ComparisonMetric {
  label: string
  key: keyof Company | string
  format: (value: any, company?: Company) => string
}

export function CompanyComparisonView({ companies }: CompanyComparisonViewProps) {
  const metrics: ComparisonMetric[] = [
    {
      label: "Name",
      key: "name",
      format: (value) => value || "N/A",
    },
    {
      label: "Ticker",
      key: "ticker",
      format: (value) => value || "N/A",
    },
    {
      label: "Exchange",
      key: "exchange",
      format: (value) => value || "N/A",
    },
    {
      label: "Country",
      key: "country",
      format: (value) => value || "N/A",
    },
    {
      label: "Market Cap",
      key: "market_cap",
      format: (value) => {
        if (!value) return "N/A"
        if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
        if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`
        return `$${value.toLocaleString()}`
      },
    },
  ]

  return (
    <div className="p-6 space-y-6">
      {/* Header Actions */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">
          Company Comparison
          <span className="text-sm font-normal text-muted-foreground ml-2">
            {companies.length} companies selected
          </span>
        </h2>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Export CSV
          </Button>
          <Button variant="outline" size="sm">
            <FileText className="mr-2 h-4 w-4" />
            Export PDF
          </Button>
        </div>
      </div>

      {/* Company Cards */}
      <div className="grid grid-cols-2 gap-4">
        {companies.map((company) => (
          <Card key={company.id} className="relative">
            <CardHeader>
              <CardTitle className="text-lg">{company.name}</CardTitle>
              <div className="text-sm text-muted-foreground">
                {company.ticker && company.exchange
                  ? `${company.ticker} • ${company.exchange}`
                  : company.ticker || company.exchange || 'N/A'}
              </div>
              <div className="flex items-center gap-2 mt-2">
                {company.country && (
                  <Badge variant="outline">{company.country}</Badge>
                )}
                {company.market_cap && (
                  <Badge variant="secondary">
                    {company.market_cap >= 1e9
                      ? `$${(company.market_cap / 1e9).toFixed(2)}B`
                      : `$${(company.market_cap / 1e6).toFixed(2)}M`}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Key Information */}
              <div className="space-y-2 text-sm">
                {company.description && (
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Description</div>
                    <div className="line-clamp-3">{company.description}</div>
                  </div>
                )}
                {company.website && (
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Website</div>
                    <a
                      href={company.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:underline truncate block"
                    >
                      {company.website}
                    </a>
                  </div>
                )}
              </div>

              <div className="text-xs text-muted-foreground">
                Last updated: {new Date(company.updated_at).toLocaleDateString()}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Separator />

      {/* Comparison Table */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Detailed Comparison</h3>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[200px]">Metric</TableHead>
                {companies.map((company) => (
                  <TableHead key={company.id} className="text-center">
                    {company.name}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {metrics.map((metric) => (
                <TableRow key={metric.key}>
                  <TableCell className="font-medium">{metric.label}</TableCell>
                  {companies.map((company) => {
                    const value = company[metric.key as keyof Company]
                    const formattedValue = metric.format(value, company)

                    return (
                      <TableCell key={company.id} className="text-center">
                        <div>
                          {metric.key === "ticker" || metric.key === "exchange" ? (
                            <Badge variant="outline">{formattedValue}</Badge>
                          ) : (
                            formattedValue
                          )}
                        </div>
                      </TableCell>
                    )
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}
