"use client"

import * as React from "react"
import { Download, FileText, TrendingDown, TrendingUp } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { MiningProject } from "@/lib/types/mining-project"
import { cn } from "@/lib/utils"

interface ProjectComparisonViewProps {
  projects: MiningProject[]
}

interface ComparisonMetric {
  label: string
  key: keyof MiningProject | string
  format: (value: any, project?: MiningProject) => string
  showDiff?: boolean
}

export function ProjectComparisonView({ projects }: ProjectComparisonViewProps) {
  const [showESGScore, setShowESGScore] = React.useState(true)
  const [showJurisdictionRisk, setShowJurisdictionRisk] = React.useState(true)
  const [showFinancingStatus, setShowFinancingStatus] = React.useState(false)

  const metrics: ComparisonMetric[] = [
    {
      label: "NPV (USD M)",
      key: "npv",
      format: (value) => value !== null && value !== undefined ? `$${value.toFixed(0)}M` : "N/A",
      showDiff: true,
    },
    {
      label: "IRR (%)",
      key: "irr",
      format: (value) => value !== null && value !== undefined ? `${value.toFixed(1)}%` : "N/A",
      showDiff: true,
    },
    {
      label: "CAPEX (USD M)",
      key: "capex",
      format: (value) => value !== null && value !== undefined ? `$${value.toFixed(0)}M` : "N/A",
      showDiff: true,
    },
    {
      label: "Resource Grade",
      key: "resourceGrade",
      format: (value, project) =>
        value ? `${value}${project?.gradeUnit || ""}` : "N/A",
    },
    {
      label: "Stage",
      key: "stage",
      format: (value) => value || "Unknown",
    },
    {
      label: "ESG Score",
      key: "esgScore",
      format: (value) => value || "N/A",
    },
    {
      label: "Jurisdiction Risk",
      key: "riskLevel",
      format: (value) => value || "Medium",
    },
  ]

  const getValueDiff = (value1: number, value2: number) => {
    const diff = ((value1 - value2) / value2) * 100
    return diff
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "Low": return "text-green-600"
      case "Medium": return "text-yellow-600"
      case "High": return "text-orange-600"
      case "Very High": return "text-red-600"
      default: return ""
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header Actions */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">
          Project Comparison
          <span className="text-sm font-normal text-muted-foreground ml-2">
            {projects.length} projects selected
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

      {/* Toggle Switches */}
      <div className="flex gap-6">
        <div className="flex items-center space-x-2">
          <Switch
            id="esg"
            checked={showESGScore}
            onCheckedChange={setShowESGScore}
          />
          <Label htmlFor="esg">ESG Score</Label>
        </div>
        <div className="flex items-center space-x-2">
          <Switch
            id="risk"
            checked={showJurisdictionRisk}
            onCheckedChange={setShowJurisdictionRisk}
          />
          <Label htmlFor="risk">Jurisdiction Risk</Label>
        </div>
        <div className="flex items-center space-x-2">
          <Switch
            id="financing"
            checked={showFinancingStatus}
            onCheckedChange={setShowFinancingStatus}
          />
          <Label htmlFor="financing">Financing Status</Label>
        </div>
      </div>

      {/* Project Cards */}
      <div className="grid grid-cols-2 gap-4">
        {projects.map((project, index) => (
          <Card key={project.id} className="relative">
            <CardHeader>
              <CardTitle className="text-lg">{project.name}</CardTitle>
              <div className="text-sm text-muted-foreground">
                {project.company || 'Unknown'}
              </div>
              <div className="flex items-center gap-2 mt-2">
                <Badge variant="outline">{project.location || 'N/A'}</Badge>
                <Badge className={cn("text-xs", getRiskColor(project.riskLevel || 'Medium'))}>
                  {project.riskLevel || 'Medium'} Risk
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Key Metrics */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">NPV</div>
                  <div className="text-xl font-bold">
                    {project.npv !== null && project.npv !== undefined
                      ? `$${project.npv.toFixed(0)}M`
                      : 'N/A'}
                  </div>
                  {index === 1 && projects.length === 2 && project.npv !== null && projects[0].npv !== null && (
                    <div className={cn(
                      "text-xs flex items-center gap-1",
                      project.npv > projects[0].npv
                        ? "text-green-600"
                        : "text-red-600"
                    )}>
                      {project.npv > projects[0].npv ? (
                        <TrendingUp className="h-3 w-3" />
                      ) : (
                        <TrendingDown className="h-3 w-3" />
                      )}
                      {Math.abs(getValueDiff(project.npv, projects[0].npv)).toFixed(1)}%
                    </div>
                  )}
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">IRR</div>
                  <div className="text-xl font-bold">
                    {project.irr !== null && project.irr !== undefined
                      ? `${project.irr.toFixed(1)}%`
                      : 'N/A'}
                  </div>
                  {index === 1 && projects.length === 2 && project.irr !== null && projects[0].irr !== null && (
                    <div className={cn(
                      "text-xs flex items-center gap-1",
                      project.irr > projects[0].irr
                        ? "text-green-600"
                        : "text-red-600"
                    )}>
                      {project.irr > projects[0].irr ? (
                        <TrendingUp className="h-3 w-3" />
                      ) : (
                        <TrendingDown className="h-3 w-3" />
                      )}
                      {Math.abs(getValueDiff(project.irr, projects[0].irr)).toFixed(1)}%
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">CAPEX</div>
                  <div className="font-semibold">
                    {project.capex !== null && project.capex !== undefined
                      ? `$${project.capex.toFixed(0)}M`
                      : 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">STAGE</div>
                  <div className="font-semibold">{project.stage || 'Unknown'}</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-sm text-muted-foreground">COMMODITIES</div>
                  <div className="font-semibold">
                    {project.commodities && project.commodities.length > 0
                      ? project.commodities[0]
                      : 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-muted-foreground">STATUS</div>
                  <div className="font-semibold">{project.status || 'Unknown'}</div>
                </div>
              </div>

              <Badge 
                variant={project.stage === "Production" ? "default" : "secondary"}
                className="w-full justify-center"
              >
                {project.stage}
              </Badge>

              <div className="text-xs text-muted-foreground">
                Last filing: 3 months ago
              </div>

              {/* Recent Activity */}
              <div>
                <h4 className="font-medium mb-2">Recent Activity</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span>Production Update - Q1 2024</span>
                    <ExternalLink className="h-3 w-3" />
                  </div>
                  <div className="text-xs text-muted-foreground">Jul 22, 2025</div>
                  
                  <div className="flex items-center justify-between">
                    <span>Resource Estimation - Q2 2024</span>
                    <ExternalLink className="h-3 w-3" />
                  </div>
                  <div className="text-xs text-muted-foreground">Jun 22, 2025</div>
                  
                  <div className="flex items-center justify-between">
                    <span>ESG Report - Q3 2024</span>
                    <ExternalLink className="h-3 w-3" />
                  </div>
                  <div className="text-xs text-muted-foreground">May 23, 2025</div>
                </div>
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
                {projects.map((project) => (
                  <TableHead key={project.id} className="text-center">
                    {project.project}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {metrics
                .filter((metric) => {
                  if (metric.key === "esgScore" && !showESGScore) return false
                  if (metric.key === "riskLevel" && !showJurisdictionRisk) return false
                  return true
                })
                .map((metric) => (
                  <TableRow key={metric.key}>
                    <TableCell className="font-medium">{metric.label}</TableCell>
                    {projects.map((project, index) => {
                      const value = project[metric.key as keyof MiningProject]
                      const formattedValue = metric.format(value, project)
                      
                      let diffElement = null
                      if (metric.showDiff && index > 0 && typeof value === "number") {
                        const baseValue = projects[0][metric.key as keyof MiningProject] as number
                        const diff = getValueDiff(value, baseValue)
                        diffElement = (
                          <div className={cn(
                            "text-xs",
                            diff > 0 ? "text-green-600" : "text-red-600"
                          )}>
                            {diff > 0 ? "+" : ""}{diff.toFixed(1)}%
                          </div>
                        )
                      }

                      return (
                        <TableCell key={project.id} className="text-center">
                          <div>
                            {metric.key === "riskLevel" ? (
                              <Badge className={cn(getRiskColor(value as string))}>
                                {formattedValue}
                              </Badge>
                            ) : metric.key === "esgScore" ? (
                              <Badge variant="outline">{formattedValue}</Badge>
                            ) : (
                              formattedValue
                            )}
                            {diffElement}
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

function ExternalLink({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </svg>
  )
} 