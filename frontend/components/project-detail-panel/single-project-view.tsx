"use client"

import * as React from "react"
import { MapPin, AlertTriangle, TrendingUp, DollarSign, Calendar, Users, FileText, ExternalLink } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { MiningProject } from "@/lib/types/mining-project"
import { cn } from "@/lib/utils"
import { SensitivityAnalysis } from "./sensitivity-analysis"
import { dummyProjects } from "@/lib/data/dummy-projects"

interface SingleProjectViewProps {
  project: MiningProject
  onProjectSelect?: (projectId: string) => void
}

export function SingleProjectView({ project, onProjectSelect }: SingleProjectViewProps) {
  // Get similar projects (for demo, just get 4 random projects)
  const similarProjects = dummyProjects
    .filter(p => p.id !== project.id && p.primaryCommodity === project.primaryCommodity)
    .slice(0, 4)

  const getRiskBadgeColor = (risk: string) => {
    switch (risk) {
      case "Low": return "bg-green-100 text-green-800"
      case "Medium": return "bg-yellow-100 text-yellow-800"
      case "High": return "bg-orange-100 text-orange-800"
      case "Very High": return "bg-red-100 text-red-800"
      default: return ""
    }
  }

  return (
    <div className="p-6 space-y-6">
      {/* Company Info Section */}
      <div className="space-y-4">
        <div className="flex items-start justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-muted-foreground">
              <MapPin className="h-4 w-4" />
              <span className="text-sm">
                Company: {project.investorsOwnership.split("(")[0].trim()}
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <MapPin className="h-4 w-4" />
              <span className="text-sm">Location: {project.jurisdiction}</span>
            </div>
          </div>
          <Badge className={cn(getRiskBadgeColor(project.riskLevel))}>
            {project.riskLevel} Risk
          </Badge>
        </div>

        <div className="space-y-2">
          <div className="text-sm text-muted-foreground">Primary Commodity:</div>
          <div className="flex items-center gap-2">
            <Badge variant="outline">{project.primaryCommodity}</Badge>
            {project.resourceGrade && (
              <span className="text-sm">
                ({project.resourceGrade} {project.gradeUnit})
              </span>
            )}
          </div>
          {project.secondaryCommodities && project.secondaryCommodities.length > 0 && (
            <>
              <div className="text-sm text-muted-foreground">Secondary Commodities:</div>
              <div className="flex gap-2">
                {project.secondaryCommodities.map((commodity) => (
                  <Badge key={commodity} variant="secondary">
                    {commodity}
                  </Badge>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="text-sm text-muted-foreground">
          Stage: <Badge variant="outline">{project.stage}</Badge>
        </div>

        <div className="text-sm text-muted-foreground">
          Major Investors: {project.investorsOwnership}
        </div>
      </div>

      {/* Key Metrics Cards */}
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-green-600" />
              <div>
                <div className="text-2xl font-semibold">
                  ${(project.postTaxNPV / 1000).toFixed(1)}B
                </div>
                <div className="text-sm text-muted-foreground">NPV (8%)</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-blue-600" />
              <div>
                <div className="text-2xl font-semibold">{project.irr.toFixed(1)}%</div>
                <div className="text-sm text-muted-foreground">IRR</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Calendar className="h-5 w-5 text-orange-600" />
              <div>
                <div className="text-2xl font-semibold">{project.paybackYears} years</div>
                <div className="text-sm text-muted-foreground">Payback</div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <Users className="h-5 w-5 text-purple-600" />
              <div>
                <div className="text-2xl font-semibold">
                  {project.containedMetal ? 
                    `${(project.containedMetal / 1000).toFixed(0)}k t` : 
                    `${project.mineLife} yrs`
                  }
                </div>
                <div className="text-sm text-muted-foreground">
                  {project.containedMetal ? "Annual Production" : "Mine Life"}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Risk Alerts */}
      {project.redFlags && project.redFlags.length > 0 && (
        <Card className="border-orange-200 bg-orange-50/50">
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-orange-600" />
              Risk Events
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {project.redFlags.map((flag, index) => (
              <div key={index} className="flex items-start gap-2 text-sm">
                <span className="text-orange-600">•</span>
                <span>{flag}</span>
              </div>
            ))}
            <div className="text-xs text-muted-foreground mt-2">
              Last scan: {new Date().toLocaleDateString()} • 14 sources
            </div>
          </CardContent>
        </Card>
      )}

      {/* Open AI Analysis Button */}
      <Button className="w-full" size="lg">
        Open full AI analysis
        <ExternalLink className="ml-2 h-4 w-4" />
      </Button>

      {/* Tabs Section */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="technical">Technical Report</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="experts">Experts</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-6 mt-6">
          {/* Detailed Metrics */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Detailed Metrics</h3>
            <div className="grid grid-cols-3 gap-6">
              <div>
                <h4 className="font-medium mb-3">Economics</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Initial Capex:</span>
                    <span className="font-medium">${project.capex}M</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Mine Life:</span>
                    <span className="font-medium">{project.mineLife} years</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">AISC:</span>
                    <span className="font-medium">${project.aisc}/t</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-3">Resources & Reserves</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Total Resources:</span>
                    <span className="font-medium">
                      {project.containedMetal ? 
                        `${(project.containedMetal / 1000000).toFixed(1)} Mt @ ${project.resourceGrade}${project.gradeUnit}` :
                        "N/A"
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Proven Reserves:</span>
                    <span className="font-medium">
                      {project.containedMetal ? 
                        `${(project.containedMetal * 0.7 / 1000000).toFixed(1)} Mt @ ${project.resourceGrade}${project.gradeUnit}` :
                        "N/A"
                      }
                    </span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-3">Production</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Annual Output:</span>
                    <span className="font-medium">
                      {project.containedMetal ? 
                        `${(project.containedMetal / project.mineLife / 1000).toFixed(0)} kt/y ${project.primaryCommodity}` :
                        "N/A"
                      }
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Strip Ratio:</span>
                    <span className="font-medium">2.1:1</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <Separator />

          {/* Sensitivity Analysis */}
          <SensitivityAnalysis project={project} />

          <Separator />

          {/* Related Documents */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Related Documents</h3>
            <div className="space-y-2">
              <Card className="p-4 hover:bg-muted/50 cursor-pointer">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <FileText className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium">Feasibility Study 2023</div>
                      <div className="text-sm text-muted-foreground">
                        12.4 MB • 2023-09-15
                      </div>
                    </div>
                  </div>
                  <Button variant="ghost" size="icon">
                    <ExternalLink className="h-4 w-4" />
                  </Button>
                </div>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="technical" className="mt-6">
          <div className="text-center py-12 text-muted-foreground">
            Technical report content would be displayed here
          </div>
        </TabsContent>

        <TabsContent value="insights" className="mt-6">
          <div className="text-center py-12 text-muted-foreground">
            AI-generated insights would be displayed here
          </div>
        </TabsContent>

        <TabsContent value="experts" className="mt-6">
          <div className="text-center py-12 text-muted-foreground">
            Expert analysis and opinions would be displayed here
          </div>
        </TabsContent>
      </Tabs>

      {/* Similar Projects */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Similar Projects</h3>
        <div className="space-y-3">
          {similarProjects.map((similarProject) => (
            <Card 
              key={similarProject.id}
              className="p-4 hover:bg-muted/50 cursor-pointer transition-colors"
              onClick={() => onProjectSelect?.(similarProject.id)}
            >
              <div className="space-y-2">
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="font-medium">{similarProject.project}</h4>
                    <div className="text-sm text-muted-foreground">
                      Grade: {similarProject.resourceGrade} {similarProject.gradeUnit}
                    </div>
                  </div>
                  <Badge variant="outline">{similarProject.stage}</Badge>
                </div>
                <div className="text-sm">
                  NPV: ${(similarProject.postTaxNPV / 1000).toFixed(1)}B
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="sticky bottom-0 bg-background border-t p-4 -mx-6 -mb-6">
        <h3 className="text-lg font-semibold mb-3">Quick Actions</h3>
        <div className="space-y-2">
          <Button className="w-full" variant="default">
            Compare Projects
          </Button>
          <Button className="w-full" variant="outline">
            Export Analysis
          </Button>
        </div>
      </div>
    </div>
  )
} 