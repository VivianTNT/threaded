'use client'

import React, { useState } from "react"
import { MiningProject } from "@/lib/types/mining-project"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Separator } from "@/components/ui/separator"
import { ExternalLink, AlertTriangle, FileText, Users, Bookmark, BookmarkCheck, ImageIcon, MessageSquare, Loader2 } from "lucide-react"
import { SensitivityAnalysis } from "./sensitivity-analysis"
import { cn } from "@/lib/utils"
import { ExportDropdown, ExportFormat } from "@/components/ui/export-dropdown"
import { exportProjects } from "@/lib/export-utils"
import { toast } from "sonner"
import { Toaster } from "@/components/ui/toaster"
import { supabase } from "@/lib/supabase/client"
import { useGlobalChat } from "@/lib/global-chat-context"
import { useChat } from "@/lib/chat-context"
import { formatNumber, formatCurrency, formatPercent, formatTonnes } from "@/lib/format-utils"

interface SingleProjectViewProps {
  project: MiningProject
  onProjectSelect?: (projectId: string) => void
  onClose?: () => void
}

const getRiskBadgeColor = (risk: string) => {
  switch (risk.toLowerCase()) {
    case "low":
      return "bg-green-100 text-green-800"
    case "medium":
      return "bg-yellow-100 text-yellow-800"
    case "high":
      return "bg-red-100 text-red-800"
    default:
      return "bg-gray-100 text-gray-800"
  }
}

export function SingleProjectView({ project, onProjectSelect, onClose }: SingleProjectViewProps) {
  const [updatingWatchlist, setUpdatingWatchlist] = useState(false)
  const [generatingImage, setGeneratingImage] = useState(false)
  const [isWatchlisted, setIsWatchlisted] = useState(project.watchlist || false)
  const [generatedImageUrl, setGeneratedImageUrl] = useState<string | null>(project.generated_image_url || null)
  const { setInput } = useGlobalChat()
  const { toggleChat } = useChat()

  // Update local state when project prop changes
  React.useEffect(() => {
    setIsWatchlisted(project.watchlist || false)
    setGeneratedImageUrl(project.generated_image_url || null)
  }, [project.watchlist, project.generated_image_url])

  const similarProjects = [
    { id: '1', name: 'Similar Project 1', npv: 3200, irr: 28 },
    { id: '2', name: 'Similar Project 2', npv: 2800, irr: 24 },
    { id: '3', name: 'Similar Project 3', npv: 4100, irr: 31 },
    { id: '4', name: 'Similar Project 4', npv: 2500, irr: 22 }
  ]

  const handleToggleWatchlist = async () => {
    try {
      setUpdatingWatchlist(true)
      const newWatchlistStatus = !isWatchlisted

      console.log('Updating watchlist for project:', project.id, 'to:', newWatchlistStatus)

      const { data, error } = await supabase
        .from('projects')
        .update({
          watchlist: newWatchlistStatus,
          watchlisted_at: newWatchlistStatus ? new Date().toISOString() : null
        })
        .eq('id', project.id)
        .select()

      if (error) {
        console.error('Supabase error:', error)
        throw error
      }

      console.log('Update successful:', data)

      // Update local state immediately for visual feedback
      setIsWatchlisted(newWatchlistStatus)

      toast.success(newWatchlistStatus ? 'Added to watchlist' : 'Removed from watchlist')

      // Trigger refresh
      window.dispatchEvent(new CustomEvent('refreshProjects'))
    } catch (error: any) {
      console.error('Error updating watchlist:', error)
      toast.error(`Failed to update watchlist: ${error.message || 'Unknown error'}`)
    } finally {
      setUpdatingWatchlist(false)
    }
  }

  const handleExport = (format: ExportFormat) => {
    exportProjects([project], format, `${project.project}-details`)
    toast.success(`Exported project as ${format.toUpperCase()}`)
  }

  const handleGenerateImage = async () => {
    try {
      setGeneratingImage(true)

      const response = await fetch('/api/projects/generate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ projectId: project.id })
      })

      const data = await response.json()

      if (!response.ok) {
        console.error('API Error:', data)
        throw new Error(data.error || data.details || 'Failed to generate image')
      }

      const { imageUrl } = data

      // Update local state with the generated image
      setGeneratedImageUrl(imageUrl)

      toast.success('Image generated successfully')

      // Trigger refresh after a small delay to ensure database is updated
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('refreshProjects'))
      }, 500)
    } catch (error: any) {
      console.error('Error generating image:', error)
      toast.error(error.message || 'Failed to generate image')
    } finally {
      setGeneratingImage(false)
    }
  }

  const handleAddToChat = () => {
    const projectContext = `Analyze this mining project:

Project: ${project.name}
Company: ${project.company || 'N/A'}
Stage: ${project.stage || 'Unknown'}
Location: ${project.location || 'N/A'}
Commodities: ${(project.commodities || []).join(', ')}
Status: ${project.status || 'Unknown'}
Ownership: ${project.ownership_percentage !== null ? `${project.ownership_percentage}%` : 'N/A'}
Resource Estimate: ${project.resource_estimate || 'N/A'}
Reserve Estimate: ${project.reserve_estimate || 'N/A'}

What is your assessment of this project?`

    // Set the input in the chat
    setInput(projectContext)

    // Close the detail panel if there's an onClose handler
    if (onClose) {
      onClose()
    }

    // Open the chat panel
    toggleChat()

    toast.success('Project added to chat')
  }

  return (
    <div className="space-y-4 p-6">
      {/* Company Info */}
      <div className="space-y-3">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-medium">{project.name}</h2>
            <p className="text-sm text-muted-foreground">
              {project.company || 'Unknown'} • {project.location || 'N/A'}
            </p>
          </div>
          <Badge className={cn("text-xs", getRiskBadgeColor(project.riskLevel || 'Medium'))}>
            {project.riskLevel || 'Medium'} Risk
          </Badge>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button
            variant={isWatchlisted ? "default" : "outline"}
            size="sm"
            onClick={handleToggleWatchlist}
            disabled={updatingWatchlist}
          >
            {updatingWatchlist ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : isWatchlisted ? (
              <BookmarkCheck className="h-4 w-4 mr-2 fill-current" />
            ) : (
              <Bookmark className="h-4 w-4 mr-2" />
            )}
            {isWatchlisted ? 'Remove from Watchlist' : 'Add to Watchlist'}
          </Button>

          <ExportDropdown onExport={handleExport} size="sm" />

          <Button
            variant="outline"
            size="sm"
            onClick={handleGenerateImage}
            disabled={generatingImage}
          >
            {generatingImage ? (
              <Loader2 className="h-4 w-4 animate-spin mr-2" />
            ) : (
              <ImageIcon className="h-4 w-4 mr-2" />
            )}
            Generate Image
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={handleAddToChat}
          >
            <MessageSquare className="h-4 w-4 mr-2" />
            Add to Chat
          </Button>

          {project.technicalReportUrl && (
            <Button
              variant="outline"
              size="sm"
              asChild
            >
              <a
                href={project.technicalReportUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                <FileText className="h-4 w-4 mr-2" />
                Technical Report
              </a>
            </Button>
          )}
        </div>

        {/* Generated Image Display */}
        {generatedImageUrl && (
          <div className="mt-4">
            <div className="relative rounded-lg overflow-hidden border">
              <img
                src={generatedImageUrl}
                alt={`AI visualization for ${project.project}`}
                className="w-full h-auto"
              />
              <div className="absolute top-2 right-2 bg-black/50 text-white px-2 py-1 rounded text-xs">
                AI Generated
              </div>
            </div>
          </div>
        )}
      </div>

      <Separator />

      {/* Key Metrics Table */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Financial Metrics</h3>
        <div className="border rounded-lg divide-y">
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">NPV (Net Present Value)</span>
              <span className="text-sm font-medium">
                {project.npv !== null && project.npv !== undefined
                  ? `$${project.npv.toFixed(0)}M`
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">IRR (Internal Rate of Return)</span>
              <span className="text-sm font-medium">
                {project.irr !== null && project.irr !== undefined
                  ? `${project.irr.toFixed(1)}%`
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">CAPEX (Capital Expenditure)</span>
              <span className="text-sm font-medium">
                {project.capex !== null && project.capex !== undefined
                  ? `$${project.capex.toFixed(0)}M`
                  : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <Separator />

      {/* Project Information */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Project Information</h3>
        <div className="border rounded-lg divide-y">
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Location</span>
              <span className="text-sm font-medium">{project.location || 'N/A'}</span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Stage</span>
              <Badge variant="outline" className="text-xs">{project.stage || 'Unknown'}</Badge>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Status</span>
              <Badge variant="outline" className="text-xs">{project.status || 'Unknown'}</Badge>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Commodities</span>
              <div className="flex flex-wrap gap-1 justify-end">
                {(project.commodities || []).map((commodity, i) => (
                  <Badge key={i} variant="outline" className="text-xs">{commodity}</Badge>
                ))}
              </div>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Ownership</span>
              <span className="text-sm font-medium">
                {project.ownership_percentage !== null && project.ownership_percentage !== undefined
                  ? `${project.ownership_percentage}%`
                  : 'N/A'}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Resource Estimate</span>
              <span className="text-sm font-medium">{project.resource_estimate || 'N/A'}</span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Reserve Estimate</span>
              <span className="text-sm font-medium">{project.reserve_estimate || 'N/A'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Description */}
      {project.description && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Description</h3>
          <p className="text-sm text-muted-foreground">{project.description}</p>
        </div>
      )}

      {/* Project URLs */}
      {project.urls && project.urls.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Resources</h3>
          <div className="flex flex-wrap gap-2">
            {project.urls.map((url, i) => (
              <Button
                key={i}
                variant="outline"
                size="sm"
                asChild
              >
                <a
                  href={url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs"
                >
                  <ExternalLink className="h-3 w-3 mr-1" />
                  Source {i + 1}
                </a>
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Project Details Table */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Additional Details</h3>
        <div className="border rounded-lg divide-y">
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Last Updated</span>
              <span className="text-xs">{new Date(project.updated_at).toLocaleDateString()}</span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Primary Commodity</span>
              <span className="text-sm">{project.primaryCommodity || 'N/A'}</span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Resource Grade</span>
              <span className="text-sm">
                {formatNumber(project.resourceGrade, { decimals: 2, suffix: ` ${project.gradeUnit || '%'}` })}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Contained Metal</span>
              <span className="text-sm">
                {formatTonnes(project.containedMetal, { decimals: 0, unit: 't' })}
              </span>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">ESG Score</span>
              <Badge variant="outline" className="text-xs">{project.esgScore}</Badge>
            </div>
          </div>
          <div className="p-2.5">
            <div className="flex justify-between items-center">
              <span className="text-sm text-muted-foreground">Technical Report</span>
              {project.technicalReportUrl ? (
                <a
                  href={project.technicalReportUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-blue-600 hover:underline flex items-center gap-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  View Report
                  <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                  </svg>
                </a>
              ) : (
                <span className="text-sm text-muted-foreground">N/A</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Risk Alerts */}
      {typeof project.redFlags === 'number' && project.redFlags > 0 ? (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-600" />
            <span className="text-sm font-medium text-red-900">
              {project.redFlags} Risk Flags Identified
            </span>
          </div>
        </div>
      ) : null}

      {/* Tabs */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="technical">Technical Report</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="experts">Experts</TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-4 mt-4">
          <Card className="p-4">
            <h4 className="text-sm font-medium mb-3">Production Profile</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Annual Production</span>
                <span>{project.annualProduction || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Resource</span>
                <span>{project.totalResource || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Strip Ratio</span>
                <span>{project.stripRatio || 'N/A'}</span>
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <h4 className="text-sm font-medium mb-3">Infrastructure</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Power Supply</span>
                <span>{project.powerSupply || 'Grid connected'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Water Source</span>
                <span>{project.waterSource || 'Local aquifer'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Transport</span>
                <span>{project.transport || 'Road & rail access'}</span>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="technical" className="mt-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Technical Documentation</h4>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="space-y-2">
              <Button variant="outline" className="w-full justify-start text-sm">
                <FileText className="mr-2 h-4 w-4" />
                NI 43-101 Technical Report (2024)
              </Button>
              <Button variant="outline" className="w-full justify-start text-sm">
                <FileText className="mr-2 h-4 w-4" />
                Feasibility Study Update (2023)
              </Button>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="mt-4">
          <Card className="p-4">
            <h4 className="text-sm font-medium mb-3">AI Analysis Summary</h4>
            <p className="text-sm text-muted-foreground">
              Advanced AI analysis of technical reports, market conditions, and risk factors.
            </p>
            <Button className="w-full mt-4" variant="outline">
              Open Full AI Analysis
            </Button>
          </Card>
        </TabsContent>

        <TabsContent value="experts" className="mt-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium">Industry Experts</h4>
              <Users className="h-4 w-4 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">
              Connect with verified mining industry experts for deeper insights.
            </p>
          </Card>
        </TabsContent>
      </Tabs>

      <Separator />

      {/* Sensitivity Analysis */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Sensitivity Analysis</h3>
        <SensitivityAnalysis project={project} />
      </div>

      <Separator />

      {/* Similar Projects */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium">Similar Projects</h3>
        <div className="grid grid-cols-2 gap-3">
          {similarProjects.map((similar) => (
            <Card 
              key={similar.id} 
              className="p-3 cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => onProjectSelect?.(similar.id)}
            >
              <div className="space-y-1">
                <h4 className="text-sm font-medium">{similar.name}</h4>
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>NPV: ${similar.npv}M</span>
                  <span>IRR: {similar.irr}%</span>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="sticky bottom-0 bg-background border-t pt-4 -mx-6 px-6 -mb-6">
        <div className="flex gap-2">
          <Button className="flex-1" size="sm">Compare Projects</Button>
          <Button className="flex-1" variant="outline" size="sm">Export Analysis</Button>
        </div>
      </div>
      <Toaster />
    </div>
  )
} 