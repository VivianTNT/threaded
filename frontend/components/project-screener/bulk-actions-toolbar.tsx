"use client"

import * as React from "react"
import { Download, Eye, GitCompare, ListPlus, X, FileSearch } from "lucide-react"

import { Button } from "@/components/ui/button"
import { MiningProject } from "@/lib/types/mining-project"

interface BulkActionsToolbarProps {
  selectedCount: number
  selectedProjects: MiningProject[]
  onClearSelection: () => void
  onProjectAnalysis?: () => void
  onCompare?: () => void
}

export function BulkActionsToolbar({
  selectedCount,
  selectedProjects,
  onClearSelection,
  onProjectAnalysis,
  onCompare,
}: BulkActionsToolbarProps) {
  const handleExportCSV = () => {
    // Create CSV content
    const headers = [
      "Project",
      "Stage",
      "Mine Life (yrs)",
      "Post-tax NPV (USD M)",
      "IRR %",
      "Payback (yrs)",
      "CAPEX (USD M)",
      "AISC (USD/t)",
      "Primary Commodity",
      "Jurisdiction",
      "Risk Level",
      "Investors/Ownership",
    ]

    const rows = selectedProjects.map((project) => [
      project.project,
      project.stage,
      project.mineLife,
      project.postTaxNPV,
      project.irr,
      project.paybackYears,
      project.capex,
      project.aisc,
      project.primaryCommodity,
      project.jurisdiction,
      project.riskLevel,
      project.investorsOwnership,
    ])

    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.map((cell) => `"${cell}"`).join(",")),
    ].join("\n")

    // Download CSV
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
    const link = document.createElement("a")
    const url = URL.createObjectURL(blob)
    link.setAttribute("href", url)
    link.setAttribute("download", `lithos-projects-export-${Date.now()}.csv`)
    link.style.visibility = "hidden"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleAddToWatchlist = () => {
    // In a real app, this would call an API to add projects to watchlist
    console.log("Adding to watchlist:", selectedProjects.map((p) => p.id))
    alert(`Added ${selectedCount} projects to watchlist`)
  }

  return (
    <div className="flex items-center justify-between rounded-lg border bg-muted/50 px-4 py-2">
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">{selectedCount} selected</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={onClearSelection}
          className="h-8 px-2"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex items-center gap-2">
        {selectedCount === 1 && onProjectAnalysis && (
          <Button
            variant="outline"
            size="sm"
            onClick={onProjectAnalysis}
            className="h-8"
          >
            <FileSearch className="mr-2 h-4 w-4" />
            Project Analysis
          </Button>
        )}
        {selectedCount >= 2 && onCompare && (
          <Button
            variant="outline"
            size="sm"
            onClick={onCompare}
            className="h-8"
          >
            <GitCompare className="mr-2 h-4 w-4" />
            Compare
          </Button>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={handleAddToWatchlist}
          className="h-8"
        >
          <ListPlus className="mr-2 h-4 w-4" />
          Add to Watch-list
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleExportCSV}
          className="h-8"
        >
          <Download className="mr-2 h-4 w-4" />
          Export Selected
        </Button>
      </div>
    </div>
  )
} 