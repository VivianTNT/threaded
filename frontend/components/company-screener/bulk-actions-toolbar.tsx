"use client"

import * as React from "react"
import { Download, GitCompare, ListPlus, X, FileSearch } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Company } from "@/lib/hooks/use-companies"

interface BulkActionsToolbarProps {
  selectedCount: number
  selectedCompanies: Company[]
  onClearSelection: () => void
  onCompanyAnalysis?: () => void
  onCompare?: () => void
}

export function BulkActionsToolbar({
  selectedCount,
  selectedCompanies,
  onClearSelection,
  onCompanyAnalysis,
  onCompare,
}: BulkActionsToolbarProps) {
  const handleExportCSV = () => {
    // Create CSV content
    const headers = [
      "Name",
      "Ticker",
      "Exchange",
      "Country",
      "Market Cap",
      "Website",
    ]

    const rows = selectedCompanies.map((company) => [
      company.name,
      company.ticker || "",
      company.exchange || "",
      company.country || "",
      company.market_cap || "",
      company.website || "",
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
    link.setAttribute("download", `lithos-companies-export-${Date.now()}.csv`)
    link.style.visibility = "hidden"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const handleAddToWatchlist = () => {
    // In a real app, this would call an API to add companies to watchlist
    console.log("Adding to watchlist:", selectedCompanies.map((c) => c.id))
    alert(`Added ${selectedCount} companies to watchlist`)
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
        {selectedCount === 1 && onCompanyAnalysis && (
          <Button
            variant="outline"
            size="sm"
            onClick={onCompanyAnalysis}
            className="h-8"
          >
            <FileSearch className="mr-2 h-4 w-4" />
            Company Analysis
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
