"use client"

import * as React from "react"
import { Filter, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Checkbox } from "@/components/ui/checkbox"

export type NewsSentiment = 'positive' | 'neutral' | 'negative'
export type NewsStage = 'Exploration' | 'Development' | 'Production' | 'All Stages'

interface NewsFiltersProps {
  onFiltersChange: (filters: NewsFilterState) => void
  availableCommodities?: string[]
}

export interface NewsFilterState {
  commodities: string[]
  sentiment: NewsSentiment[]
  stages: NewsStage[]
  dateRange: { start: string; end: string }
  sources: string[]
}

const defaultCommodities = [
  "Lithium",
  "Copper",
  "Nickel",
  "Cobalt",
  "Gold",
  "Silver",
  "Uranium",
  "Graphite",
  "Rare Earths",
  "Platinum",
  "Palladium",
  "Zinc",
  "Lead",
  "Tin",
  "Iron Ore",
  "Aluminum",
  "Steel"
]

const sentiments: NewsSentiment[] = ['positive', 'neutral', 'negative']
const stages: NewsStage[] = ['Exploration', 'Development', 'Production', 'All Stages']

const savedViews = [
  { id: "1", name: "All News" },
  { id: "2", name: "Positive Sentiment" },
  { id: "3", name: "Battery Metals" },
  { id: "4", name: "Production Updates" },
  { id: "5", name: "Today's News" },
]

const newsSources = [
  "Reuters",
  "Mining.com",
  "Bloomberg",
  "Mining Weekly",
  "Northern Miner",
  "Company Release",
  "Financial Times",
  "S&P Global",
  "Kitco",
  "Fastmarkets"
]

export function NewsFilters({ onFiltersChange, availableCommodities = defaultCommodities }: NewsFiltersProps) {
  const [selectedCommodities, setSelectedCommodities] = React.useState<string[]>([])
  const [selectedSentiment, setSelectedSentiment] = React.useState<NewsSentiment[]>([])
  const [selectedStages, setSelectedStages] = React.useState<NewsStage[]>([])
  const [selectedSources, setSelectedSources] = React.useState<string[]>([])
  const [dateRange, setDateRange] = React.useState({ start: "", end: "" })

  const applyFilters = () => {
    onFiltersChange({
      commodities: selectedCommodities,
      sentiment: selectedSentiment,
      stages: selectedStages,
      dateRange,
      sources: selectedSources
    })
  }

  const clearAllFilters = () => {
    setSelectedCommodities([])
    setSelectedSentiment([])
    setSelectedStages([])
    setSelectedSources([])
    setDateRange({ start: "", end: "" })
    onFiltersChange({
      commodities: [],
      sentiment: [],
      stages: [],
      dateRange: { start: "", end: "" },
      sources: []
    })
  }

  const activeFiltersCount =
    selectedCommodities.length +
    selectedSentiment.length +
    selectedStages.length +
    selectedSources.length +
    (dateRange.start || dateRange.end ? 1 : 0)

  return (
    <div className="space-y-4 border rounded-lg p-4 bg-muted/50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4" />
          <span className="font-medium">Filters</span>
          {activeFiltersCount > 0 && (
            <Badge variant="secondary">{activeFiltersCount} active</Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Select defaultValue="1">
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Select saved view" />
            </SelectTrigger>
            <SelectContent>
              {savedViews.map((view) => (
                <SelectItem key={view.id} value={view.id}>
                  {view.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {activeFiltersCount > 0 && (
            <Button variant="ghost" size="sm" onClick={clearAllFilters}>
              Clear all
              <X className="ml-2 h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {/* Commodity Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              Commodity
              {selectedCommodities.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedCommodities.length}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-3">
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {availableCommodities.map((commodity) => (
                <div key={commodity} className="flex items-center space-x-2">
                  <Checkbox
                    id={commodity}
                    checked={selectedCommodities.includes(commodity)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedCommodities([...selectedCommodities, commodity])
                      } else {
                        setSelectedCommodities(
                          selectedCommodities.filter((c) => c !== commodity)
                        )
                      }
                    }}
                  />
                  <label
                    htmlFor={commodity}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {commodity}
                  </label>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>

        {/* Sentiment Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              Sentiment
              {selectedSentiment.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedSentiment.length}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-3">
            <div className="space-y-2">
              {sentiments.map((sentiment) => (
                <div key={sentiment} className="flex items-center space-x-2">
                  <Checkbox
                    id={sentiment}
                    checked={selectedSentiment.includes(sentiment)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedSentiment([...selectedSentiment, sentiment])
                      } else {
                        setSelectedSentiment(selectedSentiment.filter((s) => s !== sentiment))
                      }
                    }}
                  />
                  <label
                    htmlFor={sentiment}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 capitalize"
                  >
                    {sentiment}
                  </label>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>

        {/* Stage Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              Stage
              {selectedStages.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedStages.length}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-3">
            <div className="space-y-2">
              {stages.map((stage) => (
                <div key={stage} className="flex items-center space-x-2">
                  <Checkbox
                    id={stage}
                    checked={selectedStages.includes(stage)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedStages([...selectedStages, stage])
                      } else {
                        setSelectedStages(selectedStages.filter((s) => s !== stage))
                      }
                    }}
                  />
                  <label
                    htmlFor={stage}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {stage}
                  </label>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>

        {/* Source Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              Source
              {selectedSources.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedSources.length}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-3">
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {newsSources.map((source) => (
                <div key={source} className="flex items-center space-x-2">
                  <Checkbox
                    id={source}
                    checked={selectedSources.includes(source)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedSources([...selectedSources, source])
                      } else {
                        setSelectedSources(selectedSources.filter((s) => s !== source))
                      }
                    }}
                  />
                  <label
                    htmlFor={source}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {source}
                  </label>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>

        {/* Date Range Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm">
              Date Range
              {(dateRange.start || dateRange.end) && (
                <Badge variant="secondary" className="ml-2">
                  1
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-3">
            <div className="space-y-2">
              <Label>Date Range</Label>
              <div className="flex flex-col gap-2">
                <Input
                  type="date"
                  placeholder="Start date"
                  value={dateRange.start}
                  onChange={(e) => setDateRange({ ...dateRange, start: e.target.value })}
                />
                <Input
                  type="date"
                  placeholder="End date"
                  value={dateRange.end}
                  onChange={(e) => setDateRange({ ...dateRange, end: e.target.value })}
                />
              </div>
            </div>
          </PopoverContent>
        </Popover>

        <Button size="sm" onClick={applyFilters}>
          Apply Filters
        </Button>
      </div>

      {/* Active filter badges */}
      {activeFiltersCount > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedCommodities.map((commodity) => (
            <Badge key={commodity} variant="secondary">
              {commodity}
              <button
                className="ml-1"
                onClick={() =>
                  setSelectedCommodities(selectedCommodities.filter((c) => c !== commodity))
                }
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {selectedSentiment.map((sentiment) => (
            <Badge key={sentiment} variant="secondary">
              {sentiment}
              <button
                className="ml-1"
                onClick={() => setSelectedSentiment(selectedSentiment.filter((s) => s !== sentiment))}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {selectedStages.map((stage) => (
            <Badge key={stage} variant="secondary">
              {stage}
              <button
                className="ml-1"
                onClick={() => setSelectedStages(selectedStages.filter((s) => s !== stage))}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {selectedSources.map((source) => (
            <Badge key={source} variant="secondary">
              {source}
              <button
                className="ml-1"
                onClick={() => setSelectedSources(selectedSources.filter((s) => s !== source))}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {(dateRange.start || dateRange.end) && (
            <Badge variant="secondary">
              {dateRange.start || "Any"} - {dateRange.end || "Any"}
              <button className="ml-1" onClick={() => setDateRange({ start: "", end: "" })}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
        </div>
      )}
    </div>
  )
}
