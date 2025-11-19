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

export interface CompaniesFilterState {
  commodities: string[]
  countries: string[]
  projectStages: string[]
  marketCapRange: { min: string; max: string }
  npvRange: { min: string; max: string }
  projectCountRange: { min: string; max: string }
}

interface CompaniesFiltersProps {
  onFiltersChange: (filters: CompaniesFilterState) => void
}

const commodities = [
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
  "Iron Ore",
]

const projectStages = [
  "Production Focus",
  "Development Focus",
  "Exploration Focus",
  "Diversified"
]

const savedViews = [
  { id: "1", name: "All Companies" },
  { id: "2", name: "Major Producers" },
  { id: "3", name: "Battery Metals Focus" },
  { id: "4", name: "Gold & Silver" },
  { id: "5", name: "High NPV Companies" },
]

export function CompaniesFilters({ onFiltersChange }: CompaniesFiltersProps) {
  const [selectedCommodities, setSelectedCommodities] = React.useState<string[]>([])
  const [selectedCountries, setSelectedCountries] = React.useState<string[]>([])
  const [selectedStages, setSelectedStages] = React.useState<string[]>([])
  const [marketCapRange, setMarketCapRange] = React.useState({ min: "", max: "" })
  const [npvRange, setNpvRange] = React.useState({ min: "", max: "" })
  const [projectCountRange, setProjectCountRange] = React.useState({ min: "", max: "" })

  const applyFilters = () => {
    onFiltersChange({
      commodities: selectedCommodities,
      countries: selectedCountries,
      projectStages: selectedStages,
      marketCapRange,
      npvRange,
      projectCountRange
    })
  }

  const clearAllFilters = () => {
    setSelectedCommodities([])
    setSelectedCountries([])
    setSelectedStages([])
    setMarketCapRange({ min: "", max: "" })
    setNpvRange({ min: "", max: "" })
    setProjectCountRange({ min: "", max: "" })
    onFiltersChange({
      commodities: [],
      countries: [],
      projectStages: [],
      marketCapRange: { min: "", max: "" },
      npvRange: { min: "", max: "" },
      projectCountRange: { min: "", max: "" }
    })
  }

  const activeFiltersCount =
    selectedCommodities.length +
    selectedCountries.length +
    selectedStages.length +
    (marketCapRange.min || marketCapRange.max ? 1 : 0) +
    (npvRange.min || npvRange.max ? 1 : 0) +
    (projectCountRange.min || projectCountRange.max ? 1 : 0)

  return (
    <div className="space-y-4 border rounded-lg p-4 bg-muted/20">
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
            <Button variant="outline" size="sm" className="bg-background">
              Commodities
              {selectedCommodities.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedCommodities.length}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-3">
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {commodities.map((commodity) => (
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

        {/* Project Focus Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              Project Focus
              {selectedStages.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedStages.length}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-3">
            <div className="space-y-2">
              {projectStages.map((stage) => (
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

        {/* Market Cap Range Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              Market Cap
              {(marketCapRange.min || marketCapRange.max) && (
                <Badge variant="secondary" className="ml-2">
                  1
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-3">
            <div className="space-y-2">
              <Label>Market Cap (USD M)</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  placeholder="Min"
                  value={marketCapRange.min}
                  onChange={(e) => setMarketCapRange({ ...marketCapRange, min: e.target.value })}
                />
                <span>-</span>
                <Input
                  type="number"
                  placeholder="Max"
                  value={marketCapRange.max}
                  onChange={(e) => setMarketCapRange({ ...marketCapRange, max: e.target.value })}
                />
              </div>
            </div>
          </PopoverContent>
        </Popover>

        {/* Total NPV Range Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              Total NPV
              {(npvRange.min || npvRange.max) && (
                <Badge variant="secondary" className="ml-2">
                  1
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-3">
            <div className="space-y-2">
              <Label>Total NPV (USD M)</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  placeholder="Min"
                  value={npvRange.min}
                  onChange={(e) => setNpvRange({ ...npvRange, min: e.target.value })}
                />
                <span>-</span>
                <Input
                  type="number"
                  placeholder="Max"
                  value={npvRange.max}
                  onChange={(e) => setNpvRange({ ...npvRange, max: e.target.value })}
                />
              </div>
            </div>
          </PopoverContent>
        </Popover>

        {/* Project Count Range Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              Project Count
              {(projectCountRange.min || projectCountRange.max) && (
                <Badge variant="secondary" className="ml-2">
                  1
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-3">
            <div className="space-y-2">
              <Label>Number of Projects</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  placeholder="Min"
                  value={projectCountRange.min}
                  onChange={(e) => setProjectCountRange({ ...projectCountRange, min: e.target.value })}
                />
                <span>-</span>
                <Input
                  type="number"
                  placeholder="Max"
                  value={projectCountRange.max}
                  onChange={(e) => setProjectCountRange({ ...projectCountRange, max: e.target.value })}
                />
              </div>
            </div>
          </PopoverContent>
        </Popover>

        <Button
          size="sm"
          onClick={applyFilters}
          className="bg-black text-white hover:bg-black/90 dark:bg-white dark:text-black dark:hover:bg-white/90"
        >
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
          {(marketCapRange.min || marketCapRange.max) && (
            <Badge variant="secondary">
              Market Cap: ${marketCapRange.min || "0"} - ${marketCapRange.max || "∞"}M
              <button className="ml-1" onClick={() => setMarketCapRange({ min: "", max: "" })}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
          {(npvRange.min || npvRange.max) && (
            <Badge variant="secondary">
              NPV: ${npvRange.min || "0"} - ${npvRange.max || "∞"}M
              <button className="ml-1" onClick={() => setNpvRange({ min: "", max: "" })}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
          {(projectCountRange.min || projectCountRange.max) && (
            <Badge variant="secondary">
              Projects: {projectCountRange.min || "0"} - {projectCountRange.max || "∞"}
              <button className="ml-1" onClick={() => setProjectCountRange({ min: "", max: "" })}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
        </div>
      )}
    </div>
  )
}
