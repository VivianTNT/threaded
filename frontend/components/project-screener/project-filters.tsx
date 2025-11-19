"use client"

import * as React from "react"
import { Table } from "@tanstack/react-table"
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
import { MiningProject, ProjectStage, Commodity, RiskLevel } from "@/lib/types/mining-project"

interface ProjectFiltersProps {
  table: Table<MiningProject>
}

const stages: ProjectStage[] = [
  "Exploration",
  "PEA",
  "PFS",
  "DFS",
  "Development",
  "Production",
  "Care & Maintenance",
]

const commodities: Commodity[] = [
  "Lithium",
  "Copper",
  "Nickel",
  "Cobalt",
  "Graphite",
  "Rare Earths",
  "Uranium",
  "Vanadium",
  "Manganese",
  "Tin",
  "Tungsten",
  "Molybdenum",
  "Neodymium",
  "Cerium",
]

const riskLevels: RiskLevel[] = ["Low", "Medium", "High", "Very High"]

const savedViews = [
  { id: "1", name: "All Projects" },
  { id: "2", name: "Lithium Production" },
  { id: "3", name: "High IRR (>30%)" },
  { id: "4", name: "Low Risk Jurisdictions" },
  { id: "5", name: "Development Stage" },
]

export function ProjectFilters({ table }: ProjectFiltersProps) {
  const [selectedStages, setSelectedStages] = React.useState<ProjectStage[]>([])
  const [selectedCommodities, setSelectedCommodities] = React.useState<Commodity[]>([])
  const [selectedRisks, setSelectedRisks] = React.useState<RiskLevel[]>([])
  const [npvRange, setNpvRange] = React.useState({ min: "", max: "" })
  const [irrRange, setIrrRange] = React.useState({ min: "", max: "" })
  const [capexRange, setCapexRange] = React.useState({ min: "", max: "" })

  const applyFilters = () => {
    // Apply stage filter
    if (selectedStages.length > 0) {
      table.getColumn("stage")?.setFilterValue((value: ProjectStage) =>
        selectedStages.includes(value)
      )
    } else {
      table.getColumn("stage")?.setFilterValue(undefined)
    }

    // Apply commodity filter
    if (selectedCommodities.length > 0) {
      table.getColumn("primaryCommodity")?.setFilterValue((value: Commodity) =>
        selectedCommodities.includes(value)
      )
    } else {
      table.getColumn("primaryCommodity")?.setFilterValue(undefined)
    }

    // Apply numeric range filters
    if (npvRange.min || npvRange.max) {
      table.getColumn("postTaxNPV")?.setFilterValue((value: number) => {
        const min = npvRange.min ? parseFloat(npvRange.min) : -Infinity
        const max = npvRange.max ? parseFloat(npvRange.max) : Infinity
        return value >= min && value <= max
      })
    } else {
      table.getColumn("postTaxNPV")?.setFilterValue(undefined)
    }

    if (irrRange.min || irrRange.max) {
      table.getColumn("irr")?.setFilterValue((value: number) => {
        const min = irrRange.min ? parseFloat(irrRange.min) : -Infinity
        const max = irrRange.max ? parseFloat(irrRange.max) : Infinity
        return value >= min && value <= max
      })
    } else {
      table.getColumn("irr")?.setFilterValue(undefined)
    }

    if (capexRange.min || capexRange.max) {
      table.getColumn("capex")?.setFilterValue((value: number) => {
        const min = capexRange.min ? parseFloat(capexRange.min) : -Infinity
        const max = capexRange.max ? parseFloat(capexRange.max) : Infinity
        return value >= min && value <= max
      })
    } else {
      table.getColumn("capex")?.setFilterValue(undefined)
    }
  }

  const clearAllFilters = () => {
    setSelectedStages([])
    setSelectedCommodities([])
    setSelectedRisks([])
    setNpvRange({ min: "", max: "" })
    setIrrRange({ min: "", max: "" })
    setCapexRange({ min: "", max: "" })
    table.resetColumnFilters()
  }

  const activeFiltersCount =
    selectedStages.length +
    selectedCommodities.length +
    selectedRisks.length +
    (npvRange.min || npvRange.max ? 1 : 0) +
    (irrRange.min || irrRange.max ? 1 : 0) +
    (capexRange.min || capexRange.max ? 1 : 0)

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
        {/* Stage Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
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

        {/* Commodity Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
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

        {/* Risk Level Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              Risk Level
              {selectedRisks.length > 0 && (
                <Badge variant="secondary" className="ml-2">
                  {selectedRisks.length}
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[200px] p-3">
            <div className="space-y-2">
              {riskLevels.map((risk) => (
                <div key={risk} className="flex items-center space-x-2">
                  <Checkbox
                    id={risk}
                    checked={selectedRisks.includes(risk)}
                    onCheckedChange={(checked) => {
                      if (checked) {
                        setSelectedRisks([...selectedRisks, risk])
                      } else {
                        setSelectedRisks(selectedRisks.filter((r) => r !== risk))
                      }
                    }}
                  />
                  <label
                    htmlFor={risk}
                    className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                  >
                    {risk}
                  </label>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>

        {/* NPV Range Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              NPV Range
              {(npvRange.min || npvRange.max) && (
                <Badge variant="secondary" className="ml-2">
                  1
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-3">
            <div className="space-y-2">
              <Label>Post-tax NPV (USD M)</Label>
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

        {/* IRR Range Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              IRR Range
              {(irrRange.min || irrRange.max) && (
                <Badge variant="secondary" className="ml-2">
                  1
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-3">
            <div className="space-y-2">
              <Label>IRR %</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  placeholder="Min"
                  value={irrRange.min}
                  onChange={(e) => setIrrRange({ ...irrRange, min: e.target.value })}
                />
                <span>-</span>
                <Input
                  type="number"
                  placeholder="Max"
                  value={irrRange.max}
                  onChange={(e) => setIrrRange({ ...irrRange, max: e.target.value })}
                />
              </div>
            </div>
          </PopoverContent>
        </Popover>

        {/* CAPEX Range Filter */}
        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="bg-background">
              CAPEX Range
              {(capexRange.min || capexRange.max) && (
                <Badge variant="secondary" className="ml-2">
                  1
                </Badge>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-[250px] p-3">
            <div className="space-y-2">
              <Label>CAPEX (USD M)</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  placeholder="Min"
                  value={capexRange.min}
                  onChange={(e) => setCapexRange({ ...capexRange, min: e.target.value })}
                />
                <span>-</span>
                <Input
                  type="number"
                  placeholder="Max"
                  value={capexRange.max}
                  onChange={(e) => setCapexRange({ ...capexRange, max: e.target.value })}
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
          {selectedStages.map((stage) => (
            <Badge key={stage} variant="secondary">
              Stage: {stage}
              <button
                className="ml-1"
                onClick={() => setSelectedStages(selectedStages.filter((s) => s !== stage))}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {selectedCommodities.map((commodity) => (
            <Badge key={commodity} variant="secondary">
              Commodity: {commodity}
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
          {selectedRisks.map((risk) => (
            <Badge key={risk} variant="secondary">
              Risk: {risk}
              <button
                className="ml-1"
                onClick={() => setSelectedRisks(selectedRisks.filter((r) => r !== risk))}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {(npvRange.min || npvRange.max) && (
            <Badge variant="secondary">
              NPV: ${npvRange.min || "0"} - ${npvRange.max || "∞"}M
              <button className="ml-1" onClick={() => setNpvRange({ min: "", max: "" })}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
          {(irrRange.min || irrRange.max) && (
            <Badge variant="secondary">
              IRR: {irrRange.min || "0"}% - {irrRange.max || "∞"}%
              <button className="ml-1" onClick={() => setIrrRange({ min: "", max: "" })}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
          {(capexRange.min || capexRange.max) && (
            <Badge variant="secondary">
              CAPEX: ${capexRange.min || "0"} - ${capexRange.max || "∞"}M
              <button className="ml-1" onClick={() => setCapexRange({ min: "", max: "" })}>
                <X className="h-3 w-3" />
              </button>
            </Badge>
          )}
        </div>
      )}
    </div>
  )
} 