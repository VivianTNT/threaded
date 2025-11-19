"use client"

import * as React from "react"
import { Filter } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"

export interface FilterOption {
  label: string
  value: string
}

export interface FilterConfig {
  id: string
  label: string
  type: "checkbox" | "range"
  options?: FilterOption[]
  placeholder?: { min?: string; max?: string }
}

export interface RangeValue {
  min: string
  max: string
}

export interface FilterValues {
  [key: string]: string[] | RangeValue
}

interface FilterBoxProps {
  filters: FilterConfig[]
  savedViews?: { id: string; name: string }[]
  defaultViewId?: string
  onApplyFilters: (values: FilterValues) => void
  onClearFilters?: () => void
  onViewChange?: (viewId: string) => void
}

export function FilterBox({
  filters,
  savedViews,
  defaultViewId,
  onApplyFilters,
  onClearFilters,
  onViewChange,
}: FilterBoxProps) {
  const [filterValues, setFilterValues] = React.useState<FilterValues>({})

  React.useEffect(() => {
    // Initialize filter values
    const initialValues: FilterValues = {}
    filters.forEach((filter) => {
      if (filter.type === "checkbox") {
        initialValues[filter.id] = []
      } else if (filter.type === "range") {
        initialValues[filter.id] = { min: "", max: "" }
      }
    })
    setFilterValues(initialValues)
  }, [filters])

  const handleCheckboxChange = (filterId: string, value: string, checked: boolean) => {
    setFilterValues((prev) => {
      const currentValues = (prev[filterId] as string[]) || []
      if (checked) {
        return { ...prev, [filterId]: [...currentValues, value] }
      } else {
        return { ...prev, [filterId]: currentValues.filter((v) => v !== value) }
      }
    })
  }

  const handleRangeChange = (filterId: string, field: "min" | "max", value: string) => {
    setFilterValues((prev) => {
      const currentRange = (prev[filterId] as RangeValue) || { min: "", max: "" }
      return { ...prev, [filterId]: { ...currentRange, [field]: value } }
    })
  }

  const handleApplyFilters = () => {
    onApplyFilters(filterValues)
  }

  const handleClearAll = () => {
    const clearedValues: FilterValues = {}
    filters.forEach((filter) => {
      if (filter.type === "checkbox") {
        clearedValues[filter.id] = []
      } else if (filter.type === "range") {
        clearedValues[filter.id] = { min: "", max: "" }
      }
    })
    setFilterValues(clearedValues)
    if (onClearFilters) {
      onClearFilters()
    }
  }

  return (
    <div className="space-y-4 border rounded-lg p-4 bg-muted/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4" />
          <span className="font-medium">Filters</span>
        </div>
        {savedViews && (
          <Select
            defaultValue={defaultViewId || savedViews[0]?.id}
            onValueChange={onViewChange}
          >
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {savedViews.map((view) => (
                <SelectItem key={view.id} value={view.id}>
                  {view.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {filters.map((filter) => {
          if (filter.type === "checkbox" && filter.options) {
            const selectedCount = (filterValues[filter.id] as string[])?.length || 0
            return (
              <Popover key={filter.id}>
                <PopoverTrigger asChild>
                  <Button variant="outline" size="sm" className="bg-background">
                    {filter.label}
                    {selectedCount > 0 && (
                      <Badge variant="secondary" className="ml-2 h-5">
                        {selectedCount}
                      </Badge>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[200px] p-3" align="start">
                  <div className="space-y-2 max-h-[300px] overflow-y-auto">
                    {filter.options.map((option) => (
                      <div key={option.value} className="flex items-center space-x-2">
                        <Checkbox
                          id={`${filter.id}-${option.value}`}
                          checked={(filterValues[filter.id] as string[])?.includes(option.value)}
                          onCheckedChange={(checked) =>
                            handleCheckboxChange(filter.id, option.value, !!checked)
                          }
                        />
                        <label
                          htmlFor={`${filter.id}-${option.value}`}
                          className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                        >
                          {option.label}
                        </label>
                      </div>
                    ))}
                  </div>
                </PopoverContent>
              </Popover>
            )
          } else if (filter.type === "range") {
            const rangeValue = filterValues[filter.id] as RangeValue
            const hasValue = rangeValue?.min || rangeValue?.max
            return (
              <Popover key={filter.id}>
                <PopoverTrigger asChild>
                  <Button variant="outline" size="sm" className="bg-background">
                    {filter.label}
                    {hasValue && (
                      <Badge variant="secondary" className="ml-2 h-5">
                        1
                      </Badge>
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[250px] p-3" align="start">
                  <div className="space-y-2">
                    <Label className="text-sm font-medium">{filter.label}</Label>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        placeholder={filter.placeholder?.min || "Min"}
                        value={rangeValue?.min || ""}
                        onChange={(e) => handleRangeChange(filter.id, "min", e.target.value)}
                        className="h-8"
                      />
                      <span className="text-muted-foreground">-</span>
                      <Input
                        type="number"
                        placeholder={filter.placeholder?.max || "Max"}
                        value={rangeValue?.max || ""}
                        onChange={(e) => handleRangeChange(filter.id, "max", e.target.value)}
                        className="h-8"
                      />
                    </div>
                  </div>
                </PopoverContent>
              </Popover>
            )
          }
          return null
        })}

        <Button
          size="sm"
          onClick={handleApplyFilters}
          className="bg-black text-white hover:bg-black/90 dark:bg-white dark:text-black dark:hover:bg-white/90"
        >
          Apply Filters
        </Button>
      </div>
    </div>
  )
}
