"use client"

import * as React from "react"
import {
  ColumnDef,
  ColumnFiltersState,
  SortingState,
  VisibilityState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table"
import { ArrowUpDown, ChevronDown, Download, Eye, Filter, Plus, Search, Building2, TrendingUp, Globe, Users } from "lucide-react"
import { ContextMenuChat } from "@/components/ui/context-menu-chat"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { useState, useEffect } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { ExportDropdown, ExportFormat } from "@/components/ui/export-dropdown"
import { exportProjects } from "@/lib/export-utils"
import { toast } from "sonner"
import { Toaster } from "@/components/ui/toaster"
import { supabase } from "@/lib/supabase/client"
import { formatNumber, formatCurrency, formatPercent } from "@/lib/format-utils"
// import { CompanyDetailPanel } from "@/components/companies/company-detail-panel"
import { CompaniesFilters } from "@/components/companies/companies-filters"

export interface Company {
  company_id: string
  company_name: string
  headquarters_location?: string
  company_country?: string
  website_url?: string
  stock_ticker?: string
  exchange?: string
  market_cap_usd_m?: number
  description?: string
  logo_url?: string
  founded_year?: number
  employees_count?: number
  
  // Aggregated metrics
  total_projects: number
  active_projects: number
  production_projects: number
  development_projects: number
  exploration_projects: number
  
  total_npv_usd_m?: number
  avg_irr_percent?: number
  total_capex_usd_m?: number
  total_resources_tonnes?: number
  avg_mine_life_years?: number
  
  primary_commodities?: string[]
  operating_countries?: string[]
  countries_count?: number
  
  overall_risk_level?: string
  typical_esg_score?: string
  
  has_watchlisted_projects?: boolean
  watchlisted_projects_count?: number
  
  updated_at?: string
}

function getRiskBadgeColor(risk?: string) {
  switch (risk) {
    case "Low":
      return "bg-green-100 text-green-800 hover:bg-green-100"
    case "Medium":
      return "bg-yellow-100 text-yellow-800 hover:bg-yellow-100"
    case "High":
      return "bg-orange-100 text-orange-800 hover:bg-orange-100"
    case "Very High":
      return "bg-red-100 text-red-800 hover:bg-red-100"
    default:
      return "bg-gray-100 text-gray-800"
  }
}

export function CompaniesTable() {
  const [data, setData] = useState<Company[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState({})
  const [globalFilter, setGlobalFilter] = useState("")
  
  // Detail panel state
  const [detailPanelOpen, setDetailPanelOpen] = useState(false)
  const [selectedCompany, setSelectedCompany] = useState<Company | null>(null)

  // Fetch companies data
  const fetchCompanies = async () => {
    try {
      setLoading(true)
      
      // Try to fetch from the materialized view first
      let { data: companies, error } = await supabase
        .from('companies_summary')
        .select('*')
        .order('total_npv_usd_m', { ascending: false, nullsFirst: false })
      
      // If materialized view doesn't exist, fall back to the regular view
      if (error) {
        const { data: fallbackData, error: fallbackError } = await supabase
          .from('companies_with_metrics')
          .select('*')
          .order('total_npv_usd_m', { ascending: false, nullsFirst: false })
        
        if (fallbackError) {
          // If both fail, aggregate from projects table
          const { data: projectsData, error: projectsError } = await supabase
            .from('projects')
            .select('company_name, post_tax_npv_usd_m, irr_percent, capex_usd_m, stage, primary_commodity, country, jurisdiction_risk, esg_score, watchlist')
          
          if (projectsError) throw projectsError
          
          // Aggregate projects by company
          const companiesMap = new Map<string, Company>()
          
          projectsData?.forEach(project => {
            if (!project.company_name) return
            
            if (!companiesMap.has(project.company_name)) {
              companiesMap.set(project.company_name, {
                company_id: `temp-${project.company_name}`,
                company_name: project.company_name,
                total_projects: 0,
                active_projects: 0,
                production_projects: 0,
                development_projects: 0,
                exploration_projects: 0,
                total_npv_usd_m: 0,
                total_capex_usd_m: 0,
                primary_commodities: [],
                operating_countries: [],
                countries_count: 0
              })
            }
            
            const company = companiesMap.get(project.company_name)!
            company.total_projects++
            
            if (project.stage === 'Production') company.production_projects++
            else if (project.stage === 'Development' || project.stage === 'Construction') company.development_projects++
            else if (project.stage === 'Exploration' || project.stage === 'Resource Definition') company.exploration_projects++
            
            if (project.post_tax_npv_usd_m) company.total_npv_usd_m = (company.total_npv_usd_m || 0) + project.post_tax_npv_usd_m
            if (project.capex_usd_m) company.total_capex_usd_m = (company.total_capex_usd_m || 0) + project.capex_usd_m
            
            if (project.primary_commodity && !company.primary_commodities?.includes(project.primary_commodity)) {
              company.primary_commodities?.push(project.primary_commodity)
            }
            
            if (project.country && !company.operating_countries?.includes(project.country)) {
              company.operating_countries?.push(project.country)
            }
          })
          
          companies = Array.from(companiesMap.values())
        } else {
          companies = fallbackData
        }
      }
      
      setData(companies || [])
    } catch (err: any) {
      console.error('Error fetching companies:', err)
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCompanies()
  }, [])

  const handleCompanyClick = (company: Company) => {
    setSelectedCompany(company)
    setDetailPanelOpen(true)
  }

  const handleExport = (format: ExportFormat) => {
    const dataToExport = table.getFilteredRowModel().rows.map(row => row.original)
    // Use a modified export function for companies
    const exportData = dataToExport.map(company => ({
      'Company Name': company.company_name,
      'Headquarters': company.headquarters_location || 'N/A',
      'Stock Ticker': company.stock_ticker || 'N/A',
      'Total Projects': company.total_projects,
      'Production Projects': company.production_projects,
      'Total NPV (USD M)': company.total_npv_usd_m || 0,
      'Avg IRR %': company.avg_irr_percent || 0,
      'Primary Commodities': company.primary_commodities?.join(', ') || 'N/A',
      'Operating Countries': company.operating_countries?.join(', ') || 'N/A'
    }))
    
    // Reuse the export function with modified data
    exportProjects(exportData as any, format, 'companies')
    toast.success(`Exported ${dataToExport.length} companies as ${format.toUpperCase()}`)
  }

  const columns: ColumnDef<Company>[] = [
    {
      id: "select",
      header: ({ table }) => (
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && "indeterminate")
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
          aria-label="Select all"
        />
      ),
      cell: ({ row }) => (
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => row.toggleSelected(!!value)}
          aria-label="Select row"
        />
      ),
      enableSorting: false,
      enableHiding: false,
    },
    {
      accessorKey: "company_name",
      header: "Company",
      cell: ({ row }) => (
        <ContextMenuChat
          data={row.original}
          dataType="company"
          context={row.original.company_name}
        >
          <div className="space-y-1">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                handleCompanyClick(row.original)
              }}
              className="text-sm font-semibold text-blue-600 hover:underline block"
            >
              {row.getValue("company_name")}
            </a>
            <div className="flex items-center gap-2">
              {row.original.stock_ticker && (
                <Badge variant="outline" className="text-xs">
                  {row.original.stock_ticker}
                </Badge>
              )}
              {row.original.exchange && (
                <span className="text-xs text-gray-500">{row.original.exchange}</span>
              )}
            </div>
          </div>
        </ContextMenuChat>
      ),
    },
    {
      accessorKey: "headquarters_location",
      header: "Headquarters",
      cell: ({ row }) => (
        <div className="space-y-1">
          <div className="text-sm">{row.original.headquarters_location || 'N/A'}</div>
          <div className="text-xs text-gray-500">{row.original.company_country || ''}</div>
        </div>
      ),
    },
    {
      accessorKey: "total_projects",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Projects
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const company = row.original
        return (
          <div className="space-y-1">
            <div className="text-sm font-semibold">{company.total_projects} total</div>
            <div className="text-xs text-gray-500">
              {company.production_projects > 0 && `${company.production_projects} prod`}
              {company.production_projects > 0 && company.development_projects > 0 && ', '}
              {company.development_projects > 0 && `${company.development_projects} dev`}
            </div>
          </div>
        )
      },
    },
    {
      accessorKey: "total_npv_usd_m",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
            className="flex items-center"
          >
            <span>Total NPV (USD M)</span>
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const amount = row.getValue("total_npv_usd_m") as number | null
        const formatted = formatCurrency(amount, { decimals: 0, unit: 'M' })
        return <div className="text-sm text-right font-medium">{formatted}</div>
      },
    },
    {
      accessorKey: "avg_irr_percent",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Avg IRR %
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const irr = row.getValue("avg_irr_percent") as number | null
        const formatted = formatPercent(irr, { decimals: 1 })
        return (
          <div className={cn("text-sm text-center", 
            irr && irr >= 30 ? "text-green-600 font-semibold" :
            irr && irr >= 20 ? "text-yellow-600" :
            "text-red-600"
          )}>
            {formatted}
          </div>
        )
      },
    },
    {
      accessorKey: "primary_commodities",
      header: "Commodities",
      cell: ({ row }) => {
        const commodities = row.original.primary_commodities || []
        if (commodities.length === 0) return <span className="text-sm text-gray-400">N/A</span>
        
        return (
          <div className="flex flex-wrap gap-1">
            {commodities.slice(0, 3).map((commodity, i) => (
              <Badge key={i} variant="outline" className="text-xs">
                {commodity}
              </Badge>
            ))}
            {commodities.length > 3 && (
              <Badge variant="secondary" className="text-xs">
                +{commodities.length - 3}
              </Badge>
            )}
          </div>
        )
      },
    },
    {
      accessorKey: "operating_countries",
      header: "Geographic Presence",
      cell: ({ row }) => {
        const countries = row.original.operating_countries || []
        const count = row.original.countries_count || countries.length

        return (
          <div className="space-y-1">
            <div className="text-sm font-semibold">{count} countries</div>
            <div className="text-xs text-gray-500">
              {countries.slice(0, 2).join(', ')}
              {countries.length > 2 && ` +${countries.length - 2}`}
            </div>
          </div>
        )
      },
    },
    {
      accessorKey: "market_cap_usd_m",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Market Cap (USD M)
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const amount = row.getValue("market_cap_usd_m") as number | null
        const formatted = formatCurrency(amount, { decimals: 0, unit: 'M' })
        return <div className="text-sm text-right">{formatted}</div>
      },
    },
    {
      accessorKey: "overall_risk_level",
      header: "Risk Profile",
      cell: ({ row }) => {
        const risk = row.original.overall_risk_level
        const esg = row.original.typical_esg_score
        
        return (
          <div className="space-y-1">
            {risk && (
              <Badge className={cn("text-xs", getRiskBadgeColor(risk))}>
                {risk} Risk
              </Badge>
            )}
            {esg && (
              <Badge variant="outline" className="text-xs">
                ESG: {esg}
              </Badge>
            )}
          </div>
        )
      },
    },
  ]

  const table = useReactTable({
    data,
    columns,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    getCoreRowModel: getCoreRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    onColumnVisibilityChange: setColumnVisibility,
    onRowSelectionChange: setRowSelection,
    onGlobalFilterChange: setGlobalFilter,
    globalFilterFn: "includesString",
    state: {
      sorting,
      columnFilters,
      columnVisibility,
      rowSelection,
      globalFilter,
    },
  })

  const selectedRowsCount = Object.keys(rowSelection).length

  if (loading) {
    return (
      <div className="w-full space-y-4">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-96 w-full" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="w-full p-8 text-center">
        <p className="text-muted-foreground">Failed to load companies: {error}</p>
      </div>
    )
  }

  return (
    <>
      <div className="w-full space-y-4 relative">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold tracking-tight">Companies — {data.length} organizations</h2>
          <div className="flex items-center gap-2">
            <ExportDropdown onExport={handleExport} />
            <Button variant="outline" size="sm">
              <Eye className="mr-2 h-4 w-4" />
              View Company Reports
            </Button>
            <Button size="sm">
              <Plus className="mr-2 h-4 w-4" />
              Add Company
            </Button>
          </div>
        </div>

        <CompaniesFilters 
          onFiltersChange={(filters) => {
            // Apply filters to the table
            // This would need to be implemented based on your filter logic
          }}
        />

        <div className="flex items-center justify-between py-2">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search companies..."
                value={globalFilter ?? ""}
                onChange={(event) => setGlobalFilter(event.target.value)}
                className="pl-8 w-[300px]"
              />
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  Columns <ChevronDown className="ml-2 h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-[200px]">
                <DropdownMenuLabel>Toggle columns</DropdownMenuLabel>
                <DropdownMenuSeparator />
                {table
                  .getAllColumns()
                  .filter((column) => column.getCanHide())
                  .map((column) => {
                    return (
                      <DropdownMenuCheckboxItem
                        key={column.id}
                        className="capitalize"
                        checked={column.getIsVisible()}
                        onCheckedChange={(value) => column.toggleVisibility(!!value)}
                      >
                        {column.id}
                      </DropdownMenuCheckboxItem>
                    )
                  })}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <div className="text-sm text-muted-foreground">
            {selectedRowsCount} of {table.getFilteredRowModel().rows.length} row(s) selected.
          </div>
        </div>

        <div className="rounded-lg border">
          <Table>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => {
                    return (
                      <TableHead key={header.id}>
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                      </TableHead>
                    )
                  })}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    data-state={row.getIsSelected() && "selected"}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id}>
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-24 text-center">
                    No companies found.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        <div className="flex items-center justify-end space-x-2 py-4">
          <div className="flex-1 text-sm text-muted-foreground">
            Showing {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1} to{" "}
            {Math.min(
              (table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize,
              table.getFilteredRowModel().rows.length
            )}{" "}
            of {table.getFilteredRowModel().rows.length} results
          </div>
          <div className="space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => table.previousPage()}
              disabled={!table.getCanPreviousPage()}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => table.nextPage()}
              disabled={!table.getCanNextPage()}
            >
              Next
            </Button>
          </div>
        </div>
      </div>

      {/* Company Detail Panel - TODO: Implement */}
      {/* <CompanyDetailPanel
        isOpen={detailPanelOpen}
        onClose={() => setDetailPanelOpen(false)}
        company={selectedCompany}
      /> */}
      <Toaster />
    </>
  )
}
