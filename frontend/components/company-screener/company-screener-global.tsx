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
import { ArrowUpDown, ChevronDown, Eye, Plus, Search, Bookmark, BookmarkCheck } from "lucide-react"
import { LinksPopover } from "@/components/ui/links-popover"

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
import { useCompanies, Company } from "@/lib/hooks/use-companies"
import { CompanyDetailPanel } from "@/components/company-detail-panel"
import { Skeleton } from "@/components/ui/skeleton"
import { useState, useEffect } from "react"
import { Loader2 } from "lucide-react"
import { ExportDropdown, ExportFormat } from "@/components/ui/export-dropdown"
import { toast } from "sonner"
import { Toaster } from "@/components/ui/toaster"
import { supabase } from "@/lib/supabase/client"
import { BulkActionsToolbar } from "./bulk-actions-toolbar"

export function CompanyScreenerGlobal() {
  const { companies: initialData, loading, error, refetch } = useCompanies()
  const [data, setData] = useState<Company[]>(initialData)
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState({})
  const [globalFilter, setGlobalFilter] = useState("")

  // Sync data with initialData when it changes
  useEffect(() => {
    setData(initialData)
  }, [initialData])

  // Company detail panel state
  const [detailPanelOpen, setDetailPanelOpen] = useState(false)
  const [selectedCompanies, setSelectedCompanies] = useState<Company[]>([])
  const [detailPanelMode, setDetailPanelMode] = useState<"single" | "comparison">("single")
  const [updatingWatchlist, setUpdatingWatchlist] = useState<string | null>(null)

  // Listen for refresh events
  useEffect(() => {
    const handleRefreshCompanies = () => {
      refetch()
    }

    window.addEventListener('refreshCompanies', handleRefreshCompanies)

    return () => {
      window.removeEventListener('refreshCompanies', handleRefreshCompanies)
    }
  }, [refetch])

  const handleCompanyClick = (companyId: string) => {
    const company = data.find(c => String(c.id) === String(companyId))

    if (company) {
      setSelectedCompanies([company])
      setDetailPanelMode("single")
      setDetailPanelOpen(true)
    }
  }

  const handleCompanyAnalysis = () => {
    const selectedRows = table.getFilteredSelectedRowModel().rows
    if (selectedRows.length === 1) {
      setSelectedCompanies(selectedRows.map(row => row.original))
      setDetailPanelMode("single")
      setDetailPanelOpen(true)
    }
  }

  const handleCompareCompanies = () => {
    const selectedRows = table.getFilteredSelectedRowModel().rows
    if (selectedRows.length >= 2) {
      setSelectedCompanies(selectedRows.map(row => row.original))
      setDetailPanelMode("comparison")
      setDetailPanelOpen(true)
    }
  }

  const handleClearSelection = () => {
    setRowSelection({})
  }

  // Watchlist handler
  const handleToggleWatchlist = async (company: Company) => {
    try {
      setUpdatingWatchlist(company.id)
      const newWatchlistStatus = !company.watchlist

      // Optimistically update local state
      const updatedData = data.map(c =>
        c.id === company.id
          ? { ...c, watchlist: newWatchlistStatus }
          : c
      )
      setData(updatedData)

      const { error } = await supabase
        .from('companies')
        .update({ watchlist: newWatchlistStatus })
        .eq('id', company.id)

      if (error) {
        // Revert on error
        const revertedData = data.map(c =>
          c.id === company.id
            ? { ...c, watchlist: !newWatchlistStatus }
            : c
        )
        setData(revertedData)
        throw error
      }

      toast.success(newWatchlistStatus ? 'Added to watchlist' : 'Removed from watchlist')
    } catch (error: any) {
      console.error('Error updating watchlist:', error)
      toast.error(`Failed to update watchlist: ${error?.message || 'Unknown error'}`)
    } finally {
      setUpdatingWatchlist(null)
    }
  }

  const handleExport = (format: ExportFormat) => {
    const dataToExport = table.getFilteredRowModel().rows.map(row => row.original)
    // Simple export implementation
    const jsonData = JSON.stringify(dataToExport, null, 2)
    const blob = new Blob([jsonData], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `companies-export.${format}`
    a.click()
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
      id: "watchlist",
      header: "",
      cell: ({ row }) => {
        const company = row.original
        const isUpdating = updatingWatchlist === company.id

        return (
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation()
              handleToggleWatchlist(company)
            }}
            disabled={isUpdating}
            className="hover:bg-transparent"
          >
            {isUpdating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : company.watchlist ? (
              <BookmarkCheck className="h-4 w-4 fill-foreground" />
            ) : (
              <Bookmark className="h-4 w-4 text-muted-foreground hover:text-foreground" />
            )}
          </Button>
        )
      },
      size: 50,
    },
    {
      accessorKey: "name",
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Company Name
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => (
        <div className="space-y-1">
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault()
              handleCompanyClick(row.original.id)
            }}
            className="text-sm font-semibold text-blue-600 hover:underline block"
          >
            {row.original.name}
          </a>
          {row.original.ticker && row.original.exchange && (
            <div className="text-xs text-gray-500">
              {row.original.ticker} • {row.original.exchange}
            </div>
          )}
        </div>
      ),
    },
    {
      accessorKey: "exchange",
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Exchange
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => {
        const exchange = row.original.exchange
        return exchange ? (
          <Badge variant="outline" className="font-normal">
            {exchange}
          </Badge>
        ) : (
          <span className="text-sm text-muted-foreground">N/A</span>
        )
      },
    },
    {
      accessorKey: "country",
      header: "Country",
      cell: ({ row }) => (
        <div className="text-sm">{row.original.country || 'N/A'}</div>
      ),
    },
    {
      accessorKey: "market_cap",
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Market Cap
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => {
        const marketCap = row.original.market_cap
        if (!marketCap) return <span className="text-sm text-muted-foreground">N/A</span>

        const formatted = marketCap >= 1e9
          ? `$${(marketCap / 1e9).toFixed(2)}B`
          : marketCap >= 1e6
          ? `$${(marketCap / 1e6).toFixed(2)}M`
          : `$${marketCap.toLocaleString()}`

        return <div className="text-sm font-medium text-right">{formatted}</div>
      },
    },
    {
      accessorKey: "description",
      header: "Description",
      cell: ({ row }) => (
        <div className="text-sm max-w-[300px] truncate" title={row.original.description || undefined}>
          {row.original.description || 'N/A'}
        </div>
      ),
    },
    {
      accessorKey: "website",
      header: "Links",
      cell: ({ row }) => {
        const website = row.original.website
        const urls = row.original.urls

        // Combine urls array and website into a single array
        const allLinks = []
        if (urls && urls.length > 0) {
          allLinks.push(...urls)
        } else if (website) {
          allLinks.push(website)
        }

        return <LinksPopover urls={allLinks} />
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
          <h2 className="text-2xl font-semibold tracking-tight">Global Companies — {data.length} companies</h2>
          <div className="flex items-center gap-2">
            <ExportDropdown onExport={handleExport} />
            <Button size="sm">
              <Plus className="mr-2 h-4 w-4" />
              Add Company
            </Button>
          </div>
        </div>

        {/* Bulk Actions Toolbar */}
        {selectedRowsCount > 0 && (
          <BulkActionsToolbar
            selectedCount={selectedRowsCount}
            selectedCompanies={table.getFilteredSelectedRowModel().rows.map(row => row.original)}
            onClearSelection={handleClearSelection}
            onCompanyAnalysis={selectedRowsCount === 1 ? handleCompanyAnalysis : undefined}
            onCompare={selectedRowsCount >= 2 ? handleCompareCompanies : undefined}
          />
        )}

        <div className="flex items-center justify-between py-2">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search all columns..."
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

        <div className="rounded-lg border relative overflow-hidden">
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
                    No results.
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

      {selectedCompanies.length > 0 && (
        <CompanyDetailPanel
          isOpen={detailPanelOpen}
          onClose={() => setDetailPanelOpen(false)}
          companies={selectedCompanies}
          mode={detailPanelMode}
        />
      )}
      <Toaster />
    </>
  )
}
