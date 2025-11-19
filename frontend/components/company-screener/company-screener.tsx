'use client'

import { useState, useEffect } from "react"
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
  getPaginationRowModel,
  SortingState,
  getSortedRowModel,
  ColumnFiltersState,
  getFilteredRowModel,
  VisibilityState,
  RowSelectionState,
} from "@tanstack/react-table"
import { ArrowUpDown, Bookmark, Loader2, BookmarkCheck, Search, ChevronDown } from "lucide-react"
import { LinksPopover } from "@/components/ui/links-popover"
import { Checkbox } from "@/components/ui/checkbox"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ExportDropdown, ExportFormat } from "@/components/ui/export-dropdown"
import { exportProjects } from "@/lib/export-utils"
import { Button } from "@/components/ui/button"
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
import { useWatchlistCompanies, Company } from "@/lib/hooks/use-watchlist-companies"
import { CompanyDetailPanel } from "@/components/company-detail-panel"
import { cn } from "@/lib/utils"
import { supabase } from "@/lib/supabase/client"
import { toast } from "sonner"
import { BulkActionsToolbar } from "./bulk-actions-toolbar"

export function CompanyScreener() {
  const { companies: initialData, loading, error, refetch } = useWatchlistCompanies()
  const [data, setData] = useState<Company[]>(initialData)
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})
  const [globalFilter, setGlobalFilter] = useState("")
  const [selectedCompanies, setSelectedCompanies] = useState<Company[]>([])
  const [isPanelOpen, setIsPanelOpen] = useState(false)
  const [detailPanelMode, setDetailPanelMode] = useState<"single" | "comparison">("single")
  const [updatingWatchlist, setUpdatingWatchlist] = useState<string | null>(null)

  // Sync data with initialData when it changes
  useEffect(() => {
    setData(initialData)
  }, [initialData])

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
    const company = data.find((p) => p.id === companyId)
    if (company) {
      setSelectedCompanies([company])
      setDetailPanelMode("single")
      setIsPanelOpen(true)
    }
  }

  const handleCompanyAnalysis = () => {
    const selectedRows = table.getFilteredSelectedRowModel().rows
    if (selectedRows.length === 1) {
      setSelectedCompanies(selectedRows.map(row => row.original))
      setDetailPanelMode("single")
      setIsPanelOpen(true)
    }
  }

  const handleCompareCompanies = () => {
    const selectedRows = table.getFilteredSelectedRowModel().rows
    if (selectedRows.length >= 2) {
      setSelectedCompanies(selectedRows.map(row => row.original))
      setDetailPanelMode("comparison")
      setIsPanelOpen(true)
    }
  }

  const handleClearSelection = () => {
    setRowSelection({})
  }

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
    exportProjects(dataToExport as any, format, 'watchlisted-companies')
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
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Name
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const company = row.original
        return (
          <div className="space-y-1">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                handleCompanyClick(company.id)
              }}
              className="text-sm font-semibold text-blue-600 hover:underline block"
            >
              {company.name}
            </a>
            {company.ticker && (
              <div className="text-xs text-gray-500">
                {company.ticker}
              </div>
            )}
          </div>
        )
      },
    },
    {
      accessorKey: "exchange",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Exchange
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const exchange = row.original.exchange
        return exchange ? (
          <Badge variant="outline" className="text-xs">
            {exchange}
          </Badge>
        ) : (
          <span className="text-sm text-muted-foreground">N/A</span>
        )
      },
    },
    {
      accessorKey: "country",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Country
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        return (
          <div className="text-sm">
            {row.original.country || 'N/A'}
          </div>
        )
      },
    },
    {
      accessorKey: "market_cap",
      header: ({ column }) => {
        return (
          <Button
            variant="ghost"
            onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          >
            Market Cap
            <ArrowUpDown className="ml-2 h-4 w-4" />
          </Button>
        )
      },
      cell: ({ row }) => {
        const marketCap = row.original.market_cap
        if (!marketCap) return <div className="text-sm text-muted-foreground text-right">N/A</div>

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
      cell: ({ row }) => {
        const description = row.original.description
        return (
          <div className="max-w-md text-sm text-muted-foreground truncate">
            {description || 'N/A'}
          </div>
        )
      },
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
      <div className="flex items-center justify-center h-96">
        <div className="text-muted-foreground">Loading companies...</div>
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
          <div className="text-sm text-muted-foreground">{data.length} companies</div>
          <div className="flex items-center gap-2">
            <ExportDropdown onExport={handleExport} />
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
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  return (
                    <TableHead key={header.id}>
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext()
                          )}
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
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => handleCompanyClick(row.original.id)}
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
                  No watchlisted companies found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
      <div className="flex items-center justify-between py-4">
        <div className="text-sm text-muted-foreground">
          Showing {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1} to{' '}
          {Math.min(
            (table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize,
            table.getFilteredRowModel().rows.length
          )}{' '}
          of {table.getFilteredRowModel().rows.length} results
        </div>
        <div className="flex items-center space-x-2">
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
          isOpen={isPanelOpen}
          onClose={() => setIsPanelOpen(false)}
          companies={selectedCompanies}
          mode={detailPanelMode}
        />
      )}
    </>
  )
}
