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
import { ArrowUpDown, Bookmark, Loader2, BookmarkCheck, Search, ChevronDown, RefreshCw } from "lucide-react"
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
import { cn } from "@/lib/utils"
import { supabase } from "@/lib/supabase/client"
import { toast } from "sonner"
import { Toaster } from "@/components/ui/toaster"
import { Skeleton } from "@/components/ui/skeleton"
import { useNews, NewsItem } from "@/lib/hooks/use-news"
import { format } from "date-fns"

export function NewsScreenerGlobal() {
  const { news: initialData, loading, error, refetch } = useNews()
  const [data, setData] = useState<NewsItem[]>(initialData)
  const [sorting, setSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({})
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({})
  const [globalFilter, setGlobalFilter] = useState("")
  const [updatingWatchlist, setUpdatingWatchlist] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  // Sync data with initialData when it changes
  useEffect(() => {
    setData(initialData)
  }, [initialData])

  // Listen for refresh events
  useEffect(() => {
    const handleRefreshNews = () => {
      refetch()
    }

    window.addEventListener('refreshNews', handleRefreshNews)

    return () => {
      window.removeEventListener('refreshNews', handleRefreshNews)
    }
  }, [refetch])

  const handleToggleWatchlist = async (newsItem: NewsItem) => {
    try {
      setUpdatingWatchlist(newsItem.id)
      const newWatchlistStatus = !newsItem.watchlist

      // Optimistically update local state
      const updatedData = data.map(n =>
        n.id === newsItem.id
          ? { ...n, watchlist: newWatchlistStatus }
          : n
      )
      setData(updatedData)

      const { error } = await supabase
        .from('news')
        .update({ watchlist: newWatchlistStatus })
        .eq('id', newsItem.id)

      if (error) {
        // Revert on error
        const revertedData = data.map(n =>
          n.id === newsItem.id
            ? { ...n, watchlist: !newWatchlistStatus }
            : n
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
    exportProjects(dataToExport as any, format, 'global-news')
    toast.success(`Exported ${dataToExport.length} news items as ${format.toUpperCase()}`)
  }

  const handleRefreshNews = async () => {
    try {
      setRefreshing(true)
      toast.info('Fetching latest news...')

      const response = await fetch('/api/news/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ maxPerSource: 10 }),
      })

      const result = await response.json()

      if (result.success) {
        toast.success(`Refreshed! Added ${result.totalSaved} new articles from ${result.sources.length} sources`)
        refetch()
      } else {
        toast.error(result.message || 'Failed to refresh news')
      }
    } catch (error: any) {
      console.error('Error refreshing news:', error)
      toast.error('Failed to refresh news')
    } finally {
      setRefreshing(false)
    }
  }

  const columns: ColumnDef<NewsItem>[] = [
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
        const newsItem = row.original
        const isUpdating = updatingWatchlist === newsItem.id

        return (
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation()
              handleToggleWatchlist(newsItem)
            }}
            disabled={isUpdating}
            className="hover:bg-transparent"
          >
            {isUpdating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : newsItem.watchlist ? (
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
      accessorKey: "title",
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Title
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => (
        <div className="space-y-1">
          <div className="text-sm font-semibold text-blue-600">
            {row.original.title}
          </div>
          {row.original.source && (
            <div className="text-xs text-gray-500">
              {row.original.source}
            </div>
          )}
        </div>
      ),
    },
    {
      accessorKey: "published_at",
      header: ({ column }) => (
        <Button
          variant="ghost"
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        >
          Published
          <ArrowUpDown className="ml-2 h-4 w-4" />
        </Button>
      ),
      cell: ({ row }) => {
        const date = row.original.published_at
        return (
          <div className="text-sm">
            {date ? format(new Date(date), 'MMM d, yyyy') : 'N/A'}
          </div>
        )
      },
    },
    {
      accessorKey: "commodities",
      header: "Commodities",
      cell: ({ row }) => {
        const commodities = row.original.commodities || []
        return (
          <div className="flex flex-wrap gap-1">
            {commodities.slice(0, 2).map((commodity, i) => (
              <Badge key={i} variant="outline" className="font-normal text-xs">
                {commodity}
              </Badge>
            ))}
            {commodities.length > 2 && (
              <Badge variant="outline" className="font-normal text-xs">
                +{commodities.length - 2}
              </Badge>
            )}
          </div>
        )
      },
    },
    {
      accessorKey: "sentiment",
      header: "Sentiment",
      cell: ({ row }) => {
        const sentiment = row.original.sentiment
        const sentimentColors = {
          "Positive": "bg-green-100 text-green-800",
          "Negative": "bg-red-100 text-red-800",
          "Neutral": "bg-gray-100 text-gray-800",
        }

        return sentiment ? (
          <Badge className={cn("w-fit", sentimentColors[sentiment as keyof typeof sentimentColors] || "bg-gray-100 text-gray-800")}>
            {sentiment}
          </Badge>
        ) : (
          <span className="text-sm text-muted-foreground">N/A</span>
        )
      },
    },
    {
      accessorKey: "summary",
      header: "Summary",
      cell: ({ row }) => (
        <div className="text-sm max-w-[400px] truncate" title={row.original.summary || undefined}>
          {row.original.summary || 'N/A'}
        </div>
      ),
    },
    {
      accessorKey: "urls",
      header: "Links",
      cell: ({ row }) => <LinksPopover urls={row.original.urls || []} />,
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
        <p className="text-muted-foreground">Failed to load news: {error}</p>
      </div>
    )
  }

  return (
    <>
      <div className="w-full space-y-4 relative">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold tracking-tight">Global News — {data.length} articles</h2>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefreshNews}
              disabled={refreshing}
            >
              <RefreshCw className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")} />
              {refreshing ? 'Refreshing...' : 'Refresh News'}
            </Button>
            <ExportDropdown onExport={handleExport} />
          </div>
        </div>

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

      <Toaster />
    </>
  )
}
