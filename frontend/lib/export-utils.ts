import { utils, writeFile } from 'xlsx'

export type ExportFormat = 'json' | 'csv' | 'excel'

export function exportProjects(data: any[], format: ExportFormat, filename: string = 'projects') {
  const timestamp = new Date().toISOString().slice(0, 10)

  switch (format) {
    case 'json':
      exportAsJSON(data, `${filename}_${timestamp}.json`)
      break
    case 'csv':
      exportAsCSV(data, `${filename}_${timestamp}.csv`)
      break
    case 'excel':
      exportAsExcel(data, `${filename}_${timestamp}.xlsx`)
      break
  }
}

function exportAsJSON(data: any[], filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function exportAsCSV(data: any[], filename: string) {
  if (data.length === 0) return

  // Get all unique keys from all objects
  const allKeys = new Set<string>()
  data.forEach(obj => {
    Object.keys(obj).forEach(key => allKeys.add(key))
  })
  const headers = Array.from(allKeys)

  // Create CSV content
  const csvContent = [
    headers.join(','),
    ...data.map(row =>
      headers.map(header => {
        const value = row[header]
        // Handle nested objects and arrays
        if (value === null || value === undefined) return ''
        if (typeof value === 'object') return JSON.stringify(value)
        // Escape values containing commas or quotes
        const stringValue = String(value)
        if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
          return `"${stringValue.replace(/"/g, '""')}"`
        }
        return stringValue
      }).join(',')
    )
  ].join('\n')

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function exportAsExcel(data: any[], filename: string) {
  try {
    // Create a new workbook
    const wb = utils.book_new()

    // Convert data to worksheet
    const ws = utils.json_to_sheet(data)

    // Add the worksheet to the workbook
    utils.book_append_sheet(wb, ws, 'Projects')

    // Write the file
    writeFile(wb, filename)
  } catch (error) {
    console.error('Excel export failed, falling back to CSV:', error)
    // Fallback to CSV if Excel fails
    exportAsCSV(data, filename.replace('.xlsx', '.csv'))
  }
}