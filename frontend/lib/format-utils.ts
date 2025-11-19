/**
 * Utility functions for formatting numbers and values in the UI
 * Handles null/undefined values gracefully
 */

export function formatNumber(
  value: number | null | undefined,
  options?: {
    decimals?: number
    prefix?: string
    suffix?: string
    fallback?: string
    showZero?: boolean
  }
): string {
  const {
    decimals = 1,
    prefix = '',
    suffix = '',
    fallback = 'N/A',
    showZero = false
  } = options || {}

  // Handle null or undefined
  if (value === null || value === undefined) {
    return fallback
  }

  // Handle zero values
  if (value === 0 && !showZero) {
    return fallback
  }

  // Format the number
  const formatted = value.toFixed(decimals)
  return `${prefix}${formatted}${suffix}`
}

export function formatCurrency(
  value: number | null | undefined,
  options?: {
    decimals?: number
    unit?: 'M' | 'B' | 'K' | ''
    fallback?: string
    showZero?: boolean
    suffix?: string
  }
): string {
  const {
    decimals = 1,
    unit = 'M',
    fallback = 'N/A',
    showZero = false,
    suffix = ''
  } = options || {}

  if (value === null || value === undefined) {
    return fallback
  }

  if (value === 0 && !showZero) {
    return fallback
  }

  let displayValue = value
  let displayUnit = unit

  // Auto-scale large numbers
  if (unit === 'M' && Math.abs(value) >= 1000) {
    displayValue = value / 1000
    displayUnit = 'B'
  } else if (unit === '' && Math.abs(value) >= 1000000) {
    displayValue = value / 1000000
    displayUnit = 'M'
  } else if (unit === '' && Math.abs(value) >= 1000) {
    displayValue = value / 1000
    displayUnit = 'K'
  }

  return `$${displayValue.toFixed(decimals)}${displayUnit}${suffix}`
}

export function formatPercent(
  value: number | null | undefined,
  options?: {
    decimals?: number
    fallback?: string
    showZero?: boolean
  }
): string {
  const {
    decimals = 1,
    fallback = 'N/A',
    showZero = false
  } = options || {}

  if (value === null || value === undefined) {
    return fallback
  }

  if (value === 0 && !showZero) {
    return fallback
  }

  return `${value.toFixed(decimals)}%`
}

export function formatTonnes(
  value: number | null | undefined,
  options?: {
    decimals?: number
    unit?: 'Mt' | 'Kt' | 't'
    fallback?: string
    showZero?: boolean
  }
): string {
  const {
    decimals = 1,
    unit = 'Mt',
    fallback = 'N/A',
    showZero = false
  } = options || {}

  if (value === null || value === undefined) {
    return fallback
  }

  if (value === 0 && !showZero) {
    return fallback
  }

  let displayValue = value
  let displayUnit = unit

  // Convert based on unit
  if (unit === 'Mt') {
    displayValue = value / 1000000
  } else if (unit === 'Kt') {
    displayValue = value / 1000
  }

  // Auto-scale if needed
  if (displayValue < 0.1 && unit === 'Mt') {
    displayValue = value / 1000
    displayUnit = 'Kt'
  } else if (displayValue < 0.1 && unit === 'Kt') {
    displayValue = value
    displayUnit = 't'
  }

  return `${displayValue.toFixed(decimals)} ${displayUnit}`
}

export function formatGrade(
  value: number | null | undefined,
  unit: string | null | undefined,
  options?: {
    decimals?: number
    fallback?: string
    showZero?: boolean
  }
): string {
  const {
    decimals = 2,
    fallback = 'N/A',
    showZero = false
  } = options || {}

  if (value === null || value === undefined) {
    return fallback
  }

  if (value === 0 && !showZero) {
    return fallback
  }

  const gradeUnit = unit || '%'
  return `${value.toFixed(decimals)}${gradeUnit}`
}