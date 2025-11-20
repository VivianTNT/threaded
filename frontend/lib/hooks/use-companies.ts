import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase/client'

export interface Company {
  id: string
  name: string
  ticker: string | null
  exchange: string | null
  country: string | null
  website: string | null
  description: string | null
  market_cap: number | null
  urls: string[] | null
  watchlist: boolean
  created_at: string
  updated_at: string
}

export function useCompanies() {
  const [companies, setCompanies] = useState<Company[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchCompanies()

    // Subscribe to realtime changes
    const client = supabase
    if (client) {
      const subscription = client
        .channel('companies')
        .on(
          'postgres_changes',
          { event: '*', schema: 'public', table: 'companies' },
          () => {
            fetchCompanies()
          }
        )
        .subscribe()

      return () => {
        subscription.unsubscribe()
      }
    }
  }, [])

  const fetchCompanies = async () => {
    try {
      setLoading(true)

      const client = supabase
      if (!client) {
        setError('Supabase client not available')
        setLoading(false)
        return
      }

      // Fetch companies with pagination
      let allData: any[] = []
      let offset = 0
      const pageSize = 1000
      let hasMore = true

      while (hasMore) {
        const { data, error } = await client
          .from('companies')
          .select('*')
          .order('updated_at', { ascending: false })
          .range(offset, offset + pageSize - 1)

        if (error) throw error

        if (data && data.length > 0) {
          allData = [...allData, ...data]
          offset += pageSize
          hasMore = data.length === pageSize
        } else {
          hasMore = false
        }
      }

      setCompanies(allData)
      setError(null)
    } catch (err) {
      console.error('Error fetching companies:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch companies')
    } finally {
      setLoading(false)
    }
  }

  return { companies, loading, error, refetch: fetchCompanies }
}
