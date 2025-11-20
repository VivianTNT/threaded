import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase/client'
import { Company } from './use-companies'

export function useWatchlistCompanies() {
  const [companies, setCompanies] = useState<Company[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchCompanies()

    // Subscribe to realtime changes
    const client = supabase
    if (client) {
      const subscription = client
        .channel('watchlist-companies')
        .on(
          'postgres_changes',
          { event: '*', schema: 'public', table: 'companies', filter: 'watchlist=eq.true' },
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

      // Fetch only watchlisted companies
      const { data, error } = await client
        .from('companies')
        .select('*')
        .eq('watchlist', true)
        .order('updated_at', { ascending: false })

      if (error) throw error

      setCompanies(data || [])
      setError(null)
    } catch (err) {
      console.error('Error fetching watchlisted companies:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch watchlisted companies')
    } finally {
      setLoading(false)
    }
  }

  return { companies, loading, error, refetch: fetchCompanies }
}
