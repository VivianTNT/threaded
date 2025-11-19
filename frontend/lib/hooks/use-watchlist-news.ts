import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase/client'
import { NewsItem } from './use-news'

export function useWatchlistNews() {
  const [news, setNews] = useState<NewsItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchNews = async () => {
    try {
      setLoading(true)
      const { data, error: fetchError } = await supabase
        .from('news')
        .select('*')
        .eq('watchlist', true)
        .order('published_at', { ascending: false, nullsFirst: false })
        .order('created_at', { ascending: false })

      if (fetchError) throw fetchError

      setNews(data || [])
      setError(null)
    } catch (err: any) {
      console.error('Error fetching watchlist news:', err)
      setError(err.message || 'Failed to fetch watchlist news')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchNews()

    // Subscribe to real-time updates
    const channel = supabase
      .channel('watchlist-news-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'news',
          filter: 'watchlist=eq.true',
        },
        () => {
          fetchNews()
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  return { news, loading, error, refetch: fetchNews }
}
