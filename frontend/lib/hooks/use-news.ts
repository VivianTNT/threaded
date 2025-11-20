import { useState, useEffect } from 'react'
import { supabase } from '@/lib/supabase/client'

export interface NewsItem {
  id: string
  title: string
  urls: string[] | null
  source: string | null
  published_at: string | null
  summary: string | null
  commodities: string[] | null
  project_ids: string[] | null
  sentiment: string | null
  watchlist: boolean
  created_at: string
  updated_at: string
}

export function useNews() {
  const [news, setNews] = useState<NewsItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchNews = async () => {
    try {
      setLoading(true)
      const { data, error: fetchError } = await supabase
        .from('news')
        .select('*')
        .order('published_at', { ascending: false, nullsFirst: false })
        .order('created_at', { ascending: false })

      if (fetchError) throw fetchError

      setNews(data || [])
      setError(null)
    } catch (err: any) {
      console.error('Error fetching news:', err)
      setError(err.message || 'Failed to fetch news')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchNews()

    // Subscribe to real-time updates
    const channel = supabase
      .channel('news-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'news',
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
