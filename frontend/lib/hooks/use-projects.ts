import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase/client'
import { MiningProject } from '@/lib/types/mining-project'

export function useProjects() {
  const [projects, setProjects] = useState<MiningProject[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchProjects()

    // Subscribe to realtime changes if Supabase is available
    const client = supabase
    if (client) {
      const subscription = client
        .channel('projects')
        .on(
          'postgres_changes',
          { event: '*', schema: 'public', table: 'projects' },
          () => {
            fetchProjects()
          }
        )
        .subscribe()

      return () => {
        subscription.unsubscribe()
      }
    }
  }, [])

  const fetchProjects = async () => {
    try {
      setLoading(true)

      const client = supabase
      if (!client) {
        // Use dummy data if Supabase is not available
        const { dummyProjects } = await import('@/lib/data/dummy-projects')
        setProjects(dummyProjects)
        setError(null)
        setLoading(false)
        return
      }

      // Fetch projects with pagination to bypass Supabase's 1000 row limit
      let allData: any[] = []
      let offset = 0
      const pageSize = 1000
      let hasMore = true

      while (hasMore) {
        const { data, error } = await client
          .from('projects')
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

      // Fetch companies to join company names
      const { data: companies } = await client
        .from('companies')
        .select('id, name')

      const companiesMap = new Map(companies?.map(c => [c.id, c.name]) || [])

      // Transform database data to match MiningProject interface
      const transformedProjects: MiningProject[] = (allData || []).map((project: any) => ({
        // Database fields
        id: project.id,
        company_id: project.company_id,
        name: project.name,
        location: project.location,
        stage: project.stage,
        commodities: project.commodities,
        resource_estimate: project.resource_estimate,
        reserve_estimate: project.reserve_estimate,
        ownership_percentage: project.ownership_percentage,
        status: project.status,
        description: project.description,
        urls: project.urls,
        watchlist: project.watchlist || false,
        created_at: project.created_at,
        updated_at: project.updated_at,

        // Financial metrics
        npv: project.npv,
        irr: project.irr,
        capex: project.capex,

        // Computed/display fields for backward compatibility
        project: project.name,
        company: project.company_id ? (companiesMap.get(project.company_id) || 'Unknown') : 'Unknown',
        primaryCommodity: project.commodities?.[0] || 'Unknown',
        jurisdiction: project.location || 'Unknown',
        riskLevel: 'Medium' as const, // Default risk level

        // Optional fields
        project_id: project.id,
        watchlisted_at: project.watchlisted_at,
        technicalReportUrl: project.urls?.[0]
      }))

      setProjects(transformedProjects)
      setError(null)
    } catch (err) {
      console.error('Error fetching projects:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch projects')
    } finally {
      setLoading(false)
    }
  }

  return { projects, loading, error, refetch: fetchProjects }
} 