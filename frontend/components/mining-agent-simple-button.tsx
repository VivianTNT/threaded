'use client'

import React, { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Loader2, Play } from 'lucide-react'
import { toast } from 'sonner'
import { useProjects } from '@/lib/hooks/use-projects'
import { supabase } from '@/lib/supabase/client'

export function MiningAgentSimpleButton() {
  const [isRunning, setIsRunning] = useState(false)
  const [status, setStatus] = useState('')
  const { refetch } = useProjects()

  const runMiningAgent = async () => {
    // Check if Supabase is available
    if (!supabase) {
      toast.info('Mining agent requires database configuration')
      return
    }

    setIsRunning(true)
    setStatus('Initializing mining agent...')

    try {
      // Start the mining agent
      const response = await fetch('/api/mining-agent/start', {
        method: 'POST'
      })
      
      const data = await response.json()
      
      if (data.success) {
        // The progress is now handled by the overlay
        // Just wait for completion
        const checkInterval = setInterval(async () => {
          const progressResponse = await fetch('/api/mining-agent/progress')
          const progressData = await progressResponse.json()
          
          if (progressData.stage === 'completed' || progressData.stage === 'error') {
            clearInterval(checkInterval)
            
            if (data.statistics && data.statistics.newProjects > 0) {
              // Refresh the projects list
              refetch()
            }
          }
        }, 1000)
        
        // Timeout after 2 minutes
        setTimeout(() => {
          clearInterval(checkInterval)
          setIsRunning(false)
        }, 120000)
      } else {
        toast.error(data.error || 'Mining agent failed')
      }
    } catch (error) {
      console.error('Mining agent error:', error)
      toast.error('Failed to run mining agent')
    } finally {
      setIsRunning(false)
      setStatus('')
    }
  }

  return (
    <div className="flex items-center gap-4">
      <Button
        onClick={runMiningAgent}
        disabled={isRunning}
        size="sm"
        variant="outline"
      >
        {isRunning ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Scanning...
          </>
        ) : (
          <>
            <Play className="mr-2 h-4 w-4" />
            Run Mining Agent
          </>
        )}
      </Button>
      {isRunning && status && (
        <span className="text-sm text-muted-foreground animate-pulse">
          {status}
        </span>
      )}
    </div>
  )
} 