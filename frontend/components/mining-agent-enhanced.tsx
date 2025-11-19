'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Loader2, Play, StopCircle, Clock, Database } from 'lucide-react'
import { useProjects } from '@/lib/hooks/use-projects'
import { supabase } from '@/lib/supabase/client'
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Badge } from "@/components/ui/badge"

interface MiningAgentEnhancedProps {
  onProgressChange?: (isRunning: boolean, message?: string) => void
}

export function MiningAgentEnhanced({ onProgressChange }: MiningAgentEnhancedProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [autoRunEnabled, setAutoRunEnabled] = useState(false)
  const [nextRunTime, setNextRunTime] = useState<Date | null>(null)
  const [countdown, setCountdown] = useState<string>("")
  const [agentStatus, setAgentStatus] = useState<{
    lastRun?: string
    totalProjects?: number
    canRunNow?: boolean
  }>({})
  const { refetch } = useProjects()

  // Load auto-run state from localStorage
  useEffect(() => {
    const savedAutoRun = localStorage.getItem('miningAgentAutoRun')
    const savedNextRun = localStorage.getItem('miningAgentNextRun')
    
    if (savedAutoRun === 'true' && savedNextRun) {
      const nextRun = new Date(savedNextRun)
      if (nextRun > new Date()) {
        setAutoRunEnabled(true)
        setNextRunTime(nextRun)
      } else {
        // Run immediately if time has passed
        setAutoRunEnabled(true)
        setTimeout(() => handleRunAgent(), 1000) // Small delay to ensure component is mounted
      }
    }
  }, [])

  // Update countdown
  useEffect(() => {
    if (!autoRunEnabled || !nextRunTime) {
      setCountdown("")
      return
    }

    const updateCountdown = () => {
      const now = new Date()
      const diff = nextRunTime.getTime() - now.getTime()
      
      if (diff <= 0) {
        handleRunAgent()
        const newNextRun = new Date(now.getTime() + 3 * 60 * 60 * 1000) // 3 hours
        setNextRunTime(newNextRun)
        localStorage.setItem('miningAgentNextRun', newNextRun.toISOString())
      } else {
        const hours = Math.floor(diff / (1000 * 60 * 60))
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
        setCountdown(`${hours}h ${minutes}m`)
      }
    }

    updateCountdown()
    const interval = setInterval(updateCountdown, 60000) // Update every minute

    return () => clearInterval(interval)
  }, [autoRunEnabled, nextRunTime])

  // Fetch agent status
  const fetchAgentStatus = async () => {
    try {
      const response = await fetch('/api/mining-agent/status')
      if (response.ok) {
        const data = await response.json()
        setAgentStatus({
          lastRun: data.lastRun,
          totalProjects: data.totalProjects,
          canRunNow: true
        })
      } else {
        console.warn('Agent status endpoint returned:', response.status)
      }
    } catch (error) {
      // Silently handle errors during initial load
      if (typeof window !== 'undefined') {
        console.warn('Could not fetch agent status:', error)
      }
    }
  }

  useEffect(() => {
    // Only fetch status if we're in the browser
    if (typeof window !== 'undefined') {
      fetchAgentStatus()
      const interval = setInterval(fetchAgentStatus, 30000)
      return () => clearInterval(interval)
    }
  }, [])

  const handleRunAgent = async () => {
    if (!supabase) {
      return
    }

    setIsRunning(true)
    onProgressChange?.(true, 'Initializing mining agent...')

    try {
      const response = await fetch('/api/mining-agent/scrape', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      if (!response.ok) {
        throw new Error('Failed to start mining agent')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (!reader) {
        throw new Error('No response body')
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.message === 'COMPLETE' && data.data) {
                const result = data.data
                // Dispatch custom event to refresh projects
                window.dispatchEvent(new CustomEvent('refreshProjects'))
                
                // Update status
                await fetchAgentStatus()
                
                // Small delay to ensure UI updates
                setTimeout(() => {
                  onProgressChange?.(false)
                }, 500)
              } else if (data.message === 'ERROR' && data.data) {
                throw new Error(data.data.error || 'Mining agent failed')
              } else {
                onProgressChange?.(true, data.message)
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error: any) {
      console.error('Mining agent error:', error)
      onProgressChange?.(false)
    } finally {
      setIsRunning(false)
    }
  }

  const handleAutoRun = () => {
    if (autoRunEnabled) {
      // Stop auto-run
      setAutoRunEnabled(false)
      setNextRunTime(null)
      localStorage.setItem('miningAgentAutoRun', 'false')
      localStorage.removeItem('miningAgentNextRun')
    } else {
      // Start auto-run
      setAutoRunEnabled(true)
      const now = new Date()
      const nextRun = new Date(now.getTime() + 3 * 60 * 60 * 1000) // 3 hours from now
      setNextRunTime(nextRun)
      localStorage.setItem('miningAgentAutoRun', 'true')
      localStorage.setItem('miningAgentNextRun', nextRun.toISOString())
      
      // Run immediately
      handleRunAgent()
    }
  }

  return (
    <div className="flex items-center gap-2">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              onClick={handleRunAgent}
              disabled={isRunning || !supabase}
              size="sm"
              className="gap-2 bg-black hover:bg-black/90 text-white border-black"
            >
              {isRunning ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Running
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Run Mining Agent
                </>
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Search for new mining projects and updates</p>
            {agentStatus.lastRun && (
              <p className="text-xs text-muted-foreground mt-1">
                Last run: {new Date(agentStatus.lastRun).toLocaleString()}
              </p>
            )}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              onClick={handleAutoRun}
              size="sm"
              variant={autoRunEnabled ? "default" : "outline"}
              className={autoRunEnabled ? "gap-2" : "gap-2"}
            >
              {autoRunEnabled ? (
                <>
                  <StopCircle className="h-4 w-4" />
                  Stop Auto-run
                </>
              ) : (
                <>
                  <Clock className="h-4 w-4" />
                  Auto-run
                </>
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>{autoRunEnabled ? 'Stop automatic updates' : 'Run automatically every 3 hours'}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>

      {autoRunEnabled && countdown && (
        <Badge variant="secondary" className="gap-1">
          <Clock className="h-3 w-3" />
          Next run in {countdown}
        </Badge>
      )}

      {agentStatus.totalProjects && (
        <Badge variant="outline" className="gap-1">
          <Database className="h-3 w-3" />
          {agentStatus.totalProjects} projects
        </Badge>
      )}
    </div>
  )
} 