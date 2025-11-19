'use client'

import React, { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { Loader2, Play, CheckCircle2, XCircle, Database } from 'lucide-react'
import { toast } from 'sonner'

interface ProgressState {
  stage: 'idle' | 'collecting' | 'processing' | 'completed' | 'error'
  message: string
  totalSteps: number
  currentStep: number
}

interface ScrapingResult {
  source: string
  documentsFound: number
  projectsCreated: number
  projectsUpdated: number
  errors: string[]
}

export function MiningAgentButton() {
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState<ProgressState>({
    stage: 'idle',
    message: 'Ready to scan for new technical reports',
    totalSteps: 0,
    currentStep: 0
  })
  const [results, setResults] = useState<ScrapingResult[]>([])

  // Poll for progress updates when running
  useEffect(() => {
    if (!isRunning || progress.stage === 'completed' || progress.stage === 'error') {
      return
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/mining-agent/progress')
        const data = await response.json()
        
        if (data.success) {
          setProgress(data.progress)
          
          if (data.progress.stage === 'completed' || data.progress.stage === 'error') {
            setIsRunning(false)
          }
        }
      } catch (error) {
        console.error('Failed to fetch progress:', error)
      }
    }, 1000) // Poll every second

    return () => clearInterval(interval)
  }, [isRunning, progress.stage])

  const startMiningAgent = async () => {
    setIsRunning(true)
    setResults([])
    setProgress({
      stage: 'collecting',
      message: 'Starting mining agent...',
      totalSteps: 0,
      currentStep: 0
    })

    try {
      const response = await fetch('/api/mining-agent/start', {
        method: 'POST'
      })
      
      const data = await response.json()
      
      if (data.success) {
        setResults(data.results)
        
        // Calculate totals
        const totals = data.results.reduce((acc: any, result: ScrapingResult) => ({
          documents: acc.documents + result.documentsFound,
          created: acc.created + result.projectsCreated,
          updated: acc.updated + result.projectsUpdated
        }), { documents: 0, created: 0, updated: 0 })
        
        toast.success(
          `Mining agent completed! Found ${totals.documents} documents, created ${totals.created} new projects, updated ${totals.updated} existing projects.`
        )
      } else {
        toast.error(data.error || 'Mining agent failed')
        setProgress({
          stage: 'error',
          message: data.error || 'Unknown error occurred',
          totalSteps: 0,
          currentStep: 0
        })
      }
    } catch (error) {
      console.error('Mining agent error:', error)
      toast.error('Failed to run mining agent')
      setProgress({
        stage: 'error',
        message: 'Failed to connect to mining agent',
        totalSteps: 0,
        currentStep: 0
      })
    } finally {
      setIsRunning(false)
    }
  }

  const getProgressPercentage = () => {
    if (progress.totalSteps === 0) return 0
    return Math.round((progress.currentStep / progress.totalSteps) * 100)
  }

  const getStageIcon = () => {
    switch (progress.stage) {
      case 'collecting':
        return <Database className="h-4 w-4 animate-pulse" />
      case 'processing':
        return <Loader2 className="h-4 w-4 animate-spin" />
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-600" />
      case 'error':
        return <XCircle className="h-4 w-4 text-red-600" />
      default:
        return <Play className="h-4 w-4" />
    }
  }

  const getStageColor = () => {
    switch (progress.stage) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'error':
        return 'bg-red-100 text-red-800'
      case 'collecting':
      case 'processing':
        return 'bg-blue-100 text-blue-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Mining Intelligence Agent</CardTitle>
        <CardDescription>
          Automatically scan and extract data from technical reports across SEDAR, EDGAR, LSE, and ASX
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <Button
            onClick={startMiningAgent}
            disabled={isRunning}
            size="lg"
            className="w-full sm:w-auto"
          >
            {isRunning ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running Mining Agent...
              </>
            ) : (
              <>
                <Play className="mr-2 h-4 w-4" />
                Run Mining Agent
              </>
            )}
          </Button>
        </div>

        {(isRunning || progress.stage !== 'idle') && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStageIcon()}
                <span className="text-sm font-medium">{progress.message}</span>
              </div>
              <Badge className={getStageColor()}>
                {progress.stage.charAt(0).toUpperCase() + progress.stage.slice(1)}
              </Badge>
            </div>
            
            {progress.totalSteps > 0 && progress.stage !== 'completed' && progress.stage !== 'error' && (
              <div className="space-y-1">
                <Progress value={getProgressPercentage()} className="h-2" />
                <p className="text-xs text-muted-foreground text-right">
                  {progress.currentStep} of {progress.totalSteps} steps ({getProgressPercentage()}%)
                </p>
              </div>
            )}
          </div>
        )}

        {results.length > 0 && (
          <div className="space-y-2 pt-4 border-t">
            <h4 className="text-sm font-semibold">Results by Source:</h4>
            {results.map((result) => (
              <div key={result.source} className="flex items-center justify-between text-sm">
                <span className="font-medium">{result.source}</span>
                <div className="flex gap-4 text-muted-foreground">
                  <span>{result.documentsFound} found</span>
                  <span className="text-green-600">+{result.projectsCreated} new</span>
                  <span className="text-blue-600">↻{result.projectsUpdated} updated</span>
                  {result.errors.length > 0 && (
                    <span className="text-red-600">{result.errors.length} errors</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
} 