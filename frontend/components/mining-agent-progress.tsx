'use client'

import React, { useEffect, useState } from 'react'
import { Loader2 } from 'lucide-react'

interface ProgressState {
  stage: 'idle' | 'collecting' | 'processing' | 'completed' | 'error'
  message: string
  currentStep: number
  totalSteps: number
}

export function MiningAgentProgress() {
  const [progress, setProgress] = useState<ProgressState | null>(null)
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/mining-agent/progress')
        const data = await response.json()
        
        if (data.stage && data.stage !== 'idle') {
          setProgress(data)
          setIsVisible(true)
        } else if (data.stage === 'completed' || data.stage === 'error') {
          // Hide after 3 seconds when completed
          setTimeout(() => {
            setIsVisible(false)
            setProgress(null)
          }, 3000)
        }
      } catch (error) {
        console.error('Failed to fetch progress:', error)
      }
    }, 500)

    return () => clearInterval(interval)
  }, [])

  if (!isVisible || !progress) return null

  return (
    <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-4 text-center">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <div className="space-y-2">
          <p className="text-lg font-medium">{progress.message}</p>
          {progress.totalSteps > 0 && (
            <div className="flex items-center gap-2">
              <div className="h-2 w-48 overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{
                    width: `${(progress.currentStep / progress.totalSteps) * 100}%`
                  }}
                />
              </div>
              <span className="text-sm text-muted-foreground">
                {Math.round((progress.currentStep / progress.totalSteps) * 100)}%
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
} 