"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Loader2, Search, TrendingUp, CheckCircle2, AlertCircle } from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface MiningAgentV2ButtonProps {
  onComplete?: () => void
  onProgressChange?: (isRunning: boolean, message?: string) => void
  className?: string
}

export function MiningAgentV2Button({ onComplete, onProgressChange, className }: MiningAgentV2ButtonProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [showProgress, setShowProgress] = useState(false)
  const [progressMessages, setProgressMessages] = useState<Array<{
    message: string
    timestamp: number
    type?: 'info' | 'success' | 'error' | 'warning'
  }>>([])
  const [summary, setSummary] = useState<{
    inserted: number
    updated: number
    errors?: number
    duration?: number
  } | null>(null)
  const [progress, setProgress] = useState(0)

  const runMiningAgent = async () => {
    setIsRunning(true)
    setShowProgress(false) // Don't show modal, use inline progress
    setProgressMessages([])
    setSummary(null)
    setProgress(0)
    
    if (onProgressChange) {
      onProgressChange(true, "Initializing mining agent...")
    }
    
    try {
      const response = await fetch('/api/mining-agent/scrape-v2', {
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
        throw new Error('No response stream')
      }
      
      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        
        // Process complete messages
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.message === 'COMPLETE') {
                setSummary(data.data || {})
                setProgress(100)
                
                // Add completion message
                setProgressMessages(prev => [...prev, {
                  message: `✅ Mining Agent Complete! Found ${data.data?.inserted || 0} new projects, updated ${data.data?.updated || 0}`,
                  timestamp: Date.now(),
                  type: 'success'
                }])
                
                if (onComplete) {
                  onComplete()
                }
              } else if (data.message === 'ERROR') {
                setProgressMessages(prev => [...prev, {
                  message: `❌ Error: ${data.data?.error || 'Unknown error'}`,
                  timestamp: Date.now(),
                  type: 'error'
                }])
              } else if (data.message) {
                // Update parent component's progress
                if (onProgressChange) {
                  onProgressChange(true, data.message)
                }
                
                // Determine message type
                let type: 'info' | 'success' | 'error' | 'warning' = 'info'
                
                if (data.message.includes('Added') || data.message.includes('Complete')) {
                  type = 'success'
                } else if (data.message.includes('Error') || data.message.includes('Failed')) {
                  type = 'error'
                } else if (data.message.includes('Warning')) {
                  type = 'warning'
                }
                
                setProgressMessages(prev => [...prev, {
                  message: data.message,
                  timestamp: data.timestamp || Date.now(),
                  type
                }])
                
                // Update progress based on message content
                if (data.message.includes('Starting')) setProgress(5)
                else if (data.message.includes('Generating')) setProgress(10)
                else if (data.message.includes('Searching')) setProgress(prev => Math.min(prev + 5, 60))
                else if (data.message.includes('Found')) setProgress(prev => Math.min(prev + 3, 70))
                else if (data.message.includes('Processing')) setProgress(75)
                else if (data.message.includes('Extracting')) setProgress(80)
                else if (data.message.includes('Saving')) setProgress(90)
                else if (data.message.includes('Complete')) setProgress(100)
              }
            } catch (error) {
              console.error('Error parsing SSE data:', error)
            }
          }
        }
      }
    } catch (error) {
      console.error('Mining agent error:', error)
      setProgressMessages(prev => [...prev, {
        message: `❌ Failed to run mining agent: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: Date.now(),
        type: 'error'
      }])
    } finally {
      setIsRunning(false)
      if (onProgressChange) {
        onProgressChange(false, "")
      }
    }
  }
  
  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    const scrollArea = document.getElementById('progress-scroll-area')
    if (scrollArea) {
      scrollArea.scrollTop = scrollArea.scrollHeight
    }
  }, [progressMessages])
  
  return (
    <>
      <Button
        onClick={runMiningAgent}
        disabled={isRunning}
        size="sm"
        className={cn("gap-2", className)}
      >
        {isRunning ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Running Mining Agent...
          </>
        ) : (
          <>
            <Search className="h-4 w-4" />
            Run Mining Agent
          </>
        )}
      </Button>
      
      <Dialog open={showProgress} onOpenChange={setShowProgress}>
        <DialogContent className="max-w-3xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Mining Discovery Agent
            </DialogTitle>
            <DialogDescription>
              Searching regulatory filings and technical reports for new mining projects
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            {/* Progress bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>Progress</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
            
            {/* Progress messages */}
            <ScrollArea 
              id="progress-scroll-area"
              className="h-[400px] w-full rounded-md border p-4"
            >
              <div className="space-y-2">
                {progressMessages.map((msg, index) => (
                  <div
                    key={index}
                    className={cn(
                      "text-sm font-mono",
                      msg.type === 'success' && "text-green-600 dark:text-green-400",
                      msg.type === 'error' && "text-red-600 dark:text-red-400",
                      msg.type === 'warning' && "text-yellow-600 dark:text-yellow-400",
                      msg.type === 'info' && "text-muted-foreground"
                    )}
                  >
                    <span className="text-xs text-muted-foreground mr-2">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </span>
                    {msg.message}
                  </div>
                ))}
              </div>
            </ScrollArea>
            
            {/* Summary */}
            {summary && (
              <div className="rounded-lg border bg-muted/50 p-4">
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  Discovery Complete
                </h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">New Projects:</span>
                    <Badge variant="default" className="ml-2">{summary.inserted}</Badge>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Updated:</span>
                    <Badge variant="secondary" className="ml-2">{summary.updated}</Badge>
                  </div>
                  {summary.duration && (
                    <div>
                      <span className="text-muted-foreground">Duration:</span>
                      <span className="ml-2">{(summary.duration / 1000).toFixed(1)}s</span>
                    </div>
                  )}
                </div>
                {summary.errors && summary.errors > 0 && (
                  <div className="mt-2 text-sm text-yellow-600">
                    <AlertCircle className="h-4 w-4 inline mr-1" />
                    {summary.errors} errors occurred during processing
                  </div>
                )}
              </div>
            )}
          </div>
          
          {/* Footer actions */}
          <div className="flex justify-end gap-2 mt-4">
            {!isRunning && (
              <Button
                variant="outline"
                onClick={() => setShowProgress(false)}
              >
                Close
              </Button>
            )}
            {summary && onComplete && (
              <Button
                onClick={() => {
                  setShowProgress(false)
                  onComplete()
                }}
              >
                View Projects
              </Button>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
