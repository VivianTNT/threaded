"use client"

import * as React from "react"
import { X, ArrowLeft } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { MiningProject } from "@/lib/types/mining-project"
import { SingleProjectView } from "./single-project-view-compact"
import { ProjectComparisonView } from "./project-comparison-view"

interface ProjectDetailPanelProps {
  isOpen: boolean
  onClose: () => void
  projects: MiningProject[]
  mode: "single" | "comparison"
  onProjectSelect?: (projectId: string) => void
  onUpdate?: () => void
}

export function ProjectDetailPanel({
  isOpen,
  onClose,
  projects,
  mode,
  onProjectSelect,
}: ProjectDetailPanelProps) {
  // Close on ESC key
  React.useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose()
      }
    }
    
    document.addEventListener("keydown", handleEscape)
    return () => document.removeEventListener("keydown", handleEscape)
  }, [isOpen, onClose])

  // Prevent body scroll when panel is open
  React.useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden"
    } else {
      document.body.style.overflow = "unset"
    }
    
    return () => {
      document.body.style.overflow = "unset"
    }
  }, [isOpen])

  if (!projects.length) return null

  return (
    <>
      {/* Backdrop */}
      <div
        className={cn(
          "fixed inset-0 bg-black/20 backdrop-blur-sm transition-opacity z-40",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />
      
      {/* Panel */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full bg-background shadow-xl transition-transform duration-300 ease-in-out z-50",
          "w-[60%] min-w-[600px] max-w-[1200px]",
          isOpen ? "translate-x-0" : "translate-x-full"
        )}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 bg-background border-b px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Screener
              </Button>
              <h2 className="text-lg font-semibold">
                {mode === "single" 
                  ? projects[0]?.project 
                  : `Project Comparison (${projects.length} projects selected)`}
              </h2>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-8 w-8"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Content */}
        <div className="h-[calc(100%-73px)] overflow-y-auto">
          {mode === "single" ? (
            <SingleProjectView
              project={projects[0]}
              onProjectSelect={onProjectSelect}
              onClose={onClose}
            />
          ) : (
            <ProjectComparisonView projects={projects} />
          )}
        </div>
      </div>
    </>
  )
} 