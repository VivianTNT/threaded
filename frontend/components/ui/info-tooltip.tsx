import * as React from "react"
import { Info } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

interface InfoTooltipProps {
  content: string | React.ReactNode
  children?: React.ReactNode
  side?: "top" | "right" | "bottom" | "left"
  className?: string
}

export function InfoTooltip({ 
  content, 
  children, 
  side = "top",
  className = "" 
}: InfoTooltipProps) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={`inline-flex items-center gap-1 ${className}`}>
            {children}
            <Info className="h-3 w-3 text-muted-foreground hover:text-foreground transition-colors" />
          </span>
        </TooltipTrigger>
        <TooltipContent side={side} className="max-w-xs">
          {typeof content === 'string' ? (
            <p className="text-sm">{content}</p>
          ) : (
            content
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

// Mining-specific metric explanations
export const miningMetrics = {
  npv: {
    title: "NPV (Net Present Value)",
    description: "The current value of all future cash flows from the project, discounted at a specific rate (typically 8%). Higher NPV indicates more profitable projects."
  },
  irr: {
    title: "IRR (Internal Rate of Return)",
    description: "The annual growth rate that makes NPV equal to zero. Projects with IRR > 15% are generally considered attractive in mining."
  },
  capex: {
    title: "CAPEX (Capital Expenditure)",
    description: "Initial investment required to build the mine and processing facilities. Lower CAPEX reduces project risk."
  },
  aisc: {
    title: "AISC (All-In Sustaining Cost)",
    description: "Total cost per unit of production including operating costs, sustaining capital, and G&A. Lower AISC means better margins."
  },
  payback: {
    title: "Payback Period",
    description: "Time required to recover the initial investment from cash flows. Shorter payback periods (< 3 years) are preferred."
  },
  mineLife: {
    title: "Mine Life",
    description: "Expected operational years of the mine. Longer mine life (> 10 years) provides more stable long-term returns."
  },
  stage: {
    title: "Project Stage",
    description: "Development phase: Exploration → PEA → PFS → DFS → Construction → Production. Later stages have lower risk but higher capital requirements."
  },
  grade: {
    title: "Resource Grade",
    description: "Concentration of valuable mineral in the ore. Higher grades mean more metal per tonne processed, improving economics."
  },
  strip: {
    title: "Strip Ratio",
    description: "Ratio of waste rock to ore that must be mined. Lower ratios (< 3:1) indicate more efficient mining."
  },
  recovery: {
    title: "Recovery Rate",
    description: "Percentage of metal recovered during processing. Higher recovery (> 90%) maximizes resource value."
  },
  jurisdiction: {
    title: "Jurisdiction Risk",
    description: "Political and regulatory risk of the project location. Tier 1 jurisdictions (Canada, Australia) have lowest risk."
  },
  esg: {
    title: "ESG Score",
    description: "Environmental, Social, and Governance rating. Higher scores indicate better sustainability practices and lower regulatory risk."
  },
  resources: {
    title: "Resource Category",
    description: "Confidence levels: Inferred (lowest) → Indicated → Measured (highest). Reserves are economically mineable resources."
  }
}

// Stage definitions
export const stageDefinitions = {
  "Exploration": "Early stage - identifying and defining mineral resources",
  "PEA": "Preliminary Economic Assessment - initial economic study",
  "PFS": "Pre-Feasibility Study - detailed technical and economic analysis",
  "DFS": "Definitive Feasibility Study - bankable study for financing",
  "Development": "Construction and commissioning phase",
  "Production": "Active mining and processing operations",
  "Care & Maintenance": "Temporarily suspended operations"
}
