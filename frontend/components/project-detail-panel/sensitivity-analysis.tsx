"use client"

import * as React from "react"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { MiningProject } from "@/lib/types/mining-project"

interface SensitivityAnalysisProps {
  project: MiningProject
}

interface SensitivityParameter {
  name: string
  key: string
  min: number
  max: number
  baseline: number
  unit: string
  step: number
}

export function SensitivityAnalysis({ project }: SensitivityAnalysisProps) {
  const parameters: SensitivityParameter[] = [
    {
      name: "Commodity Price",
      key: "price",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 1,
    },
    {
      name: "Throughput",
      key: "throughput",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 1,
    },
    {
      name: "Head Grade",
      key: "grade",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 1,
    },
    {
      name: "Operating Costs",
      key: "opex",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 1,
    },
    {
      name: "Capital Costs",
      key: "capex",
      min: 80,
      max: 120,
      baseline: 100,
      unit: "%",
      step: 1,
    },
  ]

  const [values, setValues] = React.useState<Record<string, number>>(
    parameters.reduce((acc, param) => ({ ...acc, [param.key]: param.baseline }), {})
  )

  const calculateNPV = () => {
    // Use database NPV field if available, otherwise return 0
    if (project.npv === null || project.npv === undefined) {
      return 0
    }

    // Simple sensitivity calculation
    const priceImpact = (values.price - 100) * 0.02
    const throughputImpact = (values.throughput - 100) * 0.015
    const gradeImpact = (values.grade - 100) * 0.018
    const opexImpact = -(values.opex - 100) * 0.012
    const capexImpact = -(values.capex - 100) * 0.008

    const totalImpact = 1 + priceImpact + throughputImpact + gradeImpact + opexImpact + capexImpact
    return project.npv * totalImpact
  }

  const calculateIRR = () => {
    // Use database IRR field if available, otherwise return 0
    if (project.irr === null || project.irr === undefined) {
      return 0
    }

    // Simple sensitivity calculation for IRR
    const priceImpact = (values.price - 100) * 0.015
    const throughputImpact = (values.throughput - 100) * 0.01
    const gradeImpact = (values.grade - 100) * 0.012
    const opexImpact = -(values.opex - 100) * 0.008
    const capexImpact = -(values.capex - 100) * 0.006

    const totalImpact = priceImpact + throughputImpact + gradeImpact + opexImpact + capexImpact
    return project.irr + totalImpact
  }

  const npv = calculateNPV()
  const irr = calculateIRR()
  const baselineNPV = project.npv || 0
  const baselineIRR = project.irr || 0

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold">Sensitivity Analysis</h3>
      
      <div className="space-y-6">
        {parameters.map((param) => (
          <div key={param.key} className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>{param.name}</Label>
              <span className="text-sm font-medium">
                {values[param.key]}{param.unit}
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-xs text-muted-foreground w-8">
                {param.min}{param.unit}
              </span>
              <Slider
                value={[values[param.key]]}
                onValueChange={(value) => 
                  setValues({ ...values, [param.key]: value[0] })
                }
                min={param.min}
                max={param.max}
                step={param.step}
                className="flex-1"
              />
              <span className="text-xs text-muted-foreground w-8">
                {param.max}{param.unit}
              </span>
            </div>
            <div className="flex justify-center">
              <div className="text-xs text-muted-foreground">
                Base: {param.baseline}{param.unit}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-4 pt-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">NPV (Net Present Value)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold text-blue-600">
              {baselineNPV > 0 ? `$${npv.toFixed(0)}M` : 'N/A'}
            </div>
            {baselineNPV > 0 && (
              <div className="text-xs text-muted-foreground">
                {npv > baselineNPV ? "+" : ""}
                {((npv - baselineNPV) / baselineNPV * 100).toFixed(1)}%
                from baseline
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">IRR (Internal Rate of Return)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold text-green-600">
              {baselineIRR > 0 ? `${irr.toFixed(1)}%` : 'N/A'}
            </div>
            {baselineIRR > 0 && (
              <div className="text-xs text-muted-foreground">
                {irr > baselineIRR ? "+" : ""}
                {(irr - baselineIRR).toFixed(1)}% pts
                from baseline
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="flex justify-center">
        <button
          className="text-sm text-blue-600 hover:underline"
          onClick={() => setValues(
            parameters.reduce((acc, param) => ({ ...acc, [param.key]: param.baseline }), {})
          )}
        >
          Reset to baseline
        </button>
      </div>
    </div>
  )
} 