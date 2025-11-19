'use client'

import React from 'react'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
} from '@/components/ui/context-menu'
import { useGlobalChat } from '@/lib/global-chat-context'
import { useChat } from '@/lib/chat-context'
import { MessageSquare, TrendingUp, Search, Lightbulb, ChartBar, Database, AlertTriangle, Target, FileText } from 'lucide-react'
import { formatProjectForChat, formatNewsForChat } from '@/lib/chat/project-context-generator'

interface ContextMenuChatProps {
  children: React.ReactNode
  data: any
  dataType: 'project' | 'metric' | 'chart' | 'cell' | 'news' | 'company'
  context?: string
}

export function ContextMenuChat({ children, data, dataType, context }: ContextMenuChatProps) {
  const { toggleChat } = useChat()
  const { setInput } = useGlobalChat()

  const generateQuery = (action: string) => {
    let query = ''

    switch (dataType) {
      case 'project':
        if (action === 'full-analysis') {
          // Use the comprehensive project summary with risk analysis
          query = formatProjectForChat(data)
        } else {
          const project = data
          query = action === 'explain'
            ? `Explain the ${project.project || project.name} project in ${project.jurisdiction}. Why is the NPV $${project.postTaxNPV}M and IRR ${project.irr}%?`
            : action === 'compare'
            ? `Compare ${project.project || project.name} with similar ${project.primaryCommodity} projects in the same region`
            : action === 'analyze'
            ? `Analyze the investment potential of ${project.project || project.name} considering its ${project.stage} stage and ${project.irr}% IRR`
            : `What are the key risks for ${project.project || project.name} given its ${project.riskLevel} risk rating?`
        }
        break

      case 'news':
        if (action === 'full-analysis') {
          // Use the comprehensive news analysis with predictive insights
          query = formatNewsForChat(data)
        } else {
          query = action === 'explain'
            ? `Explain the implications of this news: "${data.headline}"`
            : action === 'predict'
            ? `What are the predictive insights and investment opportunities from this news: "${data.headline}"?`
            : action === 'risks'
            ? `What risks and market impacts does this news present: "${data.headline}"?`
            : query
        }
        break

      case 'company':
        const company = data
        query = action === 'explain'
          ? `Provide an overview of ${company.company_name} including their mining projects and financial performance`
          : action === 'compare'
          ? `Compare ${company.company_name} with peer companies in the ${company.primary_commodity || 'mining'} sector`
          : action === 'analyze'
          ? `Analyze the investment potential of ${company.company_name} based on their project portfolio`
          : `What are the key risks and opportunities for ${company.company_name}?`
        break

      case 'metric':
        // Extract the actual value from the data object
        const metricValue = typeof data === 'object' ?
          (data.totalProjects || data.totalCompanies || data.totalFilings || data.totalDeals || JSON.stringify(data)) :
          data
        query = action === 'explain'
          ? `Explain why the ${context || 'metric'} is ${metricValue}. What factors influence this value?`
          : action === 'benchmark'
          ? `How does ${metricValue} compare to industry benchmarks for ${context || 'this metric'}?`
          : `Show me trends and patterns for ${context || 'this metric'} across the mining industry`
        break
        
      case 'chart':
        query = action === 'explain'
          ? `Explain the trends shown in this chart data: ${JSON.stringify(data).slice(0, 200)}...`
          : action === 'forecast'
          ? `Based on this historical data, what are the projections for the next 6 months?`
          : `What insights can you derive from this data pattern?`
        break
        
      case 'cell':
        query = action === 'explain'
          ? `Explain what "${data}" means in the context of ${context || 'mining projects'}`
          : action === 'similar'
          ? `Show me similar values or projects with "${data}" characteristics`
          : `How is "${data}" calculated or determined in mining analysis?`
        break
    }
    
    return query
  }

  const handleAction = (action: string) => {
    const query = generateQuery(action)
    setInput(query)
    toggleChat()  // Open the chat panel
  }

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        {children}
      </ContextMenuTrigger>
      <ContextMenuContent className="w-64">
        {(dataType === 'project' || dataType === 'news') && (
          <ContextMenuItem onClick={() => handleAction('full-analysis')} className="font-medium">
            <FileText className="mr-2 h-4 w-4" />
            <span>Add Full Analysis to Chat</span>
          </ContextMenuItem>
        )}
        
        <ContextMenuItem onClick={() => handleAction('explain')}>
          <MessageSquare className="mr-2 h-4 w-4" />
          <span>Ask AI to explain this</span>
        </ContextMenuItem>
        
        {dataType === 'project' && (
          <>
            <ContextMenuItem onClick={() => handleAction('compare')}>
              <ChartBar className="mr-2 h-4 w-4" />
              <span>Compare with similar projects</span>
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('analyze')}>
              <TrendingUp className="mr-2 h-4 w-4" />
              <span>Analyze investment potential</span>
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('risks')}>
              <AlertTriangle className="mr-2 h-4 w-4" />
              <span>Identify key risks</span>
            </ContextMenuItem>
          </>
        )}
        
        {dataType === 'news' && (
          <>
            <ContextMenuItem onClick={() => handleAction('predict')}>
              <Target className="mr-2 h-4 w-4" />
              <span>Predictive Analytics</span>
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('risks')}>
              <AlertTriangle className="mr-2 h-4 w-4" />
              <span>Risk Assessment</span>
            </ContextMenuItem>
          </>
        )}
        
        {dataType === 'company' && (
          <>
            <ContextMenuItem onClick={() => handleAction('compare')}>
              <ChartBar className="mr-2 h-4 w-4" />
              <span>Compare with peers</span>
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('analyze')}>
              <TrendingUp className="mr-2 h-4 w-4" />
              <span>Portfolio analysis</span>
            </ContextMenuItem>
          </>
        )}
        
        {dataType === 'metric' && (
          <>
            <ContextMenuItem onClick={() => handleAction('benchmark')}>
              <Database className="mr-2 h-4 w-4" />
              <span>Compare to benchmarks</span>
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('trends')}>
              <TrendingUp className="mr-2 h-4 w-4" />
              <span>Show industry trends</span>
            </ContextMenuItem>
          </>
        )}
        
        {dataType === 'chart' && (
          <>
            <ContextMenuItem onClick={() => handleAction('forecast')}>
              <TrendingUp className="mr-2 h-4 w-4" />
              <span>Forecast future trends</span>
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('insights')}>
              <Lightbulb className="mr-2 h-4 w-4" />
              <span>Extract insights</span>
            </ContextMenuItem>
          </>
        )}
        
        <ContextMenuSeparator />
        
        <ContextMenuSub>
          <ContextMenuSubTrigger>
            <Search className="mr-2 h-4 w-4" />
            <span>Search for...</span>
          </ContextMenuSubTrigger>
          <ContextMenuSubContent>
            <ContextMenuItem onClick={() => handleAction('similar')}>
              Similar projects
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('documentation')}>
              Related reports
            </ContextMenuItem>
            <ContextMenuItem onClick={() => handleAction('news')}>
              Latest news
            </ContextMenuItem>
          </ContextMenuSubContent>
        </ContextMenuSub>
      </ContextMenuContent>
    </ContextMenu>
  )
}
