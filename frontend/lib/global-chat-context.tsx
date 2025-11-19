"use client"

import React, { createContext, useContext, useState, useEffect } from "react"
import { supabase } from '@/lib/supabase/client'

// Define types
interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt?: Date;
}

interface Chat {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

interface AppContext {
  projects: any[];
  totalProjects: number;
  totalCompanies: number;
  totalFilings: number;
  totalDeals: number;
  lastUpdated: string;
}

interface GlobalChatContextType {
  // Current chat state
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
  currentChatId: string | null;
  setCurrentChatId: (id: string) => void;
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // Tool states
  isGeneratingImage: boolean;
  setIsGeneratingImage: (generating: boolean) => void;
  isSearchingWeb: boolean;
  setIsSearchingWeb: (searching: boolean) => void;
  searchResults: any[] | null;
  setSearchResults: (results: any[] | null) => void;
  uploadedFiles: any[];
  setUploadedFiles: (files: any[]) => void;

  // Chat history
  chatHistory: Chat[];
  setChatHistory: (history: Chat[]) => void;
  selectedChatId: string | null;
  setSelectedChatId: (id: string) => void;

  // Functions
  createNewChat: () => void;
  loadChat: (chatId: string) => Promise<void>;
  saveCurrentChat: () => Promise<void>;
  loadChatHistory: () => Promise<void>;

  // User info
  currentUser: any;
  setCurrentUser: (user: any) => void;

  // App context
  appContext: AppContext | null;
}

const GlobalChatContext = createContext<GlobalChatContextType | undefined>(undefined)

export function GlobalChatProvider({ children }: { children: React.ReactNode }) {
  // State for app context (needed before getInitialMessages)
  const [appContext, setAppContext] = useState<AppContext | null>(null)

  // Initialize with system message for mining
  const getInitialMessages = (): Message[] => {
    let systemContent = `You are Lithos AI, an expert mining industry assistant with real-time web search capabilities. You specialize in:

- Mining project analysis and discovery
- Commodity market trends and pricing
- Technical mining reports and feasibility studies (NI 43-101, JORC, etc.)
- Environmental and ESG considerations in mining
- Geological and resource estimation
- Mining finance and investment analysis

You can search the web for current mining news, analyze technical documents and spreadsheets, generate mining-related visualizations, and provide up-to-date industry insights. When web search is enabled, you have access to current information from mining news sites, technical report databases (SEDAR, EDGAR), commodity exchanges, and industry sources.

Always provide data-driven insights and cite sources when available. Focus on accuracy and technical precision while remaining accessible.`;

    // Add app context if available
    if (appContext) {
      systemContent += `\n\n**Current Lithos Database Context** (use when relevant to user queries):
- Total Projects: ${appContext.totalProjects}
- Mining Companies: ${appContext.totalCompanies}
- Technical Filings: ${appContext.totalFilings}
- M&A/JV Deals: ${appContext.totalDeals}
- Last Updated: ${new Date(appContext.lastUpdated).toLocaleString()}

Sample projects in database: ${appContext.projects.slice(0, 5).map(p => {
  const details = [];
  if (p.name) details.push(`Name: ${p.name}`);
  if (p.location) details.push(`Location: ${p.location}`);
  if (p.stage) details.push(`Stage: ${p.stage}`);
  if (p.commodities && p.commodities.length > 0) details.push(`Commodities: ${p.commodities.join(', ')}`);
  if (p.npv !== null && p.npv !== undefined) details.push(`NPV: $${p.npv}M`);
  if (p.irr !== null && p.irr !== undefined) details.push(`IRR: ${p.irr}%`);
  if (p.capex !== null && p.capex !== undefined) details.push(`CAPEX: $${p.capex}M`);
  return `[${details.join(' | ')}]`;
}).join('\n')}.

When users ask about specific projects or financial metrics (NPV, IRR, CAPEX), you have access to this data in the context above.

Note: This context is available for reference but should only be mentioned when directly relevant to the user's query.`;
    }

    return [
      {
        id: "1",
        role: "system",
        content: systemContent
      },
      {
        id: "2",
        role: "assistant",
        content: "Hello! I'm Lithos AI, your mining industry assistant. I can help you with:\n\n• **Project Analysis** - Analyze mining projects, feasibility studies, and technical reports\n• **Market Intelligence** - Track commodity prices, market trends, and industry news\n• **Document Analysis** - Process technical reports, spreadsheets, and geological data\n• **Web Search** - Find the latest mining news and developments\n• **Visualizations** - Generate charts, maps, and project comparisons\n\nHow can I assist you today?"
      }
    ];
  };
  
  const initialMessages = getInitialMessages();

  // State
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isGeneratingImage, setIsGeneratingImage] = useState(false)
  const [isSearchingWeb, setIsSearchingWeb] = useState(false)
  const [isDatabaseContextActive, setIsDatabaseContextActive] = useState(false)
  const [isMemoGeneratorActive, setIsMemoGeneratorActive] = useState(false)
  const [searchResults, setSearchResults] = useState<any[] | null>(null)
  const [cachedDatabaseContext, setCachedDatabaseContext] = useState<any>(null)
  const [isLoadingContext, setIsLoadingContext] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [chatHistory, setChatHistory] = useState<Chat[]>([])
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null)
  const [currentUser, setCurrentUser] = useState<any>(null)

  // Create new chat
  const createNewChat = () => {
    const newChatId = `chat_${Date.now()}`
    const newMessages = getInitialMessages()
    
    setCurrentChatId(newChatId)
    setMessages(newMessages)
    setInput("")
    setUploadedFiles([])
    setSearchResults(null)
    setIsGeneratingImage(false)
    setIsSearchingWeb(false)
    
    // Add to history
    const newChat: Chat = {
      id: newChatId,
      title: "New Mining Analysis",
      messages: newMessages,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    }
    
    setChatHistory(prev => [newChat, ...prev])
    setSelectedChatId(newChatId)
  }

  // Load a specific chat
  const loadChat = async (chatId: string) => {
    const chat = chatHistory.find(c => c.id === chatId)
    if (chat) {
      setCurrentChatId(chatId)
      setMessages(chat.messages)
      setSelectedChatId(chatId)
      setInput("")
      setUploadedFiles([])
      setSearchResults(null)
      setIsGeneratingImage(false)
      setIsSearchingWeb(false)
    }
  }

  // Save current chat
  const saveCurrentChat = async () => {
    if (!currentChatId) return
    
    // Update chat in history
    setChatHistory(prev => prev.map(chat => {
      if (chat.id === currentChatId) {
        // Generate title from first user message if needed
        let title = chat.title
        if (title === "New Mining Analysis" && messages.length > 2) {
          const firstUserMessage = messages.find(m => m.role === "user")
          if (firstUserMessage) {
            title = firstUserMessage.content.substring(0, 50) + (firstUserMessage.content.length > 50 ? "..." : "")
          }
        }
        
        return {
          ...chat,
          title,
          messages,
          updatedAt: new Date().toISOString()
        }
      }
      return chat
    }))
    
    // If Supabase is available, save to database
    if (currentUser && supabase) {
      try {
        const supabaseClient = supabase
        if (supabaseClient) {
          await supabaseClient
            .from('chat_history')
            .upsert({
              id: currentChatId,
              user_id: currentUser.id,
              title: messages.length > 2 ? messages[2]?.content?.substring(0, 50) + "..." : "New Mining Analysis",
              messages: messages,
              updated_at: new Date().toISOString()
            })
        }
      } catch (error) {
        console.error('Error saving chat:', error)
      }
    }
  }

  // Load chat history
  const loadChatHistory = async () => {
    if (!currentUser || !supabase) return
    
    try {
      const supabaseClient = supabase
      if (supabaseClient) {
        const { data, error } = await supabaseClient
          .from('chat_history')
          .select('*')
          .eq('user_id', currentUser.id)
          .order('updated_at', { ascending: false })
          .limit(50)
        
        if (data && !error) {
          setChatHistory(data.map((chat: any) => ({
            id: chat.id,
            title: chat.title,
            messages: chat.messages,
            createdAt: chat.created_at,
            updatedAt: chat.updated_at
          })))
        }
      }
    } catch (error) {
      console.error('Error loading chat history:', error)
    }
  }

  // Auto-save when messages change
  useEffect(() => {
    if (currentChatId && messages.length > 2) {
      const timer = setTimeout(() => {
        saveCurrentChat()
      }, 1000)
      return () => clearTimeout(timer)
    }
  }, [messages, currentChatId])

  // Load chat history on user change
  useEffect(() => {
    if (currentUser) {
      loadChatHistory()
    }
  }, [currentUser])

  // Initialize with a new chat
  useEffect(() => {
    if (!currentChatId) {
      createNewChat()
    }
  }, [])

  // Fetch app context
  useEffect(() => {
    const fetchAppContext = async () => {
      try {
        // Fetch projects with key fields (using correct column names from database)
        const { data: projects, count: projectCount } = await supabase
          .from('projects')
          .select('id, name, company_id, location, stage, commodities, status, npv, irr, capex, resource_estimate, reserve_estimate, ownership_percentage, description, created_at, updated_at', { count: 'exact' })
          .limit(100) // Get top 100 projects for context

        // Fetch unique companies from companies table
        const { data: companies, count: companiesCount } = await supabase
          .from('companies')
          .select('id, name', { count: 'exact' })

        const uniqueCompanies = companies?.length || companiesCount || 0

        // Mock filing and deal counts based on projects
        const filingsCount = (projectCount || 0) * 3
        const dealsCount = Math.floor((projectCount || 0) * 0.15)

        const newContext = {
          projects: projects || [],
          totalProjects: projectCount || 0,
          totalCompanies: uniqueCompanies,
          totalFilings: filingsCount,
          totalDeals: dealsCount,
          lastUpdated: new Date().toISOString()
        }

        setAppContext(newContext)

        // Update system message with context if messages exist
        setMessages(prev => {
          if (prev.length > 0 && prev[0].role === 'system') {
            const updatedMessages = [...prev]
            let systemContent = `You are Lithos AI, an expert mining industry assistant with real-time web search capabilities. You specialize in:

- Mining project analysis and discovery
- Commodity market trends and pricing
- Technical mining reports and feasibility studies (NI 43-101, JORC, etc.)
- Environmental and ESG considerations in mining
- Geological and resource estimation
- Mining finance and investment analysis

You can search the web for current mining news, analyze technical documents and spreadsheets, generate mining-related visualizations, and provide up-to-date industry insights. When web search is enabled, you have access to current information from mining news sites, technical report databases (SEDAR, EDGAR), commodity exchanges, and industry sources.

Always provide data-driven insights and cite sources when available. Focus on accuracy and technical precision while remaining accessible.

**Current Lithos Database Context** (use when relevant to user queries):
- Total Projects: ${newContext.totalProjects}
- Mining Companies: ${newContext.totalCompanies}
- Technical Filings: ${newContext.totalFilings}
- M&A/JV Deals: ${newContext.totalDeals}
- Last Updated: ${new Date(newContext.lastUpdated).toLocaleString()}

Sample projects in database: ${newContext.projects.slice(0, 5).map((p: any) => {
  const details = [];
  if (p.name) details.push(`Name: ${p.name}`);
  if (p.location) details.push(`Location: ${p.location}`);
  if (p.stage) details.push(`Stage: ${p.stage}`);
  if (p.commodities && p.commodities.length > 0) details.push(`Commodities: ${p.commodities.join(', ')}`);
  if (p.npv !== null && p.npv !== undefined) details.push(`NPV: $${p.npv}M`);
  if (p.irr !== null && p.irr !== undefined) details.push(`IRR: ${p.irr}%`);
  if (p.capex !== null && p.capex !== undefined) details.push(`CAPEX: $${p.capex}M`);
  return `[${details.join(' | ')}]`;
}).join('\n')}.

When users ask about specific projects or financial metrics (NPV, IRR, CAPEX), you have access to this data in the context above.

Note: This context is available for reference but should only be mentioned when directly relevant to the user's query.`;

            updatedMessages[0] = { ...updatedMessages[0], content: systemContent }
            return updatedMessages
          }
          return prev
        })
      } catch (error) {
        console.error('Error fetching app context:', error)
      }
    }

    fetchAppContext()
    // Refresh context every 5 minutes
    const interval = setInterval(fetchAppContext, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const value = {
    messages,
    setMessages,
    currentChatId,
    setCurrentChatId,
    input,
    setInput,
    isLoading,
    setIsLoading,
    isGeneratingImage,
    setIsGeneratingImage,
    isSearchingWeb,
    setIsSearchingWeb,
    isDatabaseContextActive,
    setIsDatabaseContextActive,
    isMemoGeneratorActive,
    setIsMemoGeneratorActive,
    cachedDatabaseContext,
    setCachedDatabaseContext,
    isLoadingContext,
    setIsLoadingContext,
    searchResults,
    setSearchResults,
    uploadedFiles,
    setUploadedFiles,
    chatHistory,
    setChatHistory,
    selectedChatId,
    setSelectedChatId,
    createNewChat,
    loadChat,
    saveCurrentChat,
    loadChatHistory,
    currentUser,
    setCurrentUser,
    appContext
  }

  return (
    <GlobalChatContext.Provider value={value}>
      {children}
    </GlobalChatContext.Provider>
  )
}

export function useGlobalChat() {
  const context = useContext(GlobalChatContext)
  if (!context) {
    throw new Error("useGlobalChat must be used within a GlobalChatProvider")
  }
  return context
} 