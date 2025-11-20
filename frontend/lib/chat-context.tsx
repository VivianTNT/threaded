"use client"

import * as React from "react"

export type ChatMode = "sidebar" | "fullscreen" | null

interface ChatContextType {
  chatMode: ChatMode
  toggleChat: () => void
  closeChat: () => void
  isOpen: boolean
  isFullscreen: boolean
}

const ChatContext = React.createContext<ChatContextType | undefined>(undefined)

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [chatMode, setChatMode] = React.useState<ChatMode>(null)

  const toggleChat = React.useCallback(() => {
    setChatMode(prev => {
      if (prev === null) return "sidebar"
      if (prev === "sidebar") return "fullscreen"
      return null
    })
  }, [])

  const closeChat = React.useCallback(() => {
    setChatMode(null)
  }, [])

  const value = React.useMemo(() => ({
    chatMode,
    toggleChat,
    closeChat,
    isOpen: chatMode !== null,
    isFullscreen: chatMode === "fullscreen"
  }), [chatMode, toggleChat, closeChat])

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = React.useContext(ChatContext)
  if (!context) {
    throw new Error("useChat must be used within a ChatProvider")
  }
  return context
} 