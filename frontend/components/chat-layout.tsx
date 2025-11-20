"use client"

import * as React from "react"
import { MessageSquare, X, History, MessageCircle, Plus, Minimize2 } from "lucide-react"
import { useChat } from "@/lib/chat-context"
import { useGlobalChat } from "@/lib/global-chat-context"
import { ChatSidebar } from "@/components/chat-sidebar"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface ChatLayoutProps {
  children: React.ReactNode
}

export function ChatLayout({ children }: ChatLayoutProps) {
  const { chatMode, toggleChat, closeChat, isOpen, isFullscreen } = useChat()
  const { chatHistory, selectedChatId, setSelectedChatId, createNewChat, loadChat } = useGlobalChat()
  
  const handleChatSelect = (chatId: string) => {
    setSelectedChatId(chatId)
    // Load the chat
    loadChat(chatId)
  }
  
  const handleNewChat = () => {
    createNewChat()
  }

  // Listen for openChat event
  React.useEffect(() => {
    const handleOpenChat = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      if (detail?.view === 'sidebar' && !isOpen) {
        toggleChat();
      }
    };
    
    window.addEventListener('openChat', handleOpenChat);
    return () => window.removeEventListener('openChat', handleOpenChat);
  }, [isOpen, toggleChat]);

  return (
    <div className="flex h-screen w-screen overflow-hidden">
      {/* Main Content Area - Dashboard */}
      <div 
        className={cn(
          "transition-all duration-300 ease-in-out flex-grow h-full",
          isFullscreen ? "hidden" : isOpen ? "md:pr-[600px]" : ""
        )}
        style={{ 
          width: isFullscreen ? "0" : "100%",
          overflowY: "auto", // Make dashboard scrollable
          overflowX: "hidden"
        }}
      >
        {children}
      </div>

      {/* Chat Toggle Button (Fixed Position) */}
      <TooltipProvider>
        <div className={cn(
          "fixed z-50",
          isFullscreen ? "bottom-6 right-6" : "bottom-6 right-6 md:top-4 md:right-4 md:bottom-auto"
        )}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button 
                variant={isOpen ? "default" : "outline"} 
                size="icon" 
                onClick={isFullscreen ? closeChat : toggleChat}
                className="rounded-full shadow-md"
              >
                {isFullscreen ? <Minimize2 className="h-5 w-5" /> : <MessageSquare className="h-5 w-5" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="left">
              {isFullscreen ? "Minimize chat" : isOpen ? "Expand chat" : "Open chat"}
            </TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>

      {/* Fullscreen Chat Mode */}
      <div 
        className={cn(
          "fixed inset-0 z-40 transition-all duration-300 ease-in-out p-4 md:p-6",
          isFullscreen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
      >
        <div className="flex h-full rounded-xl border bg-background shadow-lg overflow-hidden">
          {/* Chat History Sidebar */}
          <div className="h-full w-72 bg-muted/30 border-r overflow-hidden hidden md:block">
            <div className="flex flex-col h-full">
              <div className="p-4 border-b flex justify-between items-center">
                <h2 className="font-medium">Chat History</h2>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <History className="h-4 w-4" />
                </Button>
              </div>
              
              <div className="p-2">
                <Button 
                  variant="outline" 
                  className="w-full justify-start gap-2 text-sm"
                  onClick={handleNewChat}
                >
                  <Plus className="h-4 w-4" />
                  New Chat
                </Button>
              </div>
              
              <div className="flex-1 overflow-auto">
                <div className="p-2 space-y-1">
                  {chatHistory.map(chat => (
                    <button 
                      key={chat.id}
                      onClick={() => handleChatSelect(chat.id)}
                      className={cn(
                        "w-full text-left p-3 rounded-lg hover:bg-muted transition-colors text-sm flex items-start gap-2 border",
                        selectedChatId === chat.id 
                          ? "border-primary bg-muted" 
                          : "border-transparent hover:border-muted-foreground/20"
                      )}
                    >
                      <MessageCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                      <div className="overflow-hidden">
                        <div className="font-medium line-clamp-1">{chat.title}</div>
                        <div className="text-xs text-muted-foreground line-clamp-1 mt-0.5">
                          {new Date(chat.updatedAt).toLocaleDateString()}
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          {/* Main Chat Area */}
          <div className="flex-1 overflow-hidden">
            <ChatSidebar variant="floating" isFullscreen={true} onClose={closeChat} />
          </div>
        </div>
      </div>

      {/* Sidebar Chat Mode */}
      <div 
        className={cn(
          "fixed right-0 top-0 z-40 h-full transition-all duration-300 ease-in-out",
          chatMode === "sidebar" ? "translate-x-0 opacity-100 w-[600px] max-w-[90vw]" : "translate-x-full opacity-0 pointer-events-none w-0"
        )}
        style={{ backgroundColor: "hsl(var(--background))" }} // Match background color
      >
        <div className="h-full flex py-2">
          <div className="flex-1 rounded-l-xl border border-r-0 shadow-lg overflow-hidden bg-background">
            <ChatSidebar variant="floating" isFullscreen={false} onClose={closeChat} />
          </div>
        </div>
      </div>
    </div>
  )
} 