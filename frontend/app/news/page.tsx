'use client'

import { AppSidebar } from '@/components/app-sidebar'
import { SiteHeader } from '@/components/site-header'
import { NewsScreenerGlobal } from '@/components/news-screener/news-screener-global'
import { ChatLayout } from '@/components/chat-layout'
import { useRequireAuth } from '@/lib/auth-utils'

import {
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar'

export default function NewsPage() {
  const { user, isLoading } = useRequireAuth()

  if (isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return null // Will redirect to login
  }

  return (
    <ChatLayout>
      <SidebarProvider
        style={
          {
            "--sidebar-width": "calc(var(--spacing) * 72)",
            "--header-height": "calc(var(--spacing) * 12)",
          } as React.CSSProperties
        }
      >
        <AppSidebar variant="inset" />
        <SidebarInset>
          <SiteHeader />
          <div className="flex flex-1 flex-col">
            <div className="@container/main flex flex-1 flex-col gap-2">
              <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
                <div className="px-4 lg:px-6">
                  <NewsScreenerGlobal />
                </div>
              </div>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>
    </ChatLayout>
  )
}