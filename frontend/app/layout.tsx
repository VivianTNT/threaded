import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import './globals.css'
import { ChatProvider } from '@/lib/chat-context'
import { GlobalChatProvider } from '@/lib/global-chat-context'
import { AuthProvider } from '@/lib/auth-context'
// Removed Toaster - using inline feedback instead

export const metadata: Metadata = {
  title: 'Threaded',
  description: 'AI-powered personal fashion stylist and wardrobe intelligence',
  generator: 'v0.dev',
  icons: {
    icon: [
      { url: '/favicon.avif', type: 'image/avif' },
    ],
    shortcut: '/favicon.avif',
    apple: '/favicon.avif',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/favicon.avif" type="image/avif" />
        <link rel="shortcut icon" href="/favicon.avif" type="image/avif" />
        <link rel="apple-touch-icon" href="/favicon.avif" />
        <style>{`
html {
  font-family: ${GeistSans.style.fontFamily};
  --font-sans: ${GeistSans.variable};
  --font-mono: ${GeistMono.variable};
}
        `}</style>
      </head>
      <body className={`${GeistSans.variable} ${GeistMono.variable} antialiased`} suppressHydrationWarning>
        <AuthProvider>
          <ChatProvider>
            <GlobalChatProvider>
              {children}
            </GlobalChatProvider>
          </ChatProvider>
        </AuthProvider>
      </body>
    </html>
  )
}
