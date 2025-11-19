'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { SidebarTrigger } from '@/components/ui/sidebar'
import { ShoppingCartSheet } from '@/components/shopping-cart'
import { Input } from '@/components/ui/input'
import { Search } from 'lucide-react'

export function SiteHeader() {
  const [cartItems, setCartItems] = useState<any[]>([])

  const updateQuantity = (id: string, quantity: number) => {
    setCartItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, quantity } : item))
    )
  }

  const removeItem = (id: string) => {
    setCartItems((prev) => prev.filter((item) => item.id !== id))
  }

  return (
    <header className="flex h-(--header-height) shrink-0 items-center gap-2 border-b transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-(--header-height)">
      <div className="flex w-full items-center gap-1 px-4 lg:gap-2 lg:px-6">
        <SidebarTrigger className="-ml-1" />
        <Separator
          orientation="vertical"
          className="mx-2 data-[orientation=vertical]:h-4"
        />
        <h1 className="text-base font-medium">Threaded</h1>

        <div className="flex-1 flex items-center justify-center max-w-2xl mx-auto">
          <div className="relative w-full">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search for clothing, brands, styles..."
              className="pl-10 w-full"
            />
          </div>
        </div>

        <ShoppingCartSheet
          items={cartItems}
          onUpdateQuantity={updateQuantity}
          onRemoveItem={removeItem}
        />
      </div>
    </header>
  )
}
