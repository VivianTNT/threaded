'use client'

import { useState } from 'react'
import { AppSidebar } from '@/components/app-sidebar'
import { SiteHeader } from '@/components/site-header'
import { ChatLayout } from '@/components/chat-layout'
import { useRequireAuth } from '@/lib/auth-utils'
import { WardrobeUpload } from '@/components/wardrobe-upload'
import { ProductRecommendations, type Product } from '@/components/product-recommendations'
import { ProductDetailView } from '@/components/product-detail-view'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

import {
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar'

export default function Page() {
  const { user, isLoading } = useRequireAuth()
  const [hasWardrobe, setHasWardrobe] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const [isDetailOpen, setIsDetailOpen] = useState(false)
  const [cartItems, setCartItems] = useState<any[]>([])

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

  const handleProductClick = (product: Product) => {
    setSelectedProduct(product)
    setIsDetailOpen(true)
  }

  const handleAddToCart = (product: Product) => {
    setCartItems((prev) => [...prev, { ...product, quantity: 1 }])
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
                {!hasWardrobe ? (
                  <div className="px-4 lg:px-6 max-w-2xl mx-auto w-full">
                    <WardrobeUpload onComplete={() => setHasWardrobe(true)} />
                  </div>
                ) : (
                  <>
                    <div className="px-4 lg:px-6">
                      <div className="space-y-4">
                        <div>
                          <h1 className="text-3xl font-bold tracking-tight">
                            Recommended For You
                          </h1>
                          <p className="text-muted-foreground">
                            Curated picks based on your style profile and wardrobe
                          </p>
                        </div>

                        <Tabs defaultValue="all" className="w-full">
                          <TabsList>
                            <TabsTrigger value="all">All</TabsTrigger>
                            <TabsTrigger value="tops">Tops</TabsTrigger>
                            <TabsTrigger value="bottoms">Bottoms</TabsTrigger>
                            <TabsTrigger value="outerwear">Outerwear</TabsTrigger>
                            <TabsTrigger value="footwear">Footwear</TabsTrigger>
                            <TabsTrigger value="accessories">Accessories</TabsTrigger>
                          </TabsList>
                          <TabsContent value="all" className="mt-6">
                            <ProductRecommendations onProductClick={handleProductClick} />
                          </TabsContent>
                          <TabsContent value="tops" className="mt-6">
                            <ProductRecommendations onProductClick={handleProductClick} />
                          </TabsContent>
                          <TabsContent value="bottoms" className="mt-6">
                            <ProductRecommendations onProductClick={handleProductClick} />
                          </TabsContent>
                          <TabsContent value="outerwear" className="mt-6">
                            <ProductRecommendations onProductClick={handleProductClick} />
                          </TabsContent>
                          <TabsContent value="footwear" className="mt-6">
                            <ProductRecommendations onProductClick={handleProductClick} />
                          </TabsContent>
                          <TabsContent value="accessories" className="mt-6">
                            <ProductRecommendations onProductClick={handleProductClick} />
                          </TabsContent>
                        </Tabs>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </SidebarInset>
      </SidebarProvider>

      <ProductDetailView
        product={selectedProduct}
        isOpen={isDetailOpen}
        onClose={() => setIsDetailOpen(false)}
        onAddToCart={handleAddToCart}
      />
    </ChatLayout>
  )
} 