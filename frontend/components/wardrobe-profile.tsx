"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, Shirt, TrendingUp, Heart } from "lucide-react"

export function WardrobeProfile() {
  // Mock wardrobe data
  const wardrobeStats = {
    totalItems: 47,
    categories: {
      tops: 18,
      bottoms: 12,
      outerwear: 6,
      shoes: 8,
      accessories: 3,
    },
    topColors: ["Black", "Navy", "White", "Gray", "Beige"],
    styleProfile: ["Minimalist", "Classic", "Contemporary"],
  }

  const wardrobeItems = [
    { id: 1, name: "White Oxford Shirt", category: "Tops", color: "White", image: "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=200&h=250&fit=crop", wearCount: 15 },
    { id: 2, name: "Navy Blazer", category: "Outerwear", color: "Navy", image: "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=200&h=250&fit=crop", wearCount: 8 },
    { id: 3, name: "Black Chinos", category: "Bottoms", color: "Black", image: "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=200&h=250&fit=crop", wearCount: 12 },
    { id: 4, name: "Gray Sweater", category: "Tops", color: "Gray", image: "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=200&h=250&fit=crop", wearCount: 10 },
    { id: 5, name: "White Sneakers", category: "Shoes", color: "White", image: "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=200&h=250&fit=crop", wearCount: 20 },
    { id: 6, name: "Light Wash Jeans", category: "Bottoms", color: "Blue", image: "https://images.unsplash.com/photo-1542272604-787c3835535d?w=200&h=250&fit=crop", wearCount: 18 },
  ]

  return (
    <div className="space-y-6">
      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Total Items</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{wardrobeStats.totalItems}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Most Worn</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-semibold">White Sneakers</div>
            <p className="text-sm text-muted-foreground">20 times</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Top Category</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-lg font-semibold">Tops</div>
            <p className="text-sm text-muted-foreground">{wardrobeStats.categories.tops} items</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">Style Profile</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-1">
              {wardrobeStats.styleProfile.map((style) => (
                <Badge key={style} variant="secondary" className="text-xs">
                  {style}
                </Badge>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Color Analysis */}
      <Card>
        <CardHeader>
          <CardTitle>Color Palette</CardTitle>
          <CardDescription>Your most common colors</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            {wardrobeStats.topColors.map((color) => (
              <div key={color} className="flex flex-col items-center gap-2">
                <div
                  className="w-16 h-16 rounded-full border-2 border-border"
                  style={{
                    backgroundColor: color.toLowerCase() === "beige" ? "#F5F5DC" : color.toLowerCase(),
                  }}
                />
                <span className="text-sm text-muted-foreground">{color}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Wardrobe Items */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>My Wardrobe</CardTitle>
              <CardDescription>All your clothing items</CardDescription>
            </div>
            <Button>
              <Upload className="h-4 w-4 mr-2" />
              Add Items
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="all" className="w-full">
            <TabsList>
              <TabsTrigger value="all">All ({wardrobeStats.totalItems})</TabsTrigger>
              <TabsTrigger value="tops">Tops ({wardrobeStats.categories.tops})</TabsTrigger>
              <TabsTrigger value="bottoms">Bottoms ({wardrobeStats.categories.bottoms})</TabsTrigger>
              <TabsTrigger value="outerwear">Outerwear ({wardrobeStats.categories.outerwear})</TabsTrigger>
              <TabsTrigger value="shoes">Shoes ({wardrobeStats.categories.shoes})</TabsTrigger>
            </TabsList>
            <TabsContent value="all" className="mt-6">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                {wardrobeItems.map((item) => (
                  <div key={item.id} className="group relative">
                    <div className="aspect-[4/5] overflow-hidden rounded-md border bg-muted">
                      <img
                        src={item.image}
                        alt={item.name}
                        className="w-full h-full object-cover transition-transform group-hover:scale-105"
                      />
                    </div>
                    <div className="mt-2 space-y-1">
                      <h4 className="text-sm font-medium line-clamp-1">{item.name}</h4>
                      <div className="flex items-center justify-between">
                        <Badge variant="outline" className="text-xs">{item.category}</Badge>
                        <span className="text-xs text-muted-foreground">{item.wearCount}x</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>
            <TabsContent value="tops" className="mt-6">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                {wardrobeItems
                  .filter((item) => item.category === "Tops")
                  .map((item) => (
                    <div key={item.id} className="group relative">
                      <div className="aspect-[4/5] overflow-hidden rounded-md border bg-muted">
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-full h-full object-cover transition-transform group-hover:scale-105"
                        />
                      </div>
                      <div className="mt-2 space-y-1">
                        <h4 className="text-sm font-medium line-clamp-1">{item.name}</h4>
                        <span className="text-xs text-muted-foreground">{item.wearCount}x worn</span>
                      </div>
                    </div>
                  ))}
              </div>
            </TabsContent>
            {/* Other tabs would have similar content filtered by category */}
          </Tabs>
        </CardContent>
      </Card>

      {/* Insights Card */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Wardrobe Insights
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex items-start gap-3 p-3 rounded-lg bg-muted">
            <Shirt className="h-5 w-5 text-primary mt-0.5" />
            <div>
              <p className="font-medium">You have a minimalist color palette</p>
              <p className="text-sm text-muted-foreground">
                Most items are in neutral tones, making them easy to mix and match
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 rounded-lg bg-muted">
            <Heart className="h-5 w-5 text-primary mt-0.5" />
            <div>
              <p className="font-medium">Your white sneakers are your most worn item</p>
              <p className="text-sm text-muted-foreground">
                Consider finding a backup pair or exploring similar styles
              </p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 rounded-lg bg-muted">
            <TrendingUp className="h-5 w-5 text-primary mt-0.5" />
            <div>
              <p className="font-medium">Gap identified: Statement pieces</p>
              <p className="text-sm text-muted-foreground">
                Adding a few bold items could elevate your outfits for special occasions
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
