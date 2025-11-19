"use client"

import { useState } from "react"
import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Heart, ShoppingCart } from "lucide-react"
import { cn } from "@/lib/utils"

interface Product {
  id: string
  name: string
  brand: string
  price: number
  image: string
  category: string
  matchScore: number
  similarTo?: string
}

interface ProductRecommendationsProps {
  onProductClick: (product: Product) => void
}

// Placeholder product data
const SAMPLE_PRODUCTS: Product[] = [
  {
    id: "1",
    name: "Classic White Oxford Shirt",
    brand: "Everlane",
    price: 68,
    image: "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&h=500&fit=crop",
    category: "Tops",
    matchScore: 95,
    similarTo: "Your blue button-down"
  },
  {
    id: "2",
    name: "Slim Fit Chinos",
    brand: "J.Crew",
    price: 89,
    image: "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=400&h=500&fit=crop",
    category: "Bottoms",
    matchScore: 92,
    similarTo: "Your khaki pants"
  },
  {
    id: "3",
    name: "Minimalist Leather Sneakers",
    brand: "Common Projects",
    price: 425,
    image: "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400&h=500&fit=crop",
    category: "Footwear",
    matchScore: 88,
  },
  {
    id: "4",
    name: "Lightweight Blazer",
    brand: "Suitsupply",
    price: 399,
    image: "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=400&h=500&fit=crop",
    category: "Outerwear",
    matchScore: 90,
    similarTo: "Your navy blazer"
  },
  {
    id: "5",
    name: "Cashmere Crewneck Sweater",
    brand: "Uniqlo",
    price: 79,
    image: "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=400&h=500&fit=crop",
    category: "Tops",
    matchScore: 87,
  },
  {
    id: "6",
    name: "Dark Wash Denim Jeans",
    brand: "Levi's",
    price: 98,
    image: "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&h=500&fit=crop",
    category: "Bottoms",
    matchScore: 93,
    similarTo: "Your light wash jeans"
  },
  {
    id: "7",
    name: "Leather Crossbody Bag",
    brand: "Madewell",
    price: 178,
    image: "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=400&h=500&fit=crop",
    category: "Accessories",
    matchScore: 85,
  },
  {
    id: "8",
    name: "Classic Trench Coat",
    brand: "Burberry",
    price: 1790,
    image: "https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=400&h=500&fit=crop",
    category: "Outerwear",
    matchScore: 91,
  },
]

export function ProductRecommendations({ onProductClick }: ProductRecommendationsProps) {
  const [likedProducts, setLikedProducts] = useState<Set<string>>(new Set())

  const toggleLike = (productId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setLikedProducts((prev) => {
      const next = new Set(prev)
      if (next.has(productId)) {
        next.delete(productId)
      } else {
        next.add(productId)
      }
      return next
    })
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {SAMPLE_PRODUCTS.map((product) => (
        <Card
          key={product.id}
          className="overflow-hidden cursor-pointer hover:shadow-lg transition-shadow"
          onClick={() => onProductClick(product)}
        >
          <div className="relative aspect-[4/5] overflow-hidden bg-muted">
            <img
              src={product.image}
              alt={product.name}
              className="w-full h-full object-cover"
            />
            <button
              onClick={(e) => toggleLike(product.id, e)}
              className={cn(
                "absolute top-2 right-2 p-2 rounded-full bg-background/80 backdrop-blur-sm transition-colors",
                likedProducts.has(product.id) && "text-red-500"
              )}
            >
              <Heart className={cn("h-5 w-5", likedProducts.has(product.id) && "fill-current")} />
            </button>
            {product.matchScore >= 90 && (
              <div className="absolute top-2 left-2">
                <Badge className="bg-green-600">
                  {product.matchScore}% Match
                </Badge>
              </div>
            )}
          </div>
          <CardContent className="p-4">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">{product.brand}</p>
              <h3 className="font-semibold line-clamp-2">{product.name}</h3>
              <p className="text-lg font-bold">${product.price}</p>
              {product.similarTo && (
                <p className="text-xs text-muted-foreground">
                  Similar to: {product.similarTo}
                </p>
              )}
            </div>
          </CardContent>
          <CardFooter className="p-4 pt-0">
            <Button
              variant="outline"
              className="w-full"
              onClick={(e) => {
                e.stopPropagation()
                onProductClick(product)
              }}
            >
              <ShoppingCart className="h-4 w-4 mr-2" />
              View Details
            </Button>
          </CardFooter>
        </Card>
      ))}
    </div>
  )
}

export type { Product }
