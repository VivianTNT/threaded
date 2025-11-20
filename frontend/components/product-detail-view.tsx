"use client"

import { useState } from "react"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Heart, ShoppingCart, Check, ExternalLink, Sparkles } from "lucide-react"
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

interface ProductDetailViewProps {
  product: Product | null
  isOpen: boolean
  onClose: () => void
  onAddToCart: (product: Product) => void
}

export function ProductDetailView({
  product,
  isOpen,
  onClose,
  onAddToCart,
}: ProductDetailViewProps) {
  const [selectedSize, setSelectedSize] = useState<string>("")
  const [isLiked, setIsLiked] = useState(false)
  const [addedToCart, setAddedToCart] = useState(false)

  if (!product) return null

  const sizes = ["XS", "S", "M", "L", "XL"]

  const handleAddToCart = () => {
    onAddToCart(product)
    setAddedToCart(true)
    setTimeout(() => setAddedToCart(false), 2000)
  }

  // Mock similar items from wardrobe
  const wardrobeMatches = [
    { name: "Blue Oxford Shirt", image: "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=100&h=100&fit=crop" },
    { name: "Navy Chinos", image: "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=100&h=100&fit=crop" },
  ]

  // Mock styling suggestions
  const stylingSuggestions = [
    "Pair with dark denim and white sneakers for a casual look",
    "Layer under a blazer for smart-casual occasions",
    "Works perfectly with your existing neutral color palette",
  ]

  return (
    <Sheet open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-2xl overflow-y-auto">
        <SheetHeader className="space-y-4">
          <div className="relative aspect-[4/5] -mx-6 -mt-6 overflow-hidden bg-muted">
            <img
              src={product.image}
              alt={product.name}
              className="w-full h-full object-cover"
            />
            <button
              onClick={() => setIsLiked(!isLiked)}
              className={cn(
                "absolute top-4 right-4 p-3 rounded-full bg-background/80 backdrop-blur-sm transition-colors",
                isLiked && "text-red-500"
              )}
            >
              <Heart className={cn("h-6 w-6", isLiked && "fill-current")} />
            </button>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Badge variant="secondary">{product.category}</Badge>
              {product.matchScore >= 85 && (
                <Badge className="bg-green-600">
                  {product.matchScore}% Match
                </Badge>
              )}
            </div>
            <SheetTitle className="text-2xl">{product.name}</SheetTitle>
            <SheetDescription className="text-lg">{product.brand}</SheetDescription>
            <p className="text-3xl font-bold">${product.price}</p>
          </div>
        </SheetHeader>

        <div className="space-y-6 py-6">
          {/* Size Selection */}
          <div className="space-y-3">
            <h3 className="font-semibold">Select Size</h3>
            <div className="flex gap-2">
              {sizes.map((size) => (
                <Button
                  key={size}
                  variant={selectedSize === size ? "default" : "outline"}
                  className="flex-1"
                  onClick={() => setSelectedSize(size)}
                >
                  {size}
                </Button>
              ))}
            </div>
          </div>

          <Separator />

          {/* Product Description */}
          <div className="space-y-3">
            <h3 className="font-semibold">Description</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Elevate your wardrobe with this timeless piece. Crafted from premium materials with
              attention to detail, this {product.name.toLowerCase()} combines comfort with style.
              Perfect for both casual and dressed-up occasions.
            </p>
          </div>

          <Separator />

          {/* Similar Items from Wardrobe */}
          {product.similarTo && (
            <>
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold">Matches Your Wardrobe</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Similar to: <span className="font-medium text-foreground">{product.similarTo}</span>
                </p>
                <div className="flex gap-2 overflow-x-auto pb-2">
                  {wardrobeMatches.map((item, index) => (
                    <div key={index} className="flex-shrink-0 space-y-1">
                      <div className="w-20 h-20 rounded-md overflow-hidden border">
                        <img
                          src={item.image}
                          alt={item.name}
                          className="w-full h-full object-cover"
                        />
                      </div>
                      <p className="text-xs text-muted-foreground text-center">{item.name}</p>
                    </div>
                  ))}
                </div>
              </div>
              <Separator />
            </>
          )}

          {/* AI Styling Suggestions */}
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              <h3 className="font-semibold">Threaded AI Suggestions</h3>
            </div>
            <ul className="space-y-2">
              {stylingSuggestions.map((suggestion, index) => (
                <li key={index} className="flex gap-2 text-sm">
                  <Check className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <span className="text-muted-foreground">{suggestion}</span>
                </li>
              ))}
            </ul>
          </div>

          <Separator />

          {/* Product Details */}
          <div className="space-y-3">
            <h3 className="font-semibold">Details</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>• Premium quality materials</li>
              <li>• Machine washable</li>
              <li>• Imported</li>
              <li>• Free shipping on orders over $100</li>
              <li>• Free returns within 30 days</li>
            </ul>
          </div>
        </div>

        {/* Sticky Footer Actions */}
        <div className="sticky bottom-0 -mx-6 -mb-6 bg-background border-t p-6 space-y-3">
          <Button
            className="w-full h-12 text-base"
            onClick={handleAddToCart}
            disabled={!selectedSize || addedToCart}
          >
            {addedToCart ? (
              <>
                <Check className="h-5 w-5 mr-2" />
                Added to Cart
              </>
            ) : (
              <>
                <ShoppingCart className="h-5 w-5 mr-2" />
                Add to Cart
              </>
            )}
          </Button>
          <Button variant="outline" className="w-full h-12 text-base" asChild>
            <a href={`https://${product.brand.toLowerCase()}.com`} target="_blank" rel="noopener noreferrer">
              View on {product.brand}
              <ExternalLink className="h-4 w-4 ml-2" />
            </a>
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  )
}
