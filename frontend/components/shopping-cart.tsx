"use client"

import { useState } from "react"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ShoppingCart, Trash2, Plus, Minus } from "lucide-react"

interface CartItem {
  id: string
  name: string
  brand: string
  price: number
  image: string
  quantity: number
  size?: string
}

interface ShoppingCartProps {
  items: CartItem[]
  onUpdateQuantity: (id: string, quantity: number) => void
  onRemoveItem: (id: string) => void
}

export function ShoppingCartSheet({ items, onUpdateQuantity, onRemoveItem }: ShoppingCartProps) {
  const [isOpen, setIsOpen] = useState(false)

  const subtotal = items.reduce((sum, item) => sum + item.price * item.quantity, 0)
  const shipping = subtotal > 100 ? 0 : 10
  const tax = subtotal * 0.08
  const total = subtotal + shipping + tax

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" className="relative">
          <ShoppingCart className="h-5 w-5" />
          {items.length > 0 && (
            <Badge
              variant="destructive"
              className="absolute -top-2 -right-2 h-5 w-5 rounded-full p-0 flex items-center justify-center text-xs"
            >
              {items.length}
            </Badge>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-full sm:max-w-lg flex flex-col">
        <SheetHeader>
          <SheetTitle>Shopping Cart ({items.length})</SheetTitle>
          <SheetDescription>
            {items.length === 0
              ? "Your cart is empty"
              : "Review your items and checkout when ready"}
          </SheetDescription>
        </SheetHeader>

        {items.length === 0 ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-3">
              <ShoppingCart className="h-16 w-16 mx-auto text-muted-foreground" />
              <p className="text-muted-foreground">No items in your cart yet</p>
              <Button onClick={() => setIsOpen(false)}>Continue Shopping</Button>
            </div>
          </div>
        ) : (
          <>
            <ScrollArea className="flex-1 -mx-6 px-6">
              <div className="space-y-4 py-4">
                {items.map((item) => (
                  <div key={item.id} className="flex gap-4">
                    <div className="w-24 h-24 rounded-md overflow-hidden bg-muted flex-shrink-0">
                      <img
                        src={item.image}
                        alt={item.name}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex-1 space-y-2">
                      <div>
                        <h4 className="font-semibold line-clamp-1">{item.name}</h4>
                        <p className="text-sm text-muted-foreground">{item.brand}</p>
                        {item.size && (
                          <p className="text-xs text-muted-foreground">Size: {item.size}</p>
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Button
                            variant="outline"
                            size="icon"
                            className="h-7 w-7"
                            onClick={() => onUpdateQuantity(item.id, item.quantity - 1)}
                            disabled={item.quantity <= 1}
                          >
                            <Minus className="h-3 w-3" />
                          </Button>
                          <span className="text-sm font-medium w-8 text-center">
                            {item.quantity}
                          </span>
                          <Button
                            variant="outline"
                            size="icon"
                            className="h-7 w-7"
                            onClick={() => onUpdateQuantity(item.id, item.quantity + 1)}
                          >
                            <Plus className="h-3 w-3" />
                          </Button>
                        </div>
                        <div className="flex items-center gap-3">
                          <p className="font-semibold">${(item.price * item.quantity).toFixed(2)}</p>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 text-destructive hover:text-destructive"
                            onClick={() => onRemoveItem(item.id)}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>

            <div className="space-y-4 -mx-6 px-6 pt-4 border-t">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Subtotal</span>
                  <span className="font-medium">${subtotal.toFixed(2)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Shipping</span>
                  <span className="font-medium">
                    {shipping === 0 ? "FREE" : `$${shipping.toFixed(2)}`}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Tax</span>
                  <span className="font-medium">${tax.toFixed(2)}</span>
                </div>
                <Separator />
                <div className="flex justify-between text-base">
                  <span className="font-semibold">Total</span>
                  <span className="font-bold">${total.toFixed(2)}</span>
                </div>
              </div>

              {subtotal < 100 && (
                <p className="text-xs text-muted-foreground text-center">
                  Add ${(100 - subtotal).toFixed(2)} more for free shipping
                </p>
              )}

              <Button className="w-full h-12 text-base">
                Checkout
              </Button>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  )
}
