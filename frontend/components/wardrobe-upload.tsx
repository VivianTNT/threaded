"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, Camera, ImagePlus, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface WardrobeUploadProps {
  onComplete?: () => void
}

export function WardrobeUpload({ onComplete }: WardrobeUploadProps) {
  const [uploadedImages, setUploadedImages] = useState<string[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    const newImages: string[] = []
    Array.from(files).forEach((file) => {
      const reader = new FileReader()
      reader.onload = (event) => {
        if (event.target?.result) {
          newImages.push(event.target.result as string)
          if (newImages.length === files.length) {
            setUploadedImages((prev) => [...prev, ...newImages])
          }
        }
      }
      reader.readAsDataURL(file)
    })
  }

  const removeImage = (index: number) => {
    setUploadedImages((prev) => prev.filter((_, i) => i !== index))
  }

  const handleBuildProfile = async () => {
    setIsProcessing(true)
    // Simulate AI processing
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setIsProcessing(false)
    onComplete?.()
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Build Your Wardrobe Profile</CardTitle>
          <CardDescription>
            Upload photos of your clothing or wardrobe to help us understand your style
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={handleFileUpload}
            />

            <div className="grid grid-cols-2 gap-3">
              <Button
                variant="outline"
                className="h-32 flex-col gap-2"
                onClick={() => fileInputRef.current?.click()}
              >
                <ImagePlus className="h-8 w-8" />
                <span>Upload from Gallery</span>
              </Button>

              <Button
                variant="outline"
                className="h-32 flex-col gap-2"
                onClick={() => fileInputRef.current?.click()}
              >
                <Camera className="h-8 w-8" />
                <span>Take Photo</span>
              </Button>
            </div>

            {uploadedImages.length > 0 && (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  {uploadedImages.map((image, index) => (
                    <div key={index} className="relative aspect-square">
                      <img
                        src={image}
                        alt={`Wardrobe item ${index + 1}`}
                        className="w-full h-full object-cover rounded-md border"
                      />
                      <button
                        onClick={() => removeImage(index)}
                        className="absolute -top-2 -right-2 bg-destructive text-destructive-foreground rounded-full p-1"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>

                <div className="flex flex-col gap-3">
                  <div className="text-sm text-muted-foreground">
                    {uploadedImages.length} {uploadedImages.length === 1 ? 'photo' : 'photos'} uploaded
                  </div>
                  <Button
                    onClick={handleBuildProfile}
                    disabled={isProcessing}
                    className="w-full"
                  >
                    {isProcessing ? "Analyzing your style..." : "Build My Profile"}
                  </Button>
                </div>
              </div>
            )}

            {uploadedImages.length === 0 && (
              <div className="text-center py-8 text-sm text-muted-foreground">
                Upload at least one photo to get started
              </div>
            )}
          </div>

          <div className="border-t pt-4">
            <Button
              variant="ghost"
              className="w-full"
              onClick={onComplete}
            >
              Skip for now
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
