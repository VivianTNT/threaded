"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useAuth } from "@/lib/auth-context"
import { useRedirectIfAuthenticated } from "@/lib/auth-utils"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { CheckCircle, AlertCircle, Info } from "lucide-react"

export function SignupForm({
  className,
  ...props
}: React.ComponentPropsWithoutRef<"div">) {
  const [step, setStep] = useState<"account" | "style">("account")
  const router = useRouter()
  const { signUp } = useAuth()
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [alert, setAlert] = useState<{
    type: "success" | "error" | "info" | null;
    message: string;
  }>({ type: null, message: "" })
  
  // Form data state
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    brand: "", // This will store style preferences
    desc: "", // This will store fashion interests
    category: "", // This will store fashion category
    website: "",
    founded: ""
  })

  const [stylePreferences, setStylePreferences] = useState<string[]>([])
  
  // Redirect if already authenticated
  useRedirectIfAuthenticated()

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { id, value } = e.target
    setFormData(prev => ({ ...prev, [id]: value }))
  }
  
  const handleSelectChange = (value: string) => {
    setFormData(prev => ({ ...prev, category: value }))
  }

  const handleNextStep = (e: React.FormEvent) => {
    e.preventDefault()
    setAlert({ type: null, message: "" })
    
    // Validate account information
    if (!formData.name || !formData.email || !formData.password) {
      setAlert({
        type: "error",
        message: "Please fill in all required fields"
      })
      return
    }
    
    // Password validation
    if (formData.password.length < 8) {
      setAlert({
        type: "error",
        message: "Password must be at least 8 characters long"
      })
      return
    }
    
    setStep("style")
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setAlert({ type: null, message: "" })
    
    // Validate style preferences
    if (stylePreferences.length === 0 || !formData.category) {
      setAlert({
        type: "error",
        message: "Please select at least one style preference"
      })
      return
    }
    
    setIsSubmitting(true)
    
    try {
      const { error, success, message } = await signUp(
        formData.email,
        formData.password,
        {
          name: formData.name,
          brand: stylePreferences.join(", "), // Style preferences stored as brand
          desc: formData.desc || "Fashion enthusiast",
          category: formData.category,
          website: formData.website || "",
          founded: formData.founded || ""
        }
      )
      
      if (success) {
        setAlert({
          type: "success",
          message: message || "A confirmation link has been sent to your email"
        })
        setTimeout(() => {
          router.push("/login")
        }, 3000)
      } else {
        if (message === 'Email already taken') {
          setAlert({
            type: "error",
            message: "Email already taken"
          })
        } else if (message?.includes('verification')) {
          setAlert({
            type: "info",
            message: "Please verify your email address before logging in"
          })
        } else {
          setAlert({
            type: "error",
            message: message || "Failed to create account"
          })
        }
      }
    } catch (error) {
      setAlert({
        type: "error",
        message: "An unexpected error occurred"
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className={cn("flex flex-col gap-6", className)} {...props}>
      <Card>
        <CardHeader className="text-center">
          <CardTitle className="text-xl">Welcome to Threaded</CardTitle>
          <CardDescription>
            Your AI-powered personal fashion stylist
          </CardDescription>
        </CardHeader>
        <CardContent>
          {alert.type && (
            <Alert 
              variant={alert.type === "success" ? "default" : alert.type === "error" ? "destructive" : "default"}
              className="mb-6"
            >
              {alert.type === "success" && <CheckCircle className="h-4 w-4" />}
              {alert.type === "error" && <AlertCircle className="h-4 w-4" />}
              {alert.type === "info" && <Info className="h-4 w-4" />}
              <AlertDescription>{alert.message}</AlertDescription>
            </Alert>
          )}
          <Tabs value={step} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="account" disabled={step === "style"}>Account</TabsTrigger>
              <TabsTrigger value="style" disabled={step === "account"}>Style Profile</TabsTrigger>
            </TabsList>
            <TabsContent value="account">
              <form onSubmit={handleNextStep}>
                <div className="grid gap-6">
                  <div className="flex flex-col gap-4">
                    <Button type="button" variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="h-5 w-5 mr-2">
                        <path
                          d="M12.152 6.896c-.948 0-2.415-1.078-3.96-1.04-2.04.027-3.91 1.183-4.961 3.014-2.117 3.675-.546 9.103 1.519 12.09 1.013 1.454 2.208 3.09 3.792 3.039 1.52-.065 2.09-.987 3.935-.987 1.831 0 2.35.987 3.96.948 1.637-.026 2.676-1.48 3.676-2.948 1.156-1.688 1.636-3.325 1.662-3.415-.039-.013-3.182-1.221-3.22-4.857-.026-3.04 2.48-4.494 2.597-4.559-1.429-2.09-3.623-2.324-4.39-2.376-2-.156-3.675 1.09-4.61 1.09zM15.53 3.83c.843-1.012 1.4-2.427 1.245-3.83-1.207.052-2.662.805-3.532 1.818-.78.896-1.454 2.338-1.273 3.714 1.338.104 2.715-.688 3.559-1.701"
                          fill="currentColor"
                        />
                      </svg>
                      Continue with Apple
                    </Button>
                    <Button type="button" variant="outline" className="w-full">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="h-5 w-5 mr-2">
                        <path
                          d="M12.48 10.92v3.28h7.84c-.24 1.84-.853 3.187-1.787 4.133-1.147 1.147-2.933 2.4-6.053 2.4-4.827 0-8.6-3.893-8.6-8.72s3.773-8.72 8.6-8.72c2.6 0 4.507 1.027 5.907 2.347l2.307-2.307C18.747 1.44 16.133 0 12.48 0 5.867 0 .307 5.387.307 12s5.56 12 12.173 12c3.573 0 6.267-1.173 8.373-3.36 2.16-2.16 2.84-5.213 2.84-7.667 0-.76-.053-1.467-.173-2.053H12.48z"
                          fill="currentColor"
                        />
                      </svg>
                      Continue with Google
                    </Button>
                  </div>
                  <div className="relative text-center text-sm after:absolute after:inset-0 after:top-1/2 after:z-0 after:flex after:items-center after:border-t after:border-border">
                    <span className="relative z-10 bg-background px-2 text-muted-foreground">
                      Or continue with
                    </span>
                  </div>
                  <div className="grid gap-6">
                    <div className="grid gap-2">
                      <Label htmlFor="name">Full Name</Label>
                      <Input
                        id="name"
                        placeholder="John Smith"
                        required
                        value={formData.name}
                        onChange={handleChange}
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="email">Email</Label>
                      <Input
                        id="email"
                        type="email"
                        placeholder="john@yourcompany.com"
                        required
                        value={formData.email}
                        onChange={handleChange}
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="password">Password</Label>
                      <Input
                        id="password"
                        type="password"
                        required
                        value={formData.password}
                        onChange={handleChange}
                      />
                      <p className="text-xs text-muted-foreground">
                        Password must be at least 8 characters long
                      </p>
                    </div>
                    <Button type="submit" className="w-full">
                      Continue
                    </Button>
                  </div>
                </div>
              </form>
            </TabsContent>
            <TabsContent value="style">
              <form onSubmit={handleSubmit}>
                <div className="grid gap-6">
                  <div className="grid gap-3">
                    <Label>What's your style?</Label>
                    <p className="text-sm text-muted-foreground">Select all that apply</p>
                    <div className="grid grid-cols-2 gap-2">
                      {["Minimalist", "Streetwear", "Classic", "Bohemian", "Athleisure", "Formal", "Vintage", "Contemporary"].map((style) => (
                        <Button
                          key={style}
                          type="button"
                          variant={stylePreferences.includes(style) ? "default" : "outline"}
                          className="justify-start"
                          onClick={() => {
                            setStylePreferences(prev =>
                              prev.includes(style)
                                ? prev.filter(s => s !== style)
                                : [...prev, style]
                            )
                          }}
                        >
                          {style}
                        </Button>
                      ))}
                    </div>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="category">Gender / Clothing Type</Label>
                    <Select required value={formData.category} onValueChange={handleSelectChange}>
                      <SelectTrigger id="category">
                        <SelectValue placeholder="Select preference" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="mens">Men's Fashion</SelectItem>
                        <SelectItem value="womens">Women's Fashion</SelectItem>
                        <SelectItem value="unisex">Unisex / Gender Neutral</SelectItem>
                        <SelectItem value="all">All Fashion</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="desc">Tell us about your fashion goals (optional)</Label>
                    <Textarea
                      id="desc"
                      placeholder="I want to build a versatile capsule wardrobe, find sustainable brands, etc..."
                      className="min-h-24"
                      value={formData.desc}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setStep("account")}
                    >
                      Back
                    </Button>
                    <Button type="submit" disabled={isSubmitting}>
                      {isSubmitting ? "Creating Account..." : "Complete Setup"}
                    </Button>
                  </div>
                </div>
              </form>
            </TabsContent>
          </Tabs>
        </CardContent>
        <CardFooter className="flex justify-center border-t px-6 py-4">
          <div className="text-center text-sm">
            Already have an account?{" "}
            <Link href="/login" className="underline underline-offset-4">
              Log in
            </Link>
          </div>
        </CardFooter>
      </Card>
      <div className="text-balance text-center text-xs text-muted-foreground [&_a]:underline [&_a]:underline-offset-4 [&_a]:hover:text-primary">
        By clicking continue, you agree to our <a href="#">Terms of Service</a>{" "}
        and <a href="#">Privacy Policy</a>.
      </div>
    </div>
  )
}
