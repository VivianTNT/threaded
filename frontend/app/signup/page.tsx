"use client"

import { SignupForm } from "@/components/signup-form"

export default function SignupPage() {
  return (
    <div className="flex min-h-svh flex-col items-center justify-center gap-6 bg-muted p-6 md:p-10">
      <div className="flex w-full max-w-sm flex-col gap-6">
        <a href="/" className="flex items-center gap-2 self-center">
          <div className="flex items-center gap-2">
            <img src="/favicon.avif" alt="Threaded" className="h-6 w-6 rounded" />
            <span className="text-base font-semibold">Threaded</span>
          </div>
        </a>
        <SignupForm />
      </div>
    </div>
  )
}
