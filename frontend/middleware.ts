import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Public routes that don't require authentication
const publicRoutes = ['/', '/login', '/signup', '/api/auth/signup']

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl
  
  // Allow public routes
  if (publicRoutes.some(route => pathname === route || pathname.startsWith('/api/'))) {
    return NextResponse.next()
  }
  
  // For now, just continue - the actual auth check happens in components
  // This is a placeholder for future server-side auth checks
  return NextResponse.next()
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\..*|public).*)',
  ],
}
