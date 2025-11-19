// Application configuration
export const config = {
  // Main application URL
  appUrl: process.env.NEXT_PUBLIC_APP_URL || 'https://lithosmain.vercel.app',
  
  // Authentication redirect URLs
  auth: {
    emailRedirectTo: process.env.NEXT_PUBLIC_AUTH_REDIRECT_URL || 'https://lithosmain.vercel.app/login',
    afterSignUpRedirect: '/dashboard',
    afterSignInRedirect: '/dashboard',
    afterSignOutRedirect: '/login',
  },
  
  // Supabase configuration
  supabase: {
    url: process.env.NEXT_PUBLIC_SUPABASE_URL!,
    anonKey: process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
  }
};
