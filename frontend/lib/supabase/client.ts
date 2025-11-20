// Re-export from auth-context for backward compatibility
export { supabase } from '@/lib/auth-context'

import { createClient } from '@supabase/supabase-js'

// Server-side Supabase client with service role key
// Only create this on the server side
export const getSupabaseAdmin = () => {
  if (typeof window !== 'undefined') {
    throw new Error('Supabase admin client should only be used on the server')
  }
  
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY
  
  if (!url || !key) {
    throw new Error('Missing Supabase admin environment variables')
  }
  
  return createClient(url, key, {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  })
} 