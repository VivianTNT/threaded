import { createClient } from '@supabase/supabase-js'

// Service role client - bypasses RLS
export const supabaseService = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,  // Using anon key for now since service role has issues
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false
    }
  }
)

// Admin client with anon key - respects RLS
export const supabaseAdmin = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)
