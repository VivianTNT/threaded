"use client";

import { createContext, useContext, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { createClient } from '@supabase/supabase-js';
import { Session, User } from '@supabase/supabase-js';

// Create Supabase client - use dummy values if env vars are missing
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || 'https://placeholder.supabase.co';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || 'placeholder-key';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

type AuthContextType = {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  signUp: (email: string, password: string, userData: any) => Promise<{ error: any; success: boolean; message: string }>;
  signIn: (email: string, password: string) => Promise<{ error: any; success: boolean; message: string }>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Get session from Supabase
    const getSession = async () => {
      try {
        const { data: { session }, error } = await supabase.auth.getSession();
        if (error) {
          console.error('Error getting session:', error);
        }
        setSession(session);
        setUser(session?.user ?? null);
      } catch (error) {
        console.error('Failed to get session:', error);
      } finally {
        setIsLoading(false);
      }
    };

    getSession();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
      setUser(session?.user ?? null);
      setIsLoading(false);
    });

    return () => {
      subscription?.unsubscribe();
    };
  }, []);

  const signUp = async (email: string, password: string, userData: any) => {
    try {
      setIsLoading(true);
      
      // Call the signup API route
      const response = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password, userData }),
      });
      
      const data = await response.json();
      
      if (!response.ok || !data.success) {
        return { 
          error: data.error || new Error(data.message || 'Signup failed'), 
          success: false, 
          message: data.message || 'Signup failed' 
        };
      }
      
      return { 
        error: null, 
        success: true, 
        message: data.message 
      };
    } catch (error: any) {
      console.error('Signup error:', error);
      return { error, success: false, message: error.message || 'An unexpected error occurred' };
    } finally {
      setIsLoading(false);
    }
  };

  const signIn = async (email: string, password: string) => {
    try {
      setIsLoading(true);
      
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        if (error.message.includes('Email not confirmed')) {
          return { error, success: false, message: 'Please verify your email address before logging in' };
        }
        return { error, success: false, message: 'Incorrect email or password' };
      }

      return { error: null, success: true, message: 'Successfully logged in' };
    } catch (error: any) {
      return { error, success: false, message: error.message };
    } finally {
      setIsLoading(false);
    }
  };

  const signOut = async () => {
    try {
      setIsLoading(true);
      const { error } = await supabase.auth.signOut();
      if (error) {
        console.error('Error signing out:', error);
        throw error;
      }
      // Clear local state immediately
      setUser(null);
      setSession(null);
      router.push('/login');
    } catch (error) {
      console.error('Sign out error:', error);
      // Even if there's an error, clear local state and redirect
      setUser(null);
      setSession(null);
      router.push('/login');
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    user,
    session,
    isLoading,
    signUp,
    signIn,
    signOut,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
