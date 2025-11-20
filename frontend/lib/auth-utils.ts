"use client";

import { useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from './auth-context';

// Redirect if authenticated
export function useRedirectIfAuthenticated(redirectTo: string = '/') {
  const { user, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && user) {
      router.push(redirectTo);
    }
  }, [isLoading, user, router, redirectTo]);

  return { isLoading };
}

// Protect routes - redirect to login if not authenticated
export function useRequireAuth(redirectTo: string = '/login') {
  const { user, session, isLoading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!isLoading && !user && pathname !== redirectTo) {
      router.push(redirectTo);
    }
  }, [isLoading, user, router, redirectTo, pathname]);

  return { user, session, isLoading };
}
