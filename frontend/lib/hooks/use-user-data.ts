"use client";

import { useEffect, useState } from 'react';
import { useAuth } from '@/lib/auth-context';
import { supabase } from '@/lib/auth-context';

export function useUserData() {
  const { user } = useAuth();
  const [userData, setUserData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!user) {
      setUserData(null);
      setIsLoading(false);
      return;
    }

    const fetchUserData = async () => {
      try {
        // Try to fetch user data, without brands join since it may not exist
        const { data, error } = await supabase
          .from('usr')
          .select('*')
          .eq('email', user.email)
          .single();

        if (error) {
          // Only log if it's not a "no rows" error (user might not be in usr table yet)
          if (error.code !== 'PGRST116') {
            console.error('Error fetching user data:', error);
          }
        } else {
          setUserData(data);
        }
      } catch (error) {
        console.error('Error fetching user data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserData();
  }, [user]);

  return { userData, isLoading };
}
