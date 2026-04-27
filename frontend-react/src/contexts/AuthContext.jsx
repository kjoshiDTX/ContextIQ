import { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // undefined = still loading, null = logged out, object = logged in
  const [session, setSession] = useState(undefined);

  useEffect(() => {
    // Hydrate from existing session on mount
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session ?? null);
    });

    // Keep in sync with Supabase auth state
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const signOut = () => supabase.auth.signOut();

  return (
    <AuthContext.Provider value={{ session, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
