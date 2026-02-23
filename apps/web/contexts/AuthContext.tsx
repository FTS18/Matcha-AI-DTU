"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { useRouter, usePathname } from "next/navigation";
import { createApiClient } from "@matcha/shared";

// Initialize API client
const api = createApiClient(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "http://localhost:4000");

interface User {
  id: string;
  email: string;
  name: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (token: string, userData: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  loading: true,
  login: () => { },
  logout: () => { },
});

export const useAuth = () => useContext(AuthContext);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem("auth_token");
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const userData = await api.getMe();
        setUser(userData);
      } catch (error: any) {
        // Only clear token on explicit 401 (unauthorized).
        // Network errors or timeouts should NOT log the user out.
        if (error?.message === "Unauthorized" || error?.status === 401) {
          localStorage.removeItem("auth_token");
          setUser(null);
        }
        // For other errors (network, 5xx, timeout), keep the user
        // logged in with their existing token — it's likely transient.
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []); // Only run once on mount, NOT on every route change

  const login = (token: string, userData: User) => {
    localStorage.setItem("auth_token", token);
    setUser(userData);
    // Small delay to ensure localStorage is written before redirect
    setTimeout(() => router.push("/"), 100);
  };

  const logout = () => {
    localStorage.removeItem("auth_token");
    setUser(null);
    router.push("/login");
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
