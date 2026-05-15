"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import { createApiClient } from "@matcha/shared";
import { Loader2, ArrowRight } from "lucide-react";

const api = createApiClient(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "http://localhost:4000");

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();
  const router = useRouter();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await api.login({ email, password });
      login(response.access_token, response.user);
    } catch (err: any) {
      // The API error contains stringified JSON or plain text
      let msg = err.message;
      try {
        const parsed = JSON.parse(msg);
        if (parsed.message) msg = parsed.message;
      } catch (e) {}
      setError(msg || "Failed to log in.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center p-4 bg-background relative overflow-hidden">
      {/* Background styling elements to fit the premium theme */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/5 rounded-full blur-[100px] -z-10" />

      <div className="w-full max-w-sm border border-border bg-card/80 backdrop-blur-xl p-8 card-flat">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-heading font-bold text-foreground mb-2 uppercase tracking-wide">Welcome Back</h1>
          <p className="text-sm text-muted-foreground">Sign in to analyze tactical match footage.</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-5">
          {error && (
            <div className="p-3 text-sm text-red-400 bg-red-400/10 border border-red-400/30 font-semibold text-center">
              {error}
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs uppercase tracking-widest font-mono text-muted-foreground mr-auto block text-left">
              Email
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-background/50 border border-border px-4 py-3 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
              placeholder="coach@team.com"
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs uppercase tracking-widest font-mono text-muted-foreground text-left">
                Password
              </label>
            </div>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-background/50 border border-border px-4 py-3 text-sm text-foreground focus:outline-none focus:border-primary transition-colors font-mono tracking-widest"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 bg-primary text-[#07080F] font-semibold text-sm uppercase tracking-wider py-3.5 flex items-center justify-center gap-2 transition-all hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Sign In"}
            {!loading && <ArrowRight className="w-4 h-4" />}
          </button>
        </form>

        <div className="mt-6 text-center text-xs text-muted-foreground">
          Don't have an account?{" "}
          <Link href="/register" className="text-primary hover:underline font-medium">
            Register here
          </Link>
        </div>
      </div>
    </div>
  );
}
