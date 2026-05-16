"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";
import { createApiClient } from "@matcha/shared";
import { Loader2, ArrowRight } from "lucide-react";

const api = createApiClient(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || "http://localhost:4000");

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      // Split name into firstName and lastName
      const nameParts = name.trim().split(/\s+/);
      const firstName = nameParts[0] || "";
      const lastName = nameParts.slice(1).join(" ") || nameParts[0] || "User";

      const response = await api.register({
        email,
        password,
        firstName,
        lastName,
      });
      login(response.access_token, response.user);
    } catch (err: any) {
      let msg = err.message;
      try {
        const parsed = JSON.parse(msg);
        if (parsed.errors && Array.isArray(parsed.errors)) {
          msg = parsed.errors.map((e: any) => `${e.path.join(".")}: ${e.message}`).join(", ");
        } else if (parsed.message) {
          msg = parsed.message;
        }
      } catch (e) {}

      if (Array.isArray(msg)) msg = msg.join(", ");
      setError(msg || "Failed to register account.");
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = async () => {
    setError("");
    setLoading(true);
    try {
      const response = await api.login({ 
        email: "demo@matcha.ai", 
        password: "password123" 
      });
      login(response.access_token, response.user);
    } catch (err: any) {
      setError("Demo login currently unavailable. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center p-4 bg-background relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-primary/5 rounded-full blur-[100px] -z-10" />

      <div className="w-full max-w-sm border border-border bg-card/80 backdrop-blur-xl p-8 card-flat">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-heading font-bold text-foreground mb-2 uppercase tracking-wide">
            Create Account
          </h1>
          <p className="text-sm text-muted-foreground">Join Matcha to generate tactical match analytics.</p>
        </div>

        <form onSubmit={handleRegister} className="space-y-5">
          {error && (
            <div className="p-3 text-sm text-red-400 bg-red-400/10 border border-red-400/30 font-semibold text-center">
              {error}
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs uppercase tracking-widest font-mono text-muted-foreground mr-auto block text-left">
              Full Name
            </label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-background/50 border border-border px-4 py-3 text-sm text-foreground focus:outline-none focus:border-primary transition-colors"
              placeholder="Diego Maradona"
            />
          </div>

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
              placeholder="diego@boca.ar"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs uppercase tracking-widest font-mono text-muted-foreground text-left block">
              Password
            </label>
            <input
              type="password"
              required
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-background/50 border border-border px-4 py-3 text-sm text-foreground focus:outline-none focus:border-primary transition-colors font-mono tracking-widest"
              placeholder="••••••••"
            />
          </div>

          <div className="space-y-3">
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-primary text-[#07080F] font-semibold text-sm uppercase tracking-wider py-3.5 flex items-center justify-center gap-2 transition-all hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : "Sign Up"}
              {!loading && <ArrowRight className="w-4 h-4" />}
            </button>

            <button
              type="button"
              onClick={handleDemoLogin}
              disabled={loading}
              className="w-full bg-secondary/10 border border-secondary/30 text-secondary font-semibold text-xs uppercase tracking-widest py-3 flex items-center justify-center gap-2 transition-all hover:bg-secondary/20 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              Login as Guest (Demo)
            </button>
          </div>
        </form>

        <div className="mt-6 text-center text-xs text-muted-foreground">
          Already have an account?{" "}
          <Link href="/login" className="text-primary hover:underline font-medium">
            Sign In here
          </Link>
        </div>
      </div>
    </div>
  );
}
