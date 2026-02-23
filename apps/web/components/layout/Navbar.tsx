"use client";

import Link from "next/link";
import Image from "next/image";
import { useAuth } from "@/contexts/AuthContext";
import { LogOut, User } from "lucide-react";

export function Navbar() {
  const { user, logout } = useAuth();

  return (
    <nav className="w-full border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-50 shadow-2xl">
      {/* Content Layer */}
      <div className="relative z-10 flex items-center justify-between py-3 md:py-4 px-4 sm:px-6 md:px-8 max-w-[1440px] mx-auto">
        <Link 
          href="/" 
          className="flex items-center gap-2 sm:gap-3 transition-opacity duration-200 hover:opacity-80 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-sm group" 
          aria-label="Go to homepage"
        >
          {/* Custom Logo Image */}
          <div className="relative size-7 sm:size-8 shrink-0 overflow-hidden transform group-hover:scale-105 transition-transform duration-300">
            <Image 
              src="/favicons/logo.png" 
              alt="Matcha AI Logo" 
              fill
              className="object-contain drop-shadow-[0_0_8px_rgba(var(--color-primary),0.5)]"
              sizes="(max-width: 640px) 28px, 32px"
            />
          </div>
          
          <div className="flex items-baseline gap-1">
            <span className="font-display tracking-[0.12em] text-[16px] sm:text-[18px] md:text-[20px] text-foreground drop-shadow-md">MATCHA</span>
            <span className="font-display tracking-[0.12em] ml-0.5 text-[16px] sm:text-[18px] md:text-[20px] text-primary drop-shadow-[0_0_8px_rgba(var(--color-primary),0.5)]">AI</span>
          </div>
          <div className="hidden lg:block w-px h-4 mx-2 bg-border flex-shrink-0" />
          <span className="hidden lg:inline-block font-mono text-[9px] text-muted-foreground uppercase tracking-[0.14em]">DTU EDITION</span>
        </Link>

        {/* Global System Status & Auth */}
        <div className="flex items-center gap-3 sm:gap-4 md:gap-5">
          <div className="flex items-center gap-2 px-2 py-1 bg-destructive/20 border border-destructive/30 rounded-sm backdrop-blur-sm hidden sm:flex">
            <span className="size-1.5 rounded-full animate-blink bg-destructive shadow-[0_0_8px_rgba(var(--color-destructive),0.8)]" />
            <span className="font-mono text-[9px] text-destructive uppercase tracking-[0.14em] font-bold mt-px drop-shadow-sm">LIVE</span>
          </div>

          {user ? (
            <div className="flex items-center gap-2 sm:gap-4">
              <div className="hidden md:flex items-center gap-2 bg-background/50 px-3 py-1.5 rounded-sm border border-border/50 backdrop-blur-sm">
                <User className="size-3 text-primary" />
                <span className="font-mono text-[10px] text-foreground uppercase tracking-[0.14em] mt-px truncate max-w-[80px] lg:max-w-[120px]">
                  {user.name}
                </span>
              </div>
              <button
                onClick={logout}
                className="flex items-center gap-2 bg-destructive/10 hover:bg-destructive/20 text-destructive px-2 sm:px-3 py-1.5 rounded-sm border border-destructive/30 transition-colors cursor-pointer"
                title="Sign Out"
              >
                <LogOut className="size-3" />
                <span className="hidden sm:inline-block font-mono text-[10px] uppercase tracking-[0.1em] mt-px">Logout</span>
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="flex items-center gap-2 bg-primary/10 hover:bg-primary/20 text-primary px-3 sm:px-4 py-1.5 rounded-sm border border-primary/30 transition-colors"
            >
              <span className="font-mono text-[10px] uppercase tracking-[0.1em] mt-px font-bold">Sign In</span>
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
