"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import { LogOut, User, Film } from "lucide-react";
import { useEffect, useState } from "react";

export function Navbar() {
  const { user, logout } = useAuth();
  const pathname = usePathname();

  // 🌙 Dark Mode Logic
  const [theme, setTheme] = useState("light");

  useEffect(() => {
    const saved = localStorage.getItem("theme");
    if (saved) {
      setTheme(saved);
      document.documentElement.classList.toggle("dark", saved === "dark");
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === "light" ? "dark" : "light";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "dark");
  };

  return (
    <nav className="w-full border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-50 shadow-2xl">
      <div className="relative z-10 flex items-center justify-between py-3 md:py-4 px-4 sm:px-6 md:px-8 max-w-360 mx-auto">

        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-2 sm:gap-3 transition-opacity duration-200 hover:opacity-80"
        >
          <div className="relative size-7 sm:size-8 shrink-0 overflow-hidden">
            <Image
              src="/favicons/logo.png"
              alt="Matcha AI Logo"
              fill
              className="object-contain"
            />
          </div>

          <div className="flex items-baseline gap-1">
            <span className="font-display tracking-[0.12em] text-foreground">
              MATCHA
            </span>
            <span className="font-display tracking-[0.12em] text-primary">
              AI
            </span>
          </div>
        </Link>

        {/* Center Links */}
        <div className="hidden md:flex items-center gap-1">
          <Link
            href="/highlights"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-sm ${
              pathname === "/highlights"
                ? "text-primary"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Film className="size-3" />
            Highlights
          </Link>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-3 sm:gap-4 md:gap-5">

          {/* 🌙 Dark Mode Toggle Button */}
          <button
            onClick={toggleTheme}
            className="text-xl px-2 py-1 rounded hover:bg-white/10 transition"
            title="Toggle Theme"
          >
            {theme === "light" ? "🌙" : "☀️"}
          </button>

          {/* Auth Section */}
          {user ? (
            <div className="flex items-center gap-2 sm:gap-4">
              <div className="hidden md:flex items-center gap-2">
                <User className="size-3 text-primary" />
                <span className="text-foreground text-xs">
                  {user.name}
                </span>
              </div>

              <button
                onClick={logout}
                className="flex items-center gap-2 text-destructive px-2 py-1"
              >
                <LogOut className="size-3" />
                Logout
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="text-primary px-3 py-1"
            >
              Sign In
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
