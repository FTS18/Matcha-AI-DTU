"use client";

import Link from "next/link";
import { Terminal, ShieldCheck, ShieldOff, ImagePlus, X } from "lucide-react";
import { useAdmin } from "@/contexts/AdminContext";
import { useRef, useState } from "react";

export function Footer() {
  const { isAdmin, toggleAdmin, adOverlayUrl, setAdOverlayUrl } = useAdmin();
  const clickCount = useRef(0);
  const clickTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [pulse, setPulse] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Triple-click on version number activates admin mode
  const handleVersionClick = () => {
    clickCount.current += 1;
    if (clickTimer.current) clearTimeout(clickTimer.current);
    clickTimer.current = setTimeout(() => { clickCount.current = 0; }, 800);
    if (clickCount.current >= 3) {
      clickCount.current = 0;
      toggleAdmin();
      setPulse(true);
      setTimeout(() => setPulse(false), 600);
    }
  };

  // Convert uploaded file → data-URL and persist in AdminContext
  const handleAdUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const url = ev.target?.result as string;
      if (url) setAdOverlayUrl(url);
    };
    reader.readAsDataURL(file);
    e.target.value = ""; // reset so same file can be re-selected
  };

  return (
    <footer className="w-full border-t border-border bg-card/50 mt-auto">
      <div className="max-w-360 mx-auto px-6 md:px-8 py-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 items-center">

          {/* Left: Branding & Version */}
          <div className="flex flex-row items-center gap-3 sm:gap-4 text-muted-foreground group">
            <div className="flex items-center gap-2">
              <Terminal className="size-4 group-hover:text-primary transition-colors" />
              <span className="font-display text-sm tracking-widest text-foreground">MATCHA AI</span>
            </div>
            <div className="w-px h-3 bg-border" />
            {/* Triple-click to toggle admin mode */}
            <span
              onClick={handleVersionClick}
              className={`font-mono text-[9px] sm:text-[10px] uppercase tracking-widest select-none cursor-default transition-colors ${pulse ? "text-amber-400" : "text-muted-foreground"}`}
              title=""
            >
              v2.1.0-RC
            </span>
          </div>

          {/* Center: System Status */}
          <div className="hidden sm:flex items-center justify-start lg:justify-center gap-6">
            <div className="flex items-center gap-2">
              <span className="size-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
              <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">ORCHESTRATOR OK</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="size-1.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.4)]" />
              <span className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">INFERENCE READY</span>
            </div>
          </div>

          {/* Right: Links + Admin toggle */}
          <div className="flex items-center justify-start sm:justify-end lg:justify-end gap-6 sm:gap-8 font-mono text-[10px] uppercase tracking-widest sm:col-span-2 lg:col-span-1 border-t sm:border-t-0 border-border pt-4 sm:pt-0">
            <Link href="/" className="text-muted-foreground hover:text-primary transition-colors">
              DASHBOARD
            </Link>
            <a href="https://github.com/matcha-ai" target="_blank" rel="noopener noreferrer" className="text-muted-foreground hover:text-primary transition-colors">
              SOURCE
            </a>
            {/* Admin mode toggle button */}
            <button
              onClick={toggleAdmin}
              title={isAdmin ? "Disable admin mode" : "Enable admin mode"}
              className={`flex items-center gap-1.5 transition-all cursor-pointer focus:outline-none px-2 py-1 border ${
                isAdmin
                  ? "border-amber-400/50 text-amber-400 bg-amber-400/10 hover:bg-amber-400/20"
                  : "border-border text-muted-foreground/50 hover:text-amber-400/70 hover:border-amber-400/30 bg-transparent"
              }`}
            >
              {isAdmin
                ? <><ShieldCheck className="size-3" /><span>ADMIN</span></>
                : <><ShieldOff className="size-3" /><span>ADMIN</span></>
              }
            </button>
          </div>

        </div>

        {/* Bottom Rule */}
        <div className="mt-6 pt-4 border-t border-border flex flex-wrap justify-between items-center gap-3 text-[9px] font-mono text-muted-foreground/60 uppercase tracking-widest">
          <span>
            {isAdmin
              ? <span className="text-amber-400/80 flex items-center gap-1.5"><ShieldCheck className="size-3 inline" /> ADMIN MODE ACTIVE</span>
              : "SECURE UPLINK ESTABLISHED"
            }
          </span>

          {/* Ad overlay controls — only visible in admin mode */}
          {isAdmin && (
            <div className="flex items-center gap-2">
              {adOverlayUrl ? (
                <>
                  <img src={adOverlayUrl} alt="ad preview" className="h-6 w-auto rounded border border-amber-400/30 object-contain" />
                  <span className="text-amber-400/70">AD ACTIVE</span>
                  <button
                    onClick={() => setAdOverlayUrl(null)}
                    title="Remove ad overlay"
                    className="text-amber-400/60 hover:text-red-400 transition-colors"
                  >
                    <X className="size-3" />
                  </button>
                </>
              ) : (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  title="Upload sponsor/ad image overlay"
                  className="flex items-center gap-1 text-amber-400/60 hover:text-amber-400 transition-colors border border-amber-400/20 hover:border-amber-400/50 px-2 py-0.5"
                >
                  <ImagePlus className="size-3" />
                  <span>SET AD OVERLAY</span>
                </button>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleAdUpload}
              />
            </div>
          )}

          <span>© {new Date().getFullYear()} MATCHA RESEARCH</span>
        </div>
      </div>
    </footer>
  );
}
