"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { VideoUpload, MatchDashboard } from "@/components/dynamic-sections";
import { useAuth } from "@/contexts/AuthContext";
import { client } from "@/lib/api";

export default function Home() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [stats, setStats] = useState({
    analyses: "00",
    insights: "00",
    footage: "00h"
  });

  useEffect(() => {
    if (!loading && !user) {
      router.push("/login");
    }
  }, [user, loading, router]);

  useEffect(() => {
    if (user) {
      client.getStats().then((data: { totalMatches: number; totalEvents: number; totalHighlights: number; totalDuration: number }) => {
        const hours = (data.totalDuration / 3600).toFixed(1);
        setStats({
          analyses: String(data.totalMatches).padStart(2, '0'),
          insights: String(data.totalEvents).padStart(2, '0'),
          footage: `${hours}h`
        });
      }).catch((err: Error) => {
        console.error("Failed to fetch global stats:", err);
      });
    }
  }, [user]);

  if (loading || !user) {
    return (
      <div className="flex-1 flex items-center justify-center p-4 bg-background min-h-[calc(100vh-80px)]">
        <div className="size-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="relative w-full overflow-hidden">
      {/* ── Hero band ─────────────────────────────────── */}
      <div className="relative min-h-[calc(100vh-80px)] flex flex-col items-center justify-center border-b border-border py-16">
        {/* Full-width Background Image Layer */}
        <div 
          className="absolute inset-0 z-0 bg-cover bg-center"
          style={{ backgroundImage: 'url("/favicons/image.png")' }}
        />
        {/* Blue Tint Overlay Layer */}
        <div className="absolute inset-0 z-0 bg-background/85 backdrop-blur-[2px]" />

        {/* Hero Content Layer */}
        <div className="relative z-10 max-w-360 mx-auto px-6 sm:px-8 w-full flex flex-col items-center text-center gap-6 sm:gap-10">
          
          <div className="flex flex-col items-center opacity-0" style={{ animation: "fadeup 0.8s ease-out forwards" }}>
            <h1 className="font-display flex flex-col items-center leading-[0.85] text-[clamp(48px,12vw,120px)] text-foreground shadow-sm">
              <span>MATCH</span>
              <span 
                className="text-primary drop-shadow-[0_0_20px_rgba(var(--color-primary),0.6)] italic tracking-normal normal-case text-[clamp(36px,9vw,90px)] -mt-1 sm:-mt-4"
                style={{ fontFamily: "Georgia, 'Times New Roman', Times, serif" }}
              >
                Intelligence
              </span>
            </h1>
          </div>

          <p 
            className="font-heading max-w-2xl leading-relaxed text-[14px] sm:text-[19px] font-medium text-foreground/90 drop-shadow-md opacity-0 px-4"
            style={{ animation: "fadeup 0.8s ease-out 0.2s forwards" }}
          >
            Upload raw match footage. The AI pipeline detects goals, fouls, saves &amp; tackles —
            scores each moment — and builds highlight reels with live neural commentary.
          </p>

          {/* Stat counters */}
          <div 
            className="flex w-full sm:w-auto items-center justify-center gap-2 sm:gap-16 mt-4 backdrop-blur-md bg-background/50 px-4 sm:px-12 py-4 sm:py-6 rounded-xl border border-border/60 shadow-2xl relative overflow-hidden group opacity-0"
            style={{ animation: "fadeup 0.8s ease-out 0.4s forwards" }}
          >
            <div className="absolute inset-0 bg-linear-to-r from-primary/5 via-transparent to-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            {[
              { n: stats.analyses, label: "ANALYSES" },
              { n: stats.insights, label: "INSIGHTS" },
              { n: stats.footage,  label: "FOOTAGE" },
            ].map((s) => (
              <div key={s.label} className="text-center px-4 sm:px-0 sm:pr-16 last:pr-0 border-r last:border-r-0 border-border/40 relative z-10 flex-1 sm:flex-none">
                <div className="font-display leading-none text-[28px] sm:text-[54px] text-primary drop-shadow-[0_0_12px_rgba(var(--color-primary),0.5)]">{s.n}</div>
                <div className="font-mono mt-1 sm:mt-2 text-[7px] sm:text-[10.5px] text-muted-foreground uppercase tracking-[0.16em] font-bold">{s.label}</div>
              </div>
            ))}
          </div>

        </div>
      </div>

      {/* ── Main grid ─────────────────────────────────── */}
      <div className="relative z-10 max-w-360 mx-auto px-4 sm:px-8 w-full">
        <div className="grid gap-0 grid-cols-1 xl:grid-cols-[minmax(0,500px)_1fr]">
          {/* LEFT — Upload */}
          <div className="xl:border-r border-b xl:border-b-0 py-8 sm:py-10 xl:pr-10 border-border">
            <div className="flex items-center gap-3 mb-6">
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.14em]">FOOTAGE INPUT</span>
              <div className="flex-1 h-px bg-border" />
            </div>
            <VideoUpload />

            {/* Capability grid */}
            <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-px bg-border border border-border">
              {[
                "⚽  GOAL DETECTION",
                "🟡  FOUL DETECTION",
                "🧤  SAVE DETECTION",
                "⚡  MOTION ANALYSIS",
                "🎬  HIGHLIGHT REELS",
                "🎙  NEURAL COMMENTARY",
              ].map((f) => (
                <div
                  key={f}
                  className="font-mono px-4 py-3 transition-colors text-[9px] bg-card text-muted-foreground uppercase tracking-widest flex items-center"
                >
                  {f}
                </div>
              ))}
            </div>
          </div>

          {/* RIGHT — Feed */}
          <div className="py-8 sm:py-10 xl:pl-10">
            <div className="flex items-center gap-3 mb-6">
              <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.14em]">ANALYSIS FEED</span>
              <div className="flex-1 h-px bg-border" />
            </div>
            <MatchDashboard />
          </div>
        </div>
      </div>
    </div>
  );
}
