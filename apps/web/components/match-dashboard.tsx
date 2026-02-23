"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import { CheckCircle2, Loader2, Upload, XCircle, LayoutGrid, Clock, AlertTriangle, PlayCircle, BarChart3, Scissors, RefreshCw } from "lucide-react";
import { useMatches } from "@/hooks/useMatches";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { STATUS_CONFIG as SHARED_STATUS_CONFIG, PIPELINE_STAGES, formatTime, timeAgo } from "@matcha/shared";

const API_BASE = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL || process.env.NEXT_PUBLIC_API_URL?.replace("/api/v1", "") || "http://localhost:4000";


const THEME_MAP: Record<string, { color: string }> = {
  success: { color: "text-emerald-400 bg-emerald-500/15 border-emerald-500/40" },
  info: { color: "text-blue-400 bg-blue-500/15 border-blue-500/40" },
  warning: { color: "text-amber-400 bg-amber-500/15 border-amber-500/40" },
  error: { color: "text-red-400 bg-red-500/15 border-red-500/40" },
};

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  COMPLETED: { ...SHARED_STATUS_CONFIG.COMPLETED, ...THEME_MAP[SHARED_STATUS_CONFIG.COMPLETED.theme], icon: <CheckCircle2 className="w-3 h-3" /> },
  PROCESSING: { ...SHARED_STATUS_CONFIG.PROCESSING, ...THEME_MAP[SHARED_STATUS_CONFIG.PROCESSING.theme], icon: <Loader2 className="w-3 h-3 animate-spin" /> },
  UPLOADED: { ...SHARED_STATUS_CONFIG.UPLOADED, ...THEME_MAP[SHARED_STATUS_CONFIG.UPLOADED.theme], icon: <Upload className="w-3 h-3" /> },
  FAILED: { ...SHARED_STATUS_CONFIG.FAILED, ...THEME_MAP[SHARED_STATUS_CONFIG.FAILED.theme], icon: <XCircle className="w-3 h-3" /> },
};

const FILTER_OPTIONS = ["ALL", "COMPLETED", "PROCESSING", "UPLOADED", "FAILED"] as const;
type FilterOption = typeof FILTER_OPTIONS[number];

const MiniHeatmap = ({ matches }: { matches: any }) => (
  <svg viewBox="0 0 100 60" className="w-full h-full opacity-60 group-hover:opacity-100 transition-opacity duration-500">
    <rect x="0" y="0" width="100" height="60" fill="var(--muted)" fillOpacity="0.1" rx="2" />
    <circle cx="20" cy="30" r="15" fill="var(--accent)" fillOpacity="0.2" filter="blur(4px)" />
    <circle cx="80" cy="20" r="10" fill="var(--primary)" fillOpacity="0.2" filter="blur(4px)" />
    <path d="M50 0 L50 60" stroke="currentColor" strokeOpacity="0.1" strokeWidth="0.5" />
    <circle cx="50" cy="30" r="8" fill="none" stroke="currentColor" strokeOpacity="0.1" strokeWidth="0.5" />
  </svg>
);

export const MatchDashboard = React.memo(function MatchDashboardContent() {
  const { matches, loading, progressMap, stageMap, deleteMatch, reanalyzeMatch, refetch } = useMatches();
  const [filter, setFilter] = useState<FilterOption>("ALL");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [reanalyzingId, setReanalyzingId] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await refetch();
    setTimeout(() => setIsRefreshing(false), 1000);
  }, [refetch]);

  const handleDelete = useCallback(async (id: string) => {
    setDeletingId(id);
    await deleteMatch(id);
    setDeletingId(null);
    setConfirmId(null);
  }, [deleteMatch]);

  const handleReanalyze = useCallback(async (id: string) => {
    setReanalyzingId(id);
    await reanalyzeMatch(id);
    setReanalyzingId(null);
  }, [reanalyzeMatch]);

  const visible = Array.isArray(matches)
    ? (filter === "ALL" ? matches : matches.filter((m) => m.status?.toUpperCase() === filter.toUpperCase()))
    : [];

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 border border-dashed border-border/50 bg-card/30">
        <Loader2 className="size-6 text-accent animate-spin mb-4" />
        <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.2em]">
          INITIALIZING ANALYTICS ENGINE...
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Broadcast Tab Navigation */}
      {matches.length > 0 && (
        <div className="flex gap-1 border-b border-border overflow-x-auto pb-px hide-scrollbar">
          <div className="flex items-center px-3 sm:px-4 py-2 border-r border-border bg-muted/20 shrink-0">
            <LayoutGrid className="size-3 sm:size-3.5 text-muted-foreground mr-1.5 sm:mr-2" />
            <span className="font-mono text-[9px] sm:text-[10px] text-muted-foreground uppercase tracking-[0.15em] whitespace-nowrap">
              DATA FEED
            </span>
            <button
              onClick={handleRefresh}
              className="ml-2 p-1 hover:bg-white/10 rounded-full transition-colors group"
              title="Sync Feed"
            >
              <RefreshCw className={cn("size-3 text-muted-foreground group-hover:text-primary transition-all", isRefreshing && "animate-spin text-primary")} />
            </button>
          </div>
          {FILTER_OPTIONS.map((f) => {
            const count = f === "ALL" ? matches.length : matches.filter((m) => m.status === f).length;
            const active = filter === f;
            return (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`group relative flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-5 py-2 sm:py-2.5 transition-all duration-300 shrink-0 ${active ? "bg-accent/10" : "hover:bg-muted/40"
                  }`}
              >
                <div className={`font-mono text-[9px] sm:text-[10px] uppercase tracking-widest transition-colors whitespace-nowrap ${active ? "text-accent font-semibold" : "text-muted-foreground group-hover:text-foreground"
                  }`}>
                  {f === "ALL" ? "MASTER FEED" : STATUS_CONFIG[f]?.label ?? f}
                </div>
                {count > 0 && (
                  <div className={`px-1.5 py-0.5 font-mono text-[8.5px] sm:text-[9px] rounded-sm ${active ? "bg-accent/20 text-accent" : "bg-muted text-muted-foreground group-hover:bg-border group-hover:text-foreground"
                    }`}>
                    {count}
                  </div>
                )}
                {/* Active Indicator Line */}
                {active && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent shadow-[0_-2px_8px_rgba(var(--color-accent),0.5)]" />
                )}
              </button>
            );
          })}
        </div>
      )}


      {/* Empty State */}
      {!visible.length && (
        <div className="flex flex-col items-center justify-center py-20 border border-dashed border-border/60 bg-[radial-gradient(ellipse_at_center,var(--surface-2)_0%,transparent_100%)]">
          <AlertTriangle className="size-8 text-muted-foreground/50 mb-4" />
          <h3 className="font-display text-xl tracking-widest text-muted-foreground uppercase opacity-80">
            {matches.length ? "NO SESSIONS RECORDED" : "NO MATCH DATA"}
          </h3>
          <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.15em] mt-2 opacity-60">
            {matches.length ? "ADJUST FILTER PARAMETERS" : "UPLOAD A VIDEO TO BEGIN AUTOMATED ANALYSIS"}
          </p>
        </div>
      )}

      {/* Match Grid / List */}
      <motion.div
        className="grid grid-cols-1 gap-4"
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: {
              staggerChildren: 0.05
            }
          }
        }}
      >
        <AnimatePresence mode="popLayout">
          {visible.map((m) => {
            const cfg = STATUS_CONFIG[m.status] ?? STATUS_CONFIG.UPLOADED;
            const isConfirming = confirmId === m.id;
            const isDeleting = deletingId === m.id;
            const progress = progressMap[m.id] ?? m.progress ?? 0;
            const isProcessing = m.status === "PROCESSING" || m.status === "UPLOADED";
            const safeProgress = Math.max(0, Math.min(99, Math.round(progress)));

            // Re-map colors to match our theme strictly
            const accentColor = m.status === "COMPLETED" ? "var(--green)"
              : m.status === "PROCESSING" ? "oklch(60% 0.15 250)" // Blue
                : m.status === "FAILED" ? "var(--red)"
                  : "var(--amber-dim)";

            const formattedDate = new Date(m.createdAt).toLocaleDateString('en-US', {
              month: 'short', day: 'numeric', year: 'numeric'
            });

            const formattedTime = new Date(m.createdAt).toLocaleTimeString('en-US', {
              hour: '2-digit', minute: '2-digit'
            });

            return (
              <motion.div
                key={m.id}
                layout
                variants={{ hidden: { opacity: 0, y: 20 }, show: { opacity: 1, y: 0 } }}
                exit={{ opacity: 0, scale: 0.98, filter: "brightness(0.5)", transition: { duration: 0.2 } }}
                className="card relative group bg-card/40 backdrop-blur-md border border-white/5 transition-all duration-300 hover:bg-card/60 hover:border-white/10 overflow-hidden"
              >
                {/* Left accent bar (desktop only) */}
                <div
                  className="hidden lg:block absolute left-0 top-0 bottom-0 w-1 transition-all duration-500 group-hover:w-1.5"
                  style={{ backgroundColor: accentColor, opacity: 0.8, boxShadow: `4px 0 20px -4px ${accentColor}` }}
                />


                {/* ════ MOBILE LAYOUT (hidden on lg+) ═══════════════════════ */}
                <div className="lg:hidden">
                  {/* Cinematic full-width thumbnail */}
                  <Link href={`/matches/${m.id}`} className="block relative w-full h-40 overflow-hidden bg-black/70">
                    {(m.thumbnailUrl || m.heatmapUrl) ? (
                      <img
                        src={(m.thumbnailUrl ?? m.heatmapUrl)!.startsWith("http")
                          ? (m.thumbnailUrl ?? m.heatmapUrl)!
                          : `${API_BASE}${m.thumbnailUrl ?? m.heatmapUrl}`}
                        alt="Match preview"
                        className={`w-full h-full transition-all duration-700 group-hover:scale-105 ${m.thumbnailUrl ? "object-cover saturate-75 group-hover:saturate-100" : "object-contain p-6 opacity-40"}`}
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <PlayCircle className="size-14 text-white/10" />
                      </div>
                    )}
                    <div className="absolute inset-0 bg-linear-to-t from-black/90 via-black/20 to-transparent" />
                    <div className="absolute inset-0" style={{ background: `linear-gradient(135deg, ${accentColor}18 0%, transparent 60%)` }} />
                    {/* Status badge */}
                    <div className="absolute top-3 left-3">
                      <div className={`flex items-center gap-1 px-2 py-1 border backdrop-blur-md font-mono text-[9px] uppercase tracking-widest font-bold bg-black/60 ${cfg.color} ${m.status === "PROCESSING" ? "animate-pulse" : ""}`}>
                        {cfg.icon}&nbsp;{cfg.label}
                      </div>
                    </div>
                    {/* Time ago */}
                    <div className="absolute top-3 right-3">
                      <span className="font-mono text-[8px] text-white/40 uppercase tracking-widest bg-black/50 backdrop-blur-sm px-2 py-1 border border-white/5">
                        {timeAgo(m.createdAt)}
                      </span>
                    </div>
                    {/* Title at bottom */}
                    <div className="absolute bottom-0 left-0 right-0 px-3 pb-3 pt-8">
                      <h4 className="font-display text-sm tracking-[0.05em] text-white font-semibold truncate">{formattedDate} — Analysis</h4>
                      <p className="font-mono text-[8px] text-white/30 uppercase tracking-widest mt-0.5">{formattedTime} · {m.id.split("-")[0]}</p>
                    </div>
                  </Link>

                  {/* Progress bar */}
                  {isProcessing && (
                    <div className="px-4 py-2.5 bg-blue-500/5 border-b border-blue-500/15">
                      <div className="flex items-center justify-between mb-1.5">
                        <span className="font-mono text-[8px] uppercase tracking-widest text-blue-300/70">
                          {stageMap[m.id] ? (PIPELINE_STAGES[stageMap[m.id]] ?? stageMap[m.id]) : (m.status === "UPLOADED" && safeProgress === 0 ? "Queued" : "Analysing")}
                        </span>
                        <span className="font-mono text-[10px] tabular-nums text-blue-300 font-bold">{safeProgress}%</span>
                      </div>
                      <div className="h-1 w-full bg-blue-950/50 overflow-hidden rounded-full">
                        <div className="h-full bg-linear-to-r from-blue-500 via-cyan-400 to-emerald-400 transition-all duration-500 rounded-full" style={{ width: `${safeProgress}%` }} />
                      </div>
                    </div>
                  )}

                  {/* Bottom action strip */}
                  <div className="flex items-stretch divide-x divide-white/5 bg-black/30 border-t border-white/5">
                    <Link href={`/matches/${m.id}`} className="flex-1 flex items-center justify-around px-2 py-3">
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="font-mono text-[7px] text-white/25 uppercase tracking-widest">Duration</span>
                        <span className="font-mono text-xs text-white/60 tabular-nums">{m.duration ? formatTime(m.duration) : "--:--"}</span>
                      </div>
                      <div className="w-px h-6 bg-white/10" />
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="font-mono text-[7px] text-white/25 uppercase tracking-widest">Events</span>
                        <span className="font-display text-sm text-accent">{m.status === "COMPLETED" ? m._count.events : "--"}</span>
                      </div>
                      <div className="w-px h-6 bg-white/10" />
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="font-mono text-[7px] text-white/25 uppercase tracking-widest">Clips</span>
                        <span className="font-display text-sm text-primary">{m.status === "COMPLETED" ? m._count.highlights : "--"}</span>
                      </div>
                    </Link>
                    <div className="flex items-stretch divide-x divide-white/5">
                      <Link href={`/matches/${m.id}#highlights`} className="flex items-center justify-center w-11 text-accent/60 hover:text-accent hover:bg-accent/10 transition-colors" title="Highlights">
                        <Scissors className="size-4" />
                      </Link>
                      {!isConfirming ? (
                        <>
                          <button
                            onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleReanalyze(m.id); }}
                            disabled={reanalyzingId === m.id || m.status === "PROCESSING"}
                            className="flex items-center justify-center w-11 text-muted-foreground hover:text-accent hover:bg-accent/10 disabled:opacity-30 transition-colors"
                          >
                            <RefreshCw className={`size-4 ${reanalyzingId === m.id || m.status === "PROCESSING" ? "animate-spin text-accent" : ""}`} />
                          </button>
                          <button
                            onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(m.id); }}
                            className="flex items-center justify-center w-11 text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                          >
                            <XCircle className="size-4" />
                          </button>
                        </>
                      ) : (
                        <div className="flex items-stretch">
                          <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(m.id); }} disabled={isDeleting} className="font-mono px-3 text-[9px] bg-destructive text-white uppercase tracking-widest font-bold flex items-center gap-1">
                            {isDeleting ? <Loader2 className="size-3 animate-spin" /> : "DEL"}
                          </button>
                          <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(null); }} className="font-mono w-10 text-[10px] text-muted-foreground hover:bg-white/5 border-l border-white/10 flex items-center justify-center">✕</button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* ════ DESKTOP LAYOUT (lg+) — compact horizontal row ════════ */}
                <div className="hidden lg:flex lg:h-18 items-stretch relative overflow-hidden">
                  <div className="flex flex-1 items-stretch">
                    <Link href={`/matches/${m.id}`} className="w-30 h-full shrink-0 relative overflow-hidden group/thumb border-r border-white/10 bg-black/40">
                      {m.thumbnailUrl ? (
                        <img src={m.thumbnailUrl.startsWith("http") ? m.thumbnailUrl : `${API_BASE}${m.thumbnailUrl}`} alt="Preview" className="w-full h-full object-cover opacity-60 group-hover/thumb:opacity-100 transition-all duration-700 scale-110 group-hover/thumb:scale-100 saturate-50 group-hover/thumb:saturate-100" />
                      ) : (
                        <div className="w-full h-full p-2">
                          <MiniHeatmap matches={matches} />
                        </div>
                      )}
                      <div className="absolute inset-0 bg-linear-to-r from-black/60 via-transparent to-transparent opacity-60" />
                    </Link>
                    <Link href={`/matches/${m.id}`} className="flex-1 flex flex-col justify-center px-6 py-2 min-w-0 focus:outline-none group/id">
                      <div className="flex items-center gap-2 mb-1">
                        <div className={`px-1.5 py-0.5 border font-mono text-[7px] uppercase tracking-widest font-bold shrink-0 ${cfg.color} ${m.status === "PROCESSING" ? "animate-pulse" : ""}`}>{cfg.label}</div>
                        <h4 className="font-display text-base tracking-[0.05em] text-foreground group-hover/id:text-white transition-colors truncate">{formattedDate} — Analysis</h4>
                      </div>
                      <p className="font-mono text-[9px] text-muted-foreground/40 uppercase tracking-widest truncate">{formattedTime} • ID: {m.id.split("-")[0]}</p>
                      {isProcessing && (
                        <div className="mt-1.5 space-y-1">
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-mono text-[7px] uppercase tracking-[0.15em] text-muted-foreground/70">{stageMap[m.id] ? (PIPELINE_STAGES[stageMap[m.id]] ?? stageMap[m.id]) : (m.status === "UPLOADED" && safeProgress === 0 ? "Queued" : "Processing")}</span>
                            <span className="font-mono text-[9px] tabular-nums text-blue-300">{safeProgress}%</span>
                          </div>
                          <div className="h-1 w-full bg-white/10 overflow-hidden rounded-sm">
                            <div className="h-full bg-linear-to-r from-blue-500 to-cyan-300 transition-all duration-500" style={{ width: `${safeProgress}%` }} />
                          </div>
                        </div>
                      )}
                    </Link>
                  </div>
                  <div className="flex items-stretch border-l border-white/5">
                    <div className="flex items-stretch divide-x divide-white/5">
                      {[
                        { v: m.duration ? formatTime(m.duration) : "--:--", cls: "font-mono text-[10px] text-white/70 tabular-nums" },
                        { v: m.status === "COMPLETED" ? m._count.events.toString().padStart(2, "0") : "--", cls: "font-display text-[14px] text-accent drop-shadow-[0_0_8px_rgba(var(--color-accent),0.4)]" },
                        { v: m.status === "COMPLETED" ? m._count.highlights.toString().padStart(2, "0") : "--", cls: "font-display text-[14px] text-primary drop-shadow-[0_0_8px_rgba(var(--color-primary),0.4)]" },
                      ].map((stat, i) => (
                        <div key={i} className="w-20 flex items-center justify-center bg-white/1"><span className={stat.cls}>{stat.v}</span></div>
                      ))}
                    </div>
                    <div className="flex items-center justify-center w-30 border-l border-white/5">
                      <Link href={`/matches/${m.id}#highlights`} className="flex items-center gap-1.5 px-3 py-1.5 bg-accent/5 hover:bg-accent/15 border border-accent/20 hover:border-accent/40 text-accent transition-all rounded-sm">
                        <Scissors className="size-3" />
                        <span className="font-mono text-[8px] uppercase tracking-widest font-bold">Highlights</span>
                      </Link>
                    </div>
                    <div className="flex items-center justify-center w-30 border-l border-white/10 bg-white/2">
                      {!isConfirming ? (
                        <div className="flex items-center gap-2">
                          <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleReanalyze(m.id); }} disabled={reanalyzingId === m.id || m.status === "PROCESSING"} className={`flex items-center justify-center size-8 bg-white/5 border border-white/5 transition-all rounded-full ${reanalyzingId === m.id || m.status === "PROCESSING" ? "text-accent border-accent/30 cursor-wait" : "hover:bg-accent/10 text-muted-foreground hover:text-accent hover:border-accent/30"}`} title={m.status === "PROCESSING" ? "Analysis in progress" : "Reanalyze"}>
                            <RefreshCw className={`size-3.5 ${reanalyzingId === m.id || m.status === "PROCESSING" ? "animate-spin" : ""}`} />
                          </button>
                          <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(m.id); }} className="flex items-center justify-center size-8 bg-white/5 hover:bg-destructive/10 text-muted-foreground hover:text-destructive border border-white/5 hover:border-destructive/30 transition-all rounded-full" title="Delete">
                            <XCircle className="size-3.5" />
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center bg-card border border-destructive/20 overflow-hidden scale-90">
                          <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); handleDelete(m.id); }} disabled={isDeleting} className="font-mono px-3 py-2 text-[8px] bg-destructive text-white uppercase tracking-widest font-bold hover:brightness-110 flex items-center gap-1">
                            {isDeleting ? <><Loader2 className="size-2.5 animate-spin" />...</> : "DEL"}
                          </button>
                          <button onClick={(e) => { e.preventDefault(); e.stopPropagation(); setConfirmId(null); }} className="font-mono px-3 py-2 text-[8px] text-muted-foreground hover:bg-white/5 uppercase tracking-widest border-l border-white/10">X</button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

              </motion.div>

            );
          })}
        </AnimatePresence>
      </motion.div>
    </div>
  );
});



