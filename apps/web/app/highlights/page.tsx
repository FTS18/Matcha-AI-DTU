"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import Link from "next/link";
import {
  Play, Pause, ChevronLeft, ChevronRight, Film, Zap, Shield,
  Target, AlertTriangle, Star, Clock, ArrowLeft, Loader2,
  Volume2, VolumeX, Maximize, SkipForward,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { createApiClient } from "@matcha/shared";
import { formatTime } from "@matcha/shared";
import type { Highlight, MatchSummary } from "@matcha/shared";

// ── Event config ──────────────────────────────────────────────────────────────
const EV_STYLES: Record<string, { label: string; icon: React.ReactNode; color: string; bg: string; border: string }> = {
  GOAL:      { label: "Goal",      icon: <Target className="size-3" />,       color: "text-emerald-400", bg: "bg-emerald-400/10", border: "border-emerald-400/30" },
  TACKLE:    { label: "Tackle",    icon: <Zap className="size-3" />,          color: "text-amber-400",   bg: "bg-amber-400/10",   border: "border-amber-400/30" },
  FOUL:      { label: "Foul",      icon: <AlertTriangle className="size-3" />,color: "text-red-400",     bg: "bg-red-400/10",     border: "border-red-400/30" },
  SAVE:      { label: "Save",      icon: <Shield className="size-3" />,       color: "text-blue-400",    bg: "bg-blue-400/10",    border: "border-blue-400/30" },
  CELEBRATION:{ label: "Celeb",   icon: <Star className="size-3" />,          color: "text-purple-400",  bg: "bg-purple-400/10",  border: "border-purple-400/30" },
  HIGHLIGHT: { label: "Highlight", icon: <Film className="size-3" />,         color: "text-zinc-300",    bg: "bg-zinc-400/10",    border: "border-zinc-400/20" },
};
const EV_DEFAULT = EV_STYLES["HIGHLIGHT"];
function evStyle(type: string | null) { return EV_STYLES[(type ?? "").toUpperCase()] ?? EV_DEFAULT; }

// ── Types ─────────────────────────────────────────────────────────────────────
interface FeedItem {
  matchId: string;
  matchCreatedAt: string;
  highlight: Highlight;
  videoSrc: string;
}

// ── Mini video card ───────────────────────────────────────────────────────────
function HighlightCard({
  item, active, onActivate, apiBase,
}: {
  item: FeedItem;
  active: boolean;
  onActivate: () => void;
  apiBase: string;
}) {
  const vidRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const cfg = evStyle(item.highlight.eventType);

  // Auto-play when active
  useEffect(() => {
    const v = vidRef.current;
    if (!v) return;
    if (active) {
      v.currentTime = 0;
      v.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      v.pause();
      setPlaying(false);
    }
  }, [active]);

  const toggle = useCallback(() => {
    const v = vidRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else { v.pause(); setPlaying(false); }
  }, []);

  const pct = duration > 0 ? (current / duration) * 100 : 0;
  const clipDur = item.highlight.endTime - item.highlight.startTime;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onActivate}
      className={`relative flex flex-col bg-zinc-950 border transition-all duration-300 cursor-pointer group
        ${active ? "border-primary/60 shadow-[0_0_30px_rgba(16,185,129,0.12)]" : "border-zinc-800/80 hover:border-zinc-700"}`}
    >
      {/* ── Video ── */}
      <div className="relative aspect-video bg-black overflow-hidden">
        <video
          ref={vidRef}
          src={item.videoSrc}
          muted={muted}
          loop
          playsInline
          preload="metadata"
          className="w-full h-full object-cover"
          onLoadedMetadata={() => setDuration(vidRef.current?.duration ?? 0)}
          onTimeUpdate={() => setCurrent(vidRef.current?.currentTime ?? 0)}
        />

        {/* Active badge */}
        {active && (
          <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-primary/90 text-black text-[10px] font-black uppercase tracking-wider px-2 py-1 rounded-full">
            <span className="size-1.5 rounded-full bg-black animate-pulse" />
            Now Playing
          </div>
        )}

        {/* Play overlay */}
        <AnimatePresence>
          {!playing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex items-center justify-center bg-black/30"
              onClick={e => { e.stopPropagation(); onActivate(); toggle(); }}
            >
              <div className="size-14 rounded-full bg-black/50 backdrop-blur-sm flex items-center justify-center border border-white/20">
                <Play className="size-6 fill-white text-white ml-1" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Controls overlay (always on active) */}
        {active && (
          <div
            className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/80 to-transparent px-3 pb-2 pt-8"
            onClick={e => e.stopPropagation()}
          >
            {/* Progress bar */}
            <div className="h-0.5 bg-white/20 mb-2 overflow-hidden rounded-full">
              <div
                className="h-full bg-primary transition-all duration-300 rounded-full"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <button
                  onClick={toggle}
                  className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
                >
                  {playing ? <Pause className="size-4 fill-white text-white" /> : <Play className="size-4 fill-white text-white ml-0.5" />}
                </button>
                <button
                  onClick={() => { setMuted(m => !m); if (vidRef.current) vidRef.current.muted = !muted; }}
                  className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
                >
                  {muted ? <VolumeX className="size-4 text-white/70" /> : <Volume2 className="size-4 text-white/70" />}
                </button>
                <span className="text-[10px] text-white/60 font-mono tabular-nums">
                  {formatTime(current)} / {formatTime(duration || clipDur)}
                </span>
              </div>
              <Link
                href={item.videoSrc}
                target="_blank"
                onClick={e => e.stopPropagation()}
                className="p-1.5 rounded-full hover:bg-white/20 transition-colors"
                title="Open fullscreen"
              >
                <Maximize className="size-4 text-white/60" />
              </Link>
            </div>
          </div>
        )}
      </div>

      {/* ── Meta ── */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <span className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 border ${cfg.bg} ${cfg.border} ${cfg.color}`}>
            {cfg.icon} {cfg.label}
          </span>
          <span className="text-[10px] font-mono text-zinc-500 tabular-nums">
            {Math.round(clipDur)}s
          </span>
        </div>

        {item.highlight.commentary && (
          <p className="text-xs text-zinc-400 line-clamp-2 leading-relaxed">
            {item.highlight.commentary}
          </p>
        )}

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[10px] text-zinc-600">
            <Clock className="size-3" />
            <span className="font-mono">{formatTime(item.highlight.startTime)}</span>
          </div>
          <Link
            href={`/matches/${item.matchId}`}
            onClick={e => e.stopPropagation()}
            className="text-[10px] text-primary/70 hover:text-primary transition-colors font-medium uppercase tracking-wide"
          >
            Full Match →
          </Link>
        </div>
      </div>
    </motion.div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────
export default function HighlightsFeedPage() {
  const [feed, setFeed] = useState<FeedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeIdx, setActiveIdx] = useState(0);
  const [filterType, setFilterType] = useState<string>("ALL");

  const API_BASE = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ?? "http://localhost:4000";
  const client = useMemo(() => createApiClient(API_BASE), [API_BASE]);

  useEffect(() => {
    (async () => {
      try {
        const matches: MatchSummary[] = await client.getMatches();
        const items: FeedItem[] = [];

        // Load details for completed matches to get highlight clip URLs
        const completed = matches.filter(m => m.status === "COMPLETED").slice(0, 12);
        await Promise.all(
          completed.map(async (ms) => {
            try {
              const m = await client.getMatch(ms.id);
              for (const h of m.highlights) {
                if (h.videoUrl) {
                  items.push({
                    matchId: m.id,
                    matchCreatedAt: m.createdAt,
                    highlight: h,
                    videoSrc: client.getAssetUrl(h.videoUrl),
                  });
                }
              }
            } catch { /* skip bad match */ }
          })
        );

        // Sort by score descending
        items.sort((a, b) => b.highlight.score - a.highlight.score);
        setFeed(items);
      } catch (err) {
        console.error("Highlights feed load failed:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, [client]);

  const filtered = useMemo(() => {
    if (filterType === "ALL") return feed;
    return feed.filter(f => (f.highlight.eventType ?? "").toUpperCase() === filterType);
  }, [feed, filterType]);

  const allTypes = useMemo(() => {
    const s = new Set(feed.map(f => (f.highlight.eventType ?? "HIGHLIGHT").toUpperCase()));
    return Array.from(s);
  }, [feed]);

  const prev = () => setActiveIdx(i => Math.max(0, i - 1));
  const next = () => setActiveIdx(i => Math.min(filtered.length - 1, i + 1));

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header */}
      <nav className="sticky top-0 z-30 border-b border-border/50 bg-background/80 backdrop-blur-md px-4 sm:px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="flex items-center gap-1.5 text-muted-foreground hover:text-foreground transition-colors text-sm font-medium"
            >
              <ArrowLeft className="size-4" /> Dashboard
            </Link>
            <div className="h-4 w-px bg-border" />
            <div className="flex items-center gap-2">
              <Film className="size-4 text-primary" />
              <h1 className="text-sm font-bold text-foreground uppercase tracking-widest">
                Highlights Feed
              </h1>
            </div>
          </div>

          {/* Type filters */}
          <div className="hidden sm:flex items-center gap-1.5">
            <button
              onClick={() => { setFilterType("ALL"); setActiveIdx(0); }}
              className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1.5 border transition-all
                ${filterType === "ALL" ? "bg-primary/15 border-primary/40 text-primary" : "border-border text-muted-foreground hover:border-border/80 hover:text-foreground"}`}
            >
              All ({feed.length})
            </button>
            {allTypes.map(t => {
              const cfg = evStyle(t);
              return (
                <button
                  key={t}
                  onClick={() => { setFilterType(t); setActiveIdx(0); }}
                  className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-3 py-1.5 border transition-all
                    ${filterType === t ? `${cfg.bg} ${cfg.border} ${cfg.color}` : "border-border text-muted-foreground hover:border-border/80"}`}
                >
                  {cfg.icon} {cfg.label} ({feed.filter(f => (f.highlight.eventType ?? "").toUpperCase() === t).length})
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        {loading && (
          <div className="flex flex-col items-center justify-center py-32 gap-4">
            <Loader2 className="size-8 animate-spin text-primary/60" />
            <p className="text-sm text-muted-foreground">Loading highlights…</p>
          </div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-32 gap-4 border border-dashed border-border rounded-xl">
            <Film className="size-12 text-muted-foreground/20" />
            <p className="text-sm text-muted-foreground">
              {feed.length === 0
                ? "No highlights found — upload and analyze a match first."
                : "No highlights for this filter."}
            </p>
            <Link
              href="/"
              className="text-xs text-primary hover:text-primary/80 font-semibold uppercase tracking-wide"
            >
              Upload a match →
            </Link>
          </div>
        )}

        {!loading && filtered.length > 0 && (
          <div className="space-y-8">
            {/* ── Stats bar ── */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-sm text-muted-foreground">
                  <span className="text-foreground font-bold">{filtered.length}</span> clips
                </span>
                {filterType !== "ALL" && (
                  <button
                    onClick={() => { setFilterType("ALL"); setActiveIdx(0); }}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors underline"
                  >
                    Clear filter
                  </button>
                )}
              </div>
              {/* Navigation */}
              <div className="flex items-center gap-2">
                <button
                  onClick={prev}
                  disabled={activeIdx === 0}
                  className="p-2 border border-border rounded-lg text-muted-foreground hover:text-foreground hover:border-border/80 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="size-4" />
                </button>
                <span className="text-xs text-muted-foreground tabular-nums font-mono min-w-12 text-center">
                  {activeIdx + 1} / {filtered.length}
                </span>
                <button
                  onClick={next}
                  disabled={activeIdx >= filtered.length - 1}
                  className="p-2 border border-border rounded-lg text-muted-foreground hover:text-foreground hover:border-border/80 transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <ChevronRight className="size-4" />
                </button>
                <button
                  onClick={next}
                  disabled={activeIdx >= filtered.length - 1}
                  className="hidden sm:flex items-center gap-1.5 text-xs font-semibold text-primary border border-primary/30 hover:bg-primary/5 px-3 py-2 rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed uppercase tracking-wide"
                >
                  <SkipForward className="size-3.5" /> Next
                </button>
              </div>
            </div>

            {/* ── Mobile type filter ── */}
            <div className="flex sm:hidden gap-1.5 overflow-x-auto pb-1">
              <button
                onClick={() => { setFilterType("ALL"); setActiveIdx(0); }}
                className={`shrink-0 text-[10px] font-bold uppercase tracking-wider px-3 py-1.5 border transition-all
                  ${filterType === "ALL" ? "bg-primary/15 border-primary/40 text-primary" : "border-border text-muted-foreground"}`}
              >
                All
              </button>
              {allTypes.map(t => {
                const cfg = evStyle(t);
                return (
                  <button
                    key={t}
                    onClick={() => { setFilterType(t); setActiveIdx(0); }}
                    className={`shrink-0 inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-3 py-1.5 border transition-all
                      ${filterType === t ? `${cfg.bg} ${cfg.border} ${cfg.color}` : "border-border text-muted-foreground"}`}
                  >
                    {cfg.icon} {cfg.label}
                  </button>
                );
              })}
            </div>

            {/* ── Grid ── */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              <AnimatePresence mode="popLayout">
                {filtered.map((item, idx) => (
                  <HighlightCard
                    key={`${item.matchId}-${item.highlight.id}`}
                    item={item}
                    active={idx === activeIdx}
                    onActivate={() => setActiveIdx(idx)}
                    apiBase={API_BASE}
                  />
                ))}
              </AnimatePresence>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

