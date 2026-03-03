"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import Link from "next/link";
import {
  Play, Pause, ChevronLeft, ChevronRight, Film, Zap, Shield,
  Target, AlertTriangle, Star, Clock, ArrowLeft, Loader2,
  Volume2, VolumeX, Maximize, Heart,
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

// ── Mobile Shorts-style card (full-screen snap) ───────────────────────────────
function ShortCard({
  item, onNext, onPrev, hasNext, hasPrev,
}: {
  item: FeedItem;
  onNext: () => void;
  onPrev: () => void;
  hasNext: boolean;
  hasPrev: boolean;
}) {
  const vidRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const cfg = evStyle(item.highlight.eventType);
  const pct = duration > 0 ? (current / duration) * 100 : 0;

  // Auto-play on mount
  useEffect(() => {
    const v = vidRef.current;
    if (!v) return;
    v.currentTime = 0;
    v.play().then(() => setPlaying(true)).catch(() => {});
    return () => { v.pause(); };
  }, [item.videoSrc]);

  const toggle = useCallback(() => {
    const v = vidRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setPlaying(true); }
    else { v.pause(); setPlaying(false); }
  }, []);

  const toggleMute = useCallback(() => {
    setMuted(m => {
      if (vidRef.current) vidRef.current.muted = !m;
      return !m;
    });
  }, []);

  return (
    <div className="relative w-full h-dvh bg-black flex items-center justify-center overflow-hidden snap-start snap-always">
      {/* Video */}
      <video
        ref={vidRef}
        src={item.videoSrc}
        muted={muted}
        loop
        playsInline
        preload="auto"
        className="absolute inset-0 w-full h-full object-cover"
        onLoadedMetadata={() => setDuration(vidRef.current?.duration ?? 0)}
        onTimeUpdate={() => setCurrent(vidRef.current?.currentTime ?? 0)}
        onClick={toggle}
      />

      {/* Dark gradient overlays */}
      <div className="absolute inset-0 bg-linear-to-t from-black/70 via-transparent to-black/30 pointer-events-none" />

      {/* Progress bar — top */}
      <div className="absolute top-0 left-0 right-0 h-0.5 bg-white/10 z-20">
        <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
      </div>

      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 px-4 pt-6 pb-3 flex items-center justify-between z-20">
        <Link href="/" className="flex items-center gap-1.5 text-white/70 hover:text-white text-xs font-bold uppercase tracking-widest">
          <ArrowLeft className="size-4" /> Feed
        </Link>
        <span className={`inline-flex items-center gap-1 text-[10px] font-black uppercase tracking-wider px-2.5 py-1 rounded-full ${cfg.bg} ${cfg.border} ${cfg.color} border`}>
          {cfg.icon} {cfg.label}
        </span>
      </div>

      {/* Right side action buttons (TikTok style) */}
      <div className="absolute right-3 bottom-28 flex flex-col items-center gap-5 z-20">
        <button onClick={toggleMute} className="flex flex-col items-center gap-1">
          <div className="size-11 rounded-full bg-black/40 border border-white/20 flex items-center justify-center">
            {muted ? <VolumeX className="size-5 text-white" /> : <Volume2 className="size-5 text-white" />}
          </div>
          <span className="text-[10px] text-white/60">{muted ? "Unmute" : "Muted"}</span>
        </button>
        <div className="flex flex-col items-center gap-1">
          <div className="size-11 rounded-full bg-black/40 border border-white/20 flex items-center justify-center">
            <Heart className="size-5 text-white/80" />
          </div>
          <span className="text-[10px] text-white/60">{Math.round(item.highlight.score * 10)}</span>
        </div>
        <Link href={`/matches/${item.matchId}`} className="flex flex-col items-center gap-1">
          <div className="size-11 rounded-full bg-black/40 border border-white/20 flex items-center justify-center">
            <Maximize className="size-5 text-white/80" />
          </div>
          <span className="text-[10px] text-white/60">Match</span>
        </Link>
      </div>

      {/* Bottom info */}
      <div className="absolute bottom-0 left-0 right-0 px-4 pb-6 z-20 pr-20">
        {/* Commentary */}
        {item.highlight.commentary && (
          <p className="text-white text-sm font-medium leading-snug mb-3 line-clamp-3 drop-shadow-lg">
            {item.highlight.commentary}
          </p>
        )}
        <div className="flex items-center gap-3 text-xs text-white/50">
          <span className="flex items-center gap-1"><Clock className="size-3" /> {formatTime(item.highlight.startTime)}</span>
          <span>•</span>
          <span>{Math.round(item.highlight.endTime - item.highlight.startTime)}s clip</span>
          <span>•</span>
          <span className="text-primary font-bold">Score {item.highlight.score.toFixed(1)}</span>
        </div>
      </div>

      {/* Tap zones for navigation — large invisible hit areas */}
      {hasPrev && (
        <button
          onClick={onPrev}
          className="absolute top-1/4 left-0 w-12 bottom-1/4 z-10 opacity-0"
          aria-label="Previous"
        />
      )}
      {hasNext && (
        <button
          onClick={onNext}
          className="absolute top-1/4 right-0 w-12 bottom-1/4 z-10 opacity-0"
          aria-label="Next"
        />
      )}

      {/* Centre pause indicator */}
      <AnimatePresence>
        {!playing && (
          <motion.div
            initial={{ opacity: 0, scale: 0.7 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.7 }}
            className="absolute inset-0 flex items-center justify-center pointer-events-none z-10"
          >
            <div className="size-16 rounded-full bg-black/50 border border-white/20 flex items-center justify-center">
              <Play className="size-8 fill-white text-white ml-1" />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Desktop grid card ─────────────────────────────────────────────────────────
function HighlightCard({
  item, active, onActivate,
}: {
  item: FeedItem;
  active: boolean;
  onActivate: () => void;
}) {
  const vidRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(true);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const cfg = evStyle(item.highlight.eventType);

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

  const toggle = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
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
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25 }}
      onClick={onActivate}
      className={`group relative flex flex-col bg-zinc-950 rounded-xl overflow-hidden border cursor-pointer transition-all duration-300
        ${active
          ? "border-primary/50 shadow-[0_0_32px_rgba(16,185,129,0.15)] scale-[1.01]"
          : "border-zinc-800/60 hover:border-zinc-700 hover:scale-[1.005]"}`}
    >
      {/* Video */}
      <div className="relative aspect-video bg-black overflow-hidden">
        <video
          ref={vidRef}
          src={item.videoSrc}
          muted={muted}
          loop
          playsInline
          preload="metadata"
          className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
          onLoadedMetadata={() => setDuration(vidRef.current?.duration ?? 0)}
          onTimeUpdate={() => setCurrent(vidRef.current?.currentTime ?? 0)}
        />

        {/* Gradient */}
        <div className="absolute inset-0 bg-linear-to-t from-black/60 via-transparent to-transparent pointer-events-none" />

        {/* Active badge */}
        {active && (
          <div className="absolute top-2 left-2 flex items-center gap-1.5 bg-primary text-black text-[9px] font-black uppercase tracking-widest px-2 py-0.5 rounded-full">
            <span className="size-1.5 rounded-full bg-black animate-pulse" /> Playing
          </div>
        )}

        {/* Event type badge */}
        <div className={`absolute top-2 right-2 inline-flex items-center gap-1 text-[9px] font-black uppercase tracking-wider px-2 py-0.5 rounded-full border ${cfg.bg} ${cfg.border} ${cfg.color}`}>
          {cfg.icon} {cfg.label}
        </div>

        {/* Score badge */}
        <div className="absolute bottom-10 left-2 text-[10px] font-black text-primary bg-black/60 px-1.5 py-0.5 rounded">
          {item.highlight.score.toFixed(1)}
        </div>

        {/* Progress bar */}
        {active && (
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white/10">
            <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
          </div>
        )}

        {/* Play/pause overlay */}
        <div
          className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={toggle}
        >
          <div className="size-12 rounded-full bg-black/60 border border-white/20 flex items-center justify-center">
            {playing
              ? <Pause className="size-5 fill-white text-white" />
              : <Play className="size-5 fill-white text-white ml-0.5" />}
          </div>
        </div>

        {/* Bottom controls (active only) */}
        {active && (
          <div className="absolute bottom-1 left-2 right-2 flex items-center justify-between" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-1.5 text-[10px] text-white/60 font-mono tabular-nums">
              {formatTime(current)} / {formatTime(duration || clipDur)}
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => { setMuted(m => { if (vidRef.current) vidRef.current.muted = !m; return !m; }); }}
                className="p-1 rounded-full hover:bg-white/20 transition-colors"
              >
                {muted ? <VolumeX className="size-3.5 text-white/60" /> : <Volume2 className="size-3.5 text-white/60" />}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Meta */}
      <div className="p-3 space-y-2">
        {item.highlight.commentary && (
          <p className="text-xs text-zinc-300 line-clamp-2 leading-relaxed font-medium">
            {item.highlight.commentary}
          </p>
        )}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[10px] text-zinc-500">
            <Clock className="size-3" />
            <span className="font-mono">{formatTime(item.highlight.startTime)}</span>
            <span className="text-zinc-700">•</span>
            <span>{Math.round(clipDur)}s</span>
          </div>
          <Link
            href={`/matches/${item.matchId}`}
            onClick={e => e.stopPropagation()}
            className="text-[10px] text-primary/60 hover:text-primary font-bold uppercase tracking-wider transition-colors"
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
  // Mobile shorts: which card index is visible (driven by scroll)
  const [shortIdx, setShortIdx] = useState(0);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const API_BASE = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ?? "http://localhost:4000";
  const client = useMemo(() => createApiClient(API_BASE), [API_BASE]);

  useEffect(() => {
    (async () => {
      try {
        const matches: MatchSummary[] = await client.getMatches();
        const items: FeedItem[] = [];
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
            } catch { /* skip */ }
          })
        );
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

  // Scroll to a short programmatically
  const scrollToShort = useCallback((idx: number) => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const child = el.children[idx] as HTMLElement | undefined;
    child?.scrollIntoView({ behavior: "smooth", block: "start" });
    setShortIdx(idx);
  }, []);

  // Sync shortIdx with scroll position
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const onScroll = () => {
      const idx = Math.round(el.scrollTop / el.clientHeight);
      setShortIdx(idx);
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, []);

  const prev = () => setActiveIdx(i => Math.max(0, i - 1));
  const next = () => setActiveIdx(i => Math.min(filtered.length - 1, i + 1));

  // ── Mobile Shorts layout ──────────────────────────────────────────────────
  const MobileFeed = (
    <div
      ref={scrollContainerRef}
      className="h-dvh w-full overflow-y-scroll snap-y snap-mandatory bg-black"
      style={{ scrollbarWidth: "none" }}
    >
      {loading && (
        <div className="h-dvh flex items-center justify-center bg-black">
          <Loader2 className="size-8 animate-spin text-primary/60" />
        </div>
      )}
      {!loading && filtered.length === 0 && (
        <div className="h-dvh flex flex-col items-center justify-center bg-black gap-4">
          <Film className="size-12 text-white/20" />
          <p className="text-sm text-white/40 text-center px-8">
            {feed.length === 0 ? "No highlights yet — upload a match first." : "No clips for this filter."}
          </p>
          <Link href="/" className="text-xs text-primary font-bold uppercase tracking-wide">Upload →</Link>
        </div>
      )}
      {!loading && filtered.map((item, idx) => (
        <ShortCard
          key={`${item.matchId}-${item.highlight.id}`}
          item={item}
          onNext={() => scrollToShort(Math.min(idx + 1, filtered.length - 1))}
          onPrev={() => scrollToShort(Math.max(idx - 1, 0))}
          hasNext={idx < filtered.length - 1}
          hasPrev={idx > 0}
        />
      ))}

      {/* Floating short counter */}
      {filtered.length > 0 && (
        <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-50 bg-black/60 border border-white/10 px-3 py-1 rounded-full text-[10px] text-white/50 font-mono pointer-events-none">
          {shortIdx + 1} / {filtered.length}
        </div>
      )}
    </div>
  );

  // ── Desktop layout ─────────────────────────────────────────────────────────
  const DesktopFeed = (
    <div className="hidden sm:block min-h-screen bg-zinc-950 text-foreground">
      {/* Slim filter bar — sits under global Navbar */}
      <div className="sticky top-14 z-30 border-b border-white/5 bg-zinc-950/90 backdrop-blur-md px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <Film className="size-3.5 text-primary" />
            <span className="text-xs font-black text-white uppercase tracking-widest">Highlights Feed</span>
            <span className="text-xs text-zinc-600 font-mono">{filtered.length} clips</span>
          </div>
          {/* Type filters */}
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              onClick={() => { setFilterType("ALL"); setActiveIdx(0); }}
              className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full border transition-all
                ${filterType === "ALL" ? "bg-primary/15 border-primary/40 text-primary" : "border-white/10 text-zinc-400 hover:border-white/20 hover:text-white"}`}
            >
              All ({feed.length})
            </button>
            {allTypes.map(t => {
              const cfg = evStyle(t);
              return (
                <button
                  key={t}
                  onClick={() => { setFilterType(t); setActiveIdx(0); }}
                  className={`inline-flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full border transition-all
                    ${filterType === t ? `${cfg.bg} ${cfg.border} ${cfg.color}` : "border-white/10 text-zinc-400 hover:border-white/20"}`}
                >
                  {cfg.icon} {cfg.label} ({feed.filter(f => (f.highlight.eventType ?? "").toUpperCase() === t).length})
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {loading && (
          <div className="flex flex-col items-center justify-center py-32 gap-4">
            <Loader2 className="size-8 animate-spin text-primary/60" />
            <p className="text-sm text-zinc-500">Loading highlights…</p>
          </div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-32 gap-4 border border-dashed border-white/10 rounded-2xl">
            <Film className="size-12 text-white/10" />
            <p className="text-sm text-zinc-500">
              {feed.length === 0 ? "No highlights — upload and analyze a match first." : "No highlights for this filter."}
            </p>
            <Link href="/" className="text-xs text-primary hover:text-primary/80 font-bold uppercase tracking-wide">Upload a match →</Link>
          </div>
        )}

        {!loading && filtered.length > 0 && (
          <div className="space-y-6">
            {/* Stats + nav */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {filterType !== "ALL" && (
                  <button onClick={() => { setFilterType("ALL"); setActiveIdx(0); }} className="text-xs text-zinc-500 hover:text-white transition-colors underline">
                    Clear filter
                  </button>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button onClick={prev} disabled={activeIdx === 0} className="p-2 border border-white/10 rounded-lg text-zinc-400 hover:text-white hover:border-white/20 transition-all disabled:opacity-30 disabled:cursor-not-allowed">
                  <ChevronLeft className="size-4" />
                </button>
                <span className="text-xs text-zinc-500 tabular-nums font-mono min-w-12 text-center">{activeIdx + 1} / {filtered.length}</span>
                <button onClick={next} disabled={activeIdx >= filtered.length - 1} className="p-2 border border-white/10 rounded-lg text-zinc-400 hover:text-white hover:border-white/20 transition-all disabled:opacity-30 disabled:cursor-not-allowed">
                  <ChevronRight className="size-4" />
                </button>
              </div>
            </div>

            {/* Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              <AnimatePresence mode="popLayout">
                {filtered.map((item, idx) => (
                  <HighlightCard
                    key={`${item.matchId}-${item.highlight.id}`}
                    item={item}
                    active={idx === activeIdx}
                    onActivate={() => setActiveIdx(idx)}
                  />
                ))}
              </AnimatePresence>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  return (
    <>
      {/* ── Mobile: full-screen Shorts overlay (covers global Navbar + Footer) ── */}
      <div className="sm:hidden fixed inset-0 z-100 bg-black">
        {/* Filter pills — fixed top over the video */}
        {!loading && (
          <div className="absolute top-0 left-0 right-0 z-50 flex gap-1.5 px-3 pt-4 pb-2 overflow-x-auto bg-linear-to-b from-black/70 to-transparent" style={{ scrollbarWidth: "none" }}>
            <button
              onClick={() => { setFilterType("ALL"); setShortIdx(0); setTimeout(() => scrollToShort(0), 50); }}
              className={`shrink-0 text-[10px] font-black uppercase tracking-wider px-3 py-1 rounded-full border transition-all
                ${filterType === "ALL" ? "bg-primary text-black border-primary" : "bg-black/60 border-white/20 text-white/70"}`}
            >
              All
            </button>
            {allTypes.map(t => {
              const cfg = evStyle(t);
              return (
                <button
                  key={t}
                  onClick={() => { setFilterType(t); setShortIdx(0); setTimeout(() => scrollToShort(0), 50); }}
                  className={`shrink-0 inline-flex items-center gap-1 text-[10px] font-black uppercase tracking-wider px-3 py-1 rounded-full border transition-all
                    ${filterType === t ? `${cfg.bg} ${cfg.border} ${cfg.color}` : "bg-black/60 border-white/20 text-white/70"}`}
                >
                  {cfg.icon} {cfg.label}
                </button>
              );
            })}
          </div>
        )}
        {MobileFeed}
      </div>

      {/* ── Desktop: normal page flow under global Navbar ── */}
      {DesktopFeed}
    </>
  );
}
