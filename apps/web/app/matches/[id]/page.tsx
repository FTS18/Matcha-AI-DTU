"use client";

import { useEffect, useState, useRef, useMemo, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft, Play, Clock, Target, Shield, AlertTriangle,
  Zap, Star, BarChart3, TrendingUp, Film, Loader2,
  Trash2, Copy, Check, Trophy, Cpu, Radio,
  CheckCircle, XCircle, Pencil, X, Save
} from "lucide-react";
import { ScoreBadge, CopyButton, VideoPlayer, useMatchSocket } from "@matcha/ui";
import dynamic from "next/dynamic";

// PDFReportButton wraps both PDFDownloadLink and MatchReportPDF.
// It must be loaded dynamically with ssr:false — @react-pdf/renderer is
// ESM-only and crashes if Next.js evaluates it server-side.
const PDFReportButton = dynamic(
  () => import("@/components/PDFReportButton"),
  { ssr: false }
);

import type { MatchEvent, Highlight, EmotionScore, TrackFrame, MatchDetail } from "@matcha/shared";
import {
  getTop5Moments, countEventsByType, filterEventsByType,
  getLiveIntensity, avgConfidence, maxScore, formatTime,
  timeAgo,
  EVENT_CONFIG as SHARED_EVENT_CONFIG, DEFAULT_EVENT_CONFIG,
  STATUS_CONFIG,
} from "@matcha/shared";
import { createApiClient } from "@matcha/shared";

// Web-only props on EVENT_CONFIG (icon, bg, border) — extend the shared logic
const THEME_MAP: Record<string, { bg: string; border: string; color: string }> = {
  success: { bg: "bg-emerald-400/15", border: "border-emerald-400/40", color: "text-emerald-400" },
  warning: { bg: "bg-amber-400/15", border: "border-amber-400/40", color: "text-amber-400" },
  error: { bg: "bg-red-400/15", border: "border-red-400/40", color: "text-red-400" },
  info: { bg: "bg-blue-400/15", border: "border-blue-400/40", color: "text-blue-400" },
  accent: { bg: "bg-purple-400/15", border: "border-purple-400/40", color: "text-purple-400" },
  neutral: { bg: "bg-zinc-400/15", border: "border-zinc-400/40", color: "text-zinc-400" },
};

const EVENT_CONFIG: Record<string, { label: string; bg: string; border: string; color: string; icon: React.ReactNode }> = {
  GOAL: { ...SHARED_EVENT_CONFIG.GOAL, ...THEME_MAP[SHARED_EVENT_CONFIG.GOAL.theme], icon: <Target className="w-3.5 h-3.5" /> },
  TACKLE: { ...SHARED_EVENT_CONFIG.TACKLE, ...THEME_MAP[SHARED_EVENT_CONFIG.TACKLE.theme], icon: <Zap className="w-3.5 h-3.5" /> },
  FOUL: { ...SHARED_EVENT_CONFIG.FOUL, ...THEME_MAP[SHARED_EVENT_CONFIG.FOUL.theme], icon: <AlertTriangle className="w-3.5 h-3.5" /> },
  SAVE: { ...SHARED_EVENT_CONFIG.SAVE, ...THEME_MAP[SHARED_EVENT_CONFIG.SAVE.theme], icon: <Shield className="w-3.5 h-3.5" /> },
  Celebrate: { ...SHARED_EVENT_CONFIG.Celebrate, ...THEME_MAP[SHARED_EVENT_CONFIG.Celebrate.theme], icon: <Star className="w-3.5 h-3.5" /> },
};
const DEFAULT_EVT = { ...DEFAULT_EVENT_CONFIG, ...THEME_MAP[DEFAULT_EVENT_CONFIG.theme], icon: <Star className="w-3.5 h-3.5" /> };





/** Convert seconds → "m:ss" for editable input field */
function formatTimeInput(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** Parse "m:ss" or "mm:ss" string back to seconds, returns null on invalid */
function parseTimeInput(str: string): number | null {
  const parts = str.trim().split(":");
  if (parts.length !== 2) return null;
  const m = parseInt(parts[0]!, 10);
  const s = parseInt(parts[1]!, 10);
  if (isNaN(m) || isNaN(s) || m < 0 || s < 0 || s > 59) return null;
  return m * 60 + s;
}


function IntensityChart({ scores, duration }: { scores: EmotionScore[]; duration: number }) {
  if (!scores.length || !duration) return null;
  const W = 600, H = 60;
  const pts = scores.map(s => {
    const x = (s.timestamp / duration) * W;
    const y = H - s.motionScore * H;
    return `${x},${y}`;
  });

  return (
    <div className="w-full overflow-hidden bg-card border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp className="size-4 text-emerald-500" />
        <span className="text-xs font-semibold text-foreground uppercase tracking-wide">Match Intensity</span>
        <span className="ml-auto text-xs text-muted-foreground">{scores.length} data points</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-14" preserveAspectRatio="none">
        <defs>
          <linearGradient id="intensityGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
          </linearGradient>
        </defs>
        {pts.length > 1 && (
          <>
            <polyline
              points={[`0,${H}`, ...pts, `${W},${H}`].join(" ")}
              fill="url(#intensityGrad)" stroke="none"
            />
            <polyline
              points={pts.join(" ")}
              fill="none" stroke="#10b981" strokeWidth="1.5"
              strokeLinejoin="round" strokeLinecap="round"
            />
          </>
        )}
      </svg>
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
        <span>0:00</span><span>{formatTime(duration / 2)}</span><span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}

// â”€â”€â”€ Events Timeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function EventsTimeline({ events, duration, onSeek }: { events: MatchEvent[]; duration: number; onSeek: (t: number) => void }) {
  if (!duration) return null;
  return (
    <div className="bg-card border border-border p-4">
      <div className="flex items-center gap-2 mb-3">
        <BarChart3 className="size-4 text-emerald-500" />
        <span className="text-sm font-semibold text-foreground uppercase tracking-wide">Temporal Event Distribution</span>
        <span className="ml-auto text-xs text-muted-foreground">{events.length} data points</span>
      </div>
      <div className="relative h-8 bg-muted overflow-visible">
        {events.map((ev) => {
          const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
          const pct = (ev.timestamp / duration) * 100;
          return (
            <button
              key={ev.id}
              title={`${cfg.label} @ ${formatTime(ev.timestamp)} (score: ${ev.finalScore.toFixed(1)})`}
              onClick={() => onSeek(ev.timestamp)}
              className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full border-2 border-black cursor-pointer hover:scale-150 transition-transform z-10 ${cfg.color.replace("text-", "bg-")}`}
              style={{ left: `${pct}%` }}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground mt-1.5">
        <span>0:00</span><span>{formatTime(duration / 2)}</span><span>{formatTime(duration)}</span>
      </div>
      <div className="flex flex-wrap gap-3 mt-3">
        {Object.entries(EVENT_CONFIG).map(([k, v]) => (
          <div key={k} className="flex items-center gap-1 text-xs text-muted-foreground">
            <div className={`size-2 rounded-full ${v.color.replace("text-", "bg-")}`} />
            {v.label}
          </div>
        ))}
      </div>
    </div>
  );
}

// â”€â”€â”€ Delete Modal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function DeleteModal({ onConfirm, onCancel, loading }: { onConfirm: () => void; onCancel: () => void; loading: boolean }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="bg-card border border-border rounded-2xl p-6 max-w-sm w-full mx-4 shadow-2xl">
        <div className="flex items-center gap-3 mb-4">
          <div className="size-10 rounded-full bg-destructive/15 border border-destructive/30 flex items-center justify-center">
            <Trash2 className="size-5 text-destructive" />
          </div>
          <div>
            <p className="font-semibold text-foreground">Delete Analysis</p>
            <p className="text-xs text-muted-foreground">This cannot be undone</p>
          </div>
        </div>
        <p className="text-sm text-foreground/80 mb-6 leading-relaxed">
          All events, highlights, emotion scores, and commentary for this match will be permanently removed.
        </p>
        <div className="flex gap-3">
          <button
            onClick={onCancel}
            className="flex-1 py-2 rounded-lg border border-border text-foreground text-sm hover:bg-muted transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={loading}
            className="flex-1 py-2 rounded-lg bg-destructive hover:bg-destructive/90 disabled:opacity-50 text-destructive-foreground text-sm font-medium transition-colors flex items-center justify-center gap-2 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-destructive"
          >
            {loading ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

// â”€â”€â”€ Main page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
export default function MatchDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [match, setMatch] = useState<MatchDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeHighlight, setActiveHighlight] = useState<Highlight | null>(null);
  const [activeTab, setActiveTab] = useState<"highlights" | "events" | "analytics">("highlights");
  const [eventTypeFilter, setEventTypeFilter] = useState<string>("ALL");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [showOverlay, setShowOverlay] = useState(true);
  // Highlight accept/reject & edit state
  const [acceptedHighlights, setAcceptedHighlights] = useState<Set<string>>(new Set());
  const [rejectedHighlights, setRejectedHighlights] = useState<Set<string>>(new Set());
  const [editingHighlight, setEditingHighlight] = useState<string | null>(null);
  const [editForm, setEditForm] = useState<{
    startTime: string; endTime: string; eventType: string;
  }>({ startTime: "", endTime: "", eventType: "" });

  // seekFnRef: VideoPlayer injects its internal seekTo so parent buttons can seek
  const videoSeekRef = useRef<(t: number) => void>(() => { });

  const API_BASE = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ?? "http://localhost:4000";
  const client = useMemo(() => createApiClient(`${API_BASE}`), []);

  const { liveEvents, isConnected } = useMatchSocket({
    matchId: id as string,
    url: API_BASE,
  });

  const getAssetUrl = useCallback((url: string | null) => client.getAssetUrl(url), [client]);

  // Throttle currentTime updates — only update state every 500ms to avoid 60fps re-renders
  const lastTimeUpdateRef = useRef(0);
  const handleTimeUpdate = useCallback((t: number) => {
    const now = Date.now();
    if (now - lastTimeUpdateRef.current > 500) {
      lastTimeUpdateRef.current = now;
      setCurrentTime(t);
    }
  }, []);

  useEffect(() => {
    if (!id) return;

    const load = async () => {
      try {
        const data = await client.getMatch(id as string);
        if (data && data.id) setMatch(data);
      } catch (err) {
        console.error("Match load failed:", err);
      } finally {
        setLoading(false);
      }
    };
    load();
    const iv = setInterval(load, 5000);

    return () => {
      clearInterval(iv);
    };
  }, [id, client]);

  const seekTo = useCallback((t: number) => { videoSeekRef.current(t); }, []);

  const playHighlight = useCallback((h: Highlight) => {
    setActiveHighlight(h);
    setTimeout(() => videoSeekRef.current(h.startTime), 100);
  }, []);

  const handleDelete = useCallback(async () => {
    setDeleting(true);
    try {
      await client.deleteMatch(id);
      router.push("/");
    } catch { setDeleting(false); }
  }, [id, router, client]);

  const handleReanalyze = useCallback(async () => {
    setReanalyzing(true);
    try {
      await client.reanalyze(id);
      setMatch(prev => prev ? { ...prev, status: "PROCESSING", trackingData: null, teamColors: null } : prev);
    } catch { /* ignore */ } finally {
      setReanalyzing(false);
    }
  }, [id, client]);

  // ── Highlight accept / reject / edit handlers ──────────────────────────
  const handleAcceptHighlight = useCallback((highlightId: string) => {
    setAcceptedHighlights(prev => { const s = new Set(prev); s.add(highlightId); return s; });
    setRejectedHighlights(prev => { const s = new Set(prev); s.delete(highlightId); return s; });
  }, []);

  const handleRejectHighlight = useCallback(async (highlightId: string) => {
    try {
      await client.deleteHighlight(id, highlightId);
      setRejectedHighlights(prev => { const s = new Set(prev); s.add(highlightId); return s; });
      setMatch(prev => prev ? {
        ...prev,
        highlights: prev.highlights.filter(h => h.id !== highlightId),
      } : prev);
    } catch (err) {
      console.error("Failed to reject highlight:", err);
    }
  }, [id, client]);

  const startEditHighlight = useCallback((h: Highlight) => {
    setEditingHighlight(h.id);
    setEditForm({
      startTime: formatTimeInput(h.startTime),
      endTime: formatTimeInput(h.endTime),
      eventType: h.eventType ?? "",
    });
  }, []);

  const cancelEditHighlight = useCallback(() => {
    setEditingHighlight(null);
  }, []);

  const saveEditHighlight = useCallback(async (highlightId: string) => {
    try {
      const startSecs = parseTimeInput(editForm.startTime);
      const endSecs = parseTimeInput(editForm.endTime);
      if (startSecs === null || endSecs === null || endSecs <= startSecs) return;
      const data: { startTime: number; endTime: number; eventType?: string } = {
        startTime: startSecs,
        endTime: endSecs,
      };
      if (editForm.eventType) data.eventType = editForm.eventType;
      await client.updateHighlight(id, highlightId, data);
      setMatch(prev => {
        if (!prev) return prev;
        const updated = prev.highlights.map(h =>
          h.id === highlightId
            ? { ...h, startTime: startSecs, endTime: endSecs, eventType: editForm.eventType || h.eventType }
            : h
        );
        // Re-sort by startTime so the list order and seekbar positions stay consistent
        updated.sort((a, b) => a.startTime - b.startTime);
        return { ...prev, highlights: updated };
      });
      setEditingHighlight(null);
    } catch (err) {
      console.error("Failed to update highlight:", err);
    }
  }, [id, client, editForm]);

  // useMemo calls must be above early returns — Rules of Hooks.
  // null-safe defaults ensure they always run unconditionally.
  const events = match?.events ?? [];
  const emotionScores = match?.emotionScores ?? [];

  const byType = useMemo(() => countEventsByType(events), [events]);
  const topScore = useMemo(() => maxScore(events), [events]);
  const avgConf = useMemo(() => avgConfidence(events), [events]);
  const top5Moments = useMemo(() => getTop5Moments(events), [events]);
  const liveIntensity = useMemo(() => getLiveIntensity(emotionScores, currentTime), [emotionScores, currentTime]);
  const allEventTypes = useMemo(() => Array.from(new Set(events.map(e => e.type))), [events]);
  const filteredEvents = useMemo(() => filterEventsByType(events, eventTypeFilter), [events, eventTypeFilter]);
  const sortedLive = useMemo(() => [...liveEvents].sort((a, b) => b.timestamp - a.timestamp), [liveEvents]);


  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center text-muted-foreground">
        <Loader2 className="size-6 animate-spin mr-2" /> Loading match…
      </div>
    );
  }
  if (!match) {
    return (
      <div className="min-h-screen bg-black flex flex-col items-center justify-center text-zinc-400 gap-4">
        <Film className="w-12 h-12 opacity-30" />
        <p>Match not found.</p>
        <Link href="/" className="text-emerald-400 hover:underline text-sm">â† Back to dashboard</Link>
      </div>
    );
  }

  const duration = match.duration ?? 0;
  const goalCount = byType["GOAL"] ?? 0;
  const saveCount = byType["SAVE"] ?? 0;
  const processingProgress = Math.max(
    0,
    Math.min(
      match.status === "COMPLETED" ? 100 : 99,
      Math.round(match.progress ?? 0),
    ),
  );

  return (
    <div className="min-h-screen bg-background text-foreground">
      {showDeleteModal && (
        <DeleteModal
          onConfirm={handleDelete}
          onCancel={() => setShowDeleteModal(false)}
          loading={deleting}
        />
      )}

      {/* ══════ HEADER ══════ */}
      <nav className="border-b border-border/50 px-4 sm:px-6 py-4 bg-linear-to-b from-muted/50 to-background">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
            <div className="flex items-center gap-3">
              <Link href="/" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors duration-200 text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-lg p-2 -m-2">
                <ArrowLeft className="size-5" /> Back
              </Link>
              <div className="h-5 w-px bg-border/50" />
              <div className="flex flex-col gap-1">
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Match ID</p>
                <p className="font-mono text-sm text-foreground font-medium">{match.id.slice(0, 8)}...</p>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className={`inline-flex items-center gap-2 text-xs px-3 py-2 border rounded-lg font-semibold uppercase tracking-wide transition-all
                ${match.status === "COMPLETED" ? "text-emerald-400 bg-emerald-500/10 border-emerald-400/40"
                  : match.status === "PROCESSING" ? "text-blue-400 bg-blue-500/10 border-blue-400/40 animate-pulse"
                    : "text-zinc-400 bg-zinc-500/10 border-zinc-400/20"}`}>
                {match.status === "PROCESSING" && <div className="size-2 bg-blue-400 rounded-full animate-pulse" />}
                {match.status === "COMPLETED" && <div className="size-2 bg-emerald-400 rounded-full" />}
                {match.status}
              </span>
              {match.status === "COMPLETED" && (
                <PDFReportButton
                  data={{
                    id: match.id,
                    status: match.status,
                    duration: match.duration ?? 0,
                    summary: match.summary ?? undefined,
                    createdAt: match.createdAt,
                    events: match.events,
                    highlights: match.highlights,
                    teamColors: match.teamColors ?? undefined,
                    heatmapUrl: match.heatmapUrl ?? undefined,
                    topSpeedKmh: match.topSpeedKmh ?? undefined,
                  }}
                />
              )}
              <button
                onClick={handleReanalyze}
                disabled={reanalyzing || match.status === "PROCESSING"}
                className="flex items-center gap-2 text-xs text-foreground bg-muted hover:bg-muted/80 border border-border/50 px-3 py-2 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed font-medium cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                title="Re-run AI analysis"
                aria-label="Re-analyze match"
              >
                {reanalyzing ? <Loader2 className="size-4 animate-spin" /> : <Cpu className="size-4" />}
                <span className="hidden sm:inline">Re-analyze</span>
              </button>
              <button
                onClick={() => setShowDeleteModal(true)}
                className="flex items-center gap-2 text-xs text-zinc-500 hover:text-red-400 bg-muted hover:bg-red-500/5 border border-border/50 hover:border-red-400/40 px-3 py-2 rounded-lg transition-all duration-200 font-medium cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400"
                aria-label="Delete match analysis"
              >
                <Trash2 className="size-4" />
                <span className="hidden sm:inline">Delete</span>
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 sm:py-10 space-y-8">

        {/* ══════ KEY STATS ══════ */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
          {[
            { label: "Duration", value: formatTime(duration), sub: "match length", icon: Clock },
            { label: "Events", value: match.events.length.toString(), sub: `${(avgConf * 100).toFixed(0)}% confidence`, icon: BarChart3 },
            { label: "Highlights", value: match.highlights.length.toString(), sub: "top moments", icon: Star },
            { label: "Top Score", value: topScore.toFixed(1), sub: "max intensity", icon: Zap },
          ].map((stat) => {
            const IconComp = stat.icon;
            return (
              <div key={stat.label} className="bg-card border border-border/50 p-4 sm:p-5 rounded-xl hover:border-primary/40 transition-all duration-300 hover:bg-card/80">
                <div className="flex items-start justify-between mb-3">
                  <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">{stat.label}</p>
                  <IconComp className="size-4 text-primary/60" />
                </div>
                <p className="text-2xl sm:text-3xl font-bold text-foreground mb-1">{stat.value}</p>
                <p className="text-xs text-muted-foreground/70">{stat.sub}</p>
              </div>
            );
          })}
        </div>

        {match.summary && (
          <div className="bg-card border border-border p-5">
            <div className="flex items-center gap-2 mb-3">
              <Cpu className="size-4 text-emerald-400" />
              <span className="text-sm font-semibold text-foreground font-heading uppercase tracking-wide">Tactical Intelligence Summary</span>
              <span className="ml-auto text-[10px] text-muted-foreground border border-border-2 px-2 py-0.5 uppercase tracking-wider">AI Engine: Gemini 2.0 Flash</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">{match.summary}</p>
          </div>
        )}

        {top5Moments.length > 0 && (
          <div>
            <div className="flex items-center gap-2 mb-4">
              <Trophy className="size-4 sm:size-5 text-amber-400" />
              <span className="text-sm sm:text-base font-semibold text-foreground font-heading uppercase tracking-widest">High-Impact Sequential Analysis</span>
              <span className="text-[10px] sm:text-xs text-muted-foreground ml-auto hidden sm:inline-block">Select event for immediate tactical review</span>
            </div>

            {/* Scrollable Row (Desktop) / Vertical Stack (Mobile) */}
            <div className="flex flex-col sm:flex-row sm:overflow-x-auto pb-2 gap-3 sm:gap-4 hide-scrollbar snap-x snap-mandatory">
              {top5Moments.map((ev, i) => {
                const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
                const rank = i + 1;

                // Tiered styling based on rank
                const isGold = rank === 1;
                const isSilver = rank === 2;
                const isBronze = rank === 3;

                const rankBorder = isGold ? "border-amber-400/50 hover:border-amber-400"
                  : isSilver ? "border-zinc-300/40 hover:border-zinc-300"
                    : isBronze ? "border-amber-700/50 hover:border-amber-600"
                      : "border-border hover:border-border-2";

                const rankBg = isGold ? "bg-amber-400/5 hover:bg-amber-400/10"
                  : isSilver ? "bg-zinc-300/5 hover:bg-zinc-300/10"
                    : isBronze ? "bg-amber-700/5 hover:bg-amber-700/10"
                      : "bg-card hover:bg-muted/50";

                const rankTextColor = isGold ? "text-amber-400" : isSilver ? "text-zinc-300" : isBronze ? "text-amber-600" : "text-muted-foreground/50";

                return (
                  <button
                    key={ev.id}
                    onClick={() => seekTo(ev.timestamp)}
                    className={`group relative flex flex-col text-left border p-4 sm:p-5 transition-all duration-300 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary sm:w-64 sm:shrink-0 snap-start ${rankBorder} ${rankBg}`}
                  >
                    {/* Rank Badge Header */}
                    <div className="flex items-start justify-between w-full mb-3">
                      <div className={`flex items-center justify-center size-6 sm:size-7 rounded-sm bg-background border ${rankBorder} ${rankTextColor} font-black text-xs sm:text-sm`}>
                        {rank}
                      </div>
                      <span className={`inline-flex items-center gap-1.5 text-[10px] sm:text-xs px-2 py-0.5 border ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                        {cfg.icon} {cfg.label}
                      </span>
                    </div>

                    {/* Meta Info */}
                    <p className="font-mono text-sm text-foreground/80 mb-1 group-hover:text-foreground transition-colors">
                      {formatTime(ev.timestamp)}
                    </p>

                    {/* Context Score */}
                    <div className="flex items-center gap-3 mb-3 w-full">
                      <div className="text-xl sm:text-2xl font-display font-medium tracking-wide">
                        <ScoreBadge score={ev.finalScore} />
                      </div>
                      <div className="h-1 flex-1 bg-background/50 overflow-hidden border border-border/50">
                        <div
                          className={`h-full transition-all duration-700 ease-out ${isGold ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.5)]' : 'bg-emerald-500'}`}
                          style={{ width: `${(ev.finalScore / 10) * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Commentary snippet */}
                    {ev.commentary ? (
                      <p className="text-xs text-muted-foreground line-clamp-2 mt-auto">
                        {ev.commentary}
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground/40 italic mt-auto">
                        No telemetry text recorded for this event
                      </p>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {Object.keys(byType).length > 0 && (
          <div className="flex flex-wrap gap-2">
            {Object.entries(byType).map(([type, count]) => {
              const cfg = EVENT_CONFIG[type] ?? DEFAULT_EVT;
              return (
                <div key={type} className={`flex items-center gap-2 px-3 py-1.5 border text-sm ${cfg.bg} ${cfg.border}`}>
                  <span className={cfg.color}>{cfg.icon}</span>
                  <span className={`font-medium ${cfg.color} uppercase tracking-wide text-xs`}>{cfg.label}</span>
                  <span className="text-muted-foreground text-xs font-mono">{count}</span>
                </div>
              );
            })}
          </div>
        )}

        <div className="grid lg:grid-cols-5 gap-6">
          {/* â”€â”€ Left: Video + intensity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          <div className="lg:col-span-3 space-y-4">
            {match.uploadUrl && (
              <div className="space-y-3">
                <VideoPlayer
                  src={getAssetUrl(match.uploadUrl)}
                  events={match.events}
                  highlights={match.highlights}
                  initialTeamColors={match.teamColors}
                  trackingData={match.trackingData}
                  seekFnRef={videoSeekRef}
                  onTimeUpdate={handleTimeUpdate}
                />

                <div className="bg-muted/30 border border-border px-3 sm:px-4 py-3 flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-6">
                  <div className="flex items-center justify-between sm:justify-start gap-3 w-full sm:w-auto">
                    <div className="flex items-center gap-1.5">
                      <div className={`size-2 ${match.status === "COMPLETED" ? "bg-emerald-400" :
                        match.status === "PROCESSING" ? "bg-blue-400 animate-pulse" : "bg-muted-foreground"
                        }`} />
                      <span className="text-[10px] sm:text-xs font-bold tracking-widest text-muted-foreground uppercase">
                        {match.status === "PROCESSING" ? "Live" : "Full Time"}
                      </span>
                    </div>
                    <span className="font-mono text-xs text-muted-foreground sm:hidden">{formatTime(currentTime)}</span>
                  </div>
                  <div className="flex items-center gap-4 flex-1 w-full justify-between sm:justify-start">
                    <div className="text-center">
                      <p className="text-[9px] text-muted-foreground uppercase tracking-wide">Goals</p>
                      <p className="text-xl font-black text-foreground leading-none">{goalCount}</p>
                    </div>
                    <div className="text-muted-foreground/50 text-sm">•</div>
                    <div className="text-center">
                      <p className="text-[9px] text-muted-foreground uppercase tracking-wide">Saves</p>
                      <p className="text-xl font-black text-foreground leading-none">{saveCount}</p>
                    </div>
                    <div className="flex-1 space-y-1 ml-4">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[9px] text-muted-foreground uppercase tracking-wide">Intensity</span>
                        <span className="text-[9px] font-mono text-emerald-400">{(liveIntensity * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-1.5 bg-border overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all duration-500"
                          style={{ width: `${liveIntensity * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  <span className="font-mono text-xs text-muted-foreground hidden sm:block">{formatTime(currentTime)}</span>
                </div>
                {(match.status === "PROCESSING" || match.status === "UPLOADED") && (
                  <div className="bg-blue-500/5 border border-blue-500/25 px-3 sm:px-4 py-3">
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <span className="text-[10px] sm:text-xs uppercase tracking-widest font-bold text-blue-300">
                        {match.status === "UPLOADED" && processingProgress === 0 ? "Queued for analysis" : "Analysis progress"}
                      </span>
                      <span className="font-mono text-xs text-blue-200 tabular-nums">{processingProgress}%</span>
                    </div>
                    <div className="h-2 bg-blue-950/50 overflow-hidden rounded-sm">
                      <div
                        className="h-full bg-linear-to-r from-blue-500 via-cyan-400 to-emerald-400 transition-all duration-500"
                        style={{ width: `${processingProgress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
            {match.emotionScores.length > 0 && (
              <IntensityChart scores={match.emotionScores} duration={duration} />
            )}
            {match.events.length > 0 && (
              <EventsTimeline events={match.events} duration={duration} onSeek={seekTo} />
            )}
          </div>

          {/* â”€â”€ Right: Highlights + Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
          <div className="lg:col-span-2 space-y-4">
            {/* ── Live Detection Feed ─────────────────────────────────────
                 Shown during processing. Events appear in real-time via WS.    */}
            {(match.status === "PROCESSING" || (match.status === "UPLOADED" && liveEvents.length > 0)) && (
              <div className="bg-card border border-blue-500/30 overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-2.5 border-b border-border bg-blue-500/5">
                  <Radio className="size-3.5 text-blue-400 animate-pulse" />
                  <span className="text-xs font-bold text-blue-300 tracking-widest uppercase">Live Detection</span>
                  <span className="ml-auto text-[10px] text-muted-foreground">{liveEvents.length} events found</span>
                </div>
                <div className="max-h-52 overflow-y-auto space-y-px">
                  {liveEvents.length === 0 && (
                    <div className="flex items-center gap-2 px-4 py-3 text-xs text-muted-foreground/80">
                      <Loader2 className="size-3.5 animate-spin" /> Scanning frames…
                    </div>
                  )}
                  {sortedLive.map((ev, i) => {
                    const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
                    return (
                      <div
                        key={i}
                        role="button"
                        tabIndex={0}
                        onClick={() => seekTo(ev.timestamp)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seekTo(ev.timestamp); } }}
                        className={`w-full flex items-center gap-3 px-4 py-2 text-left hover:bg-muted transition-colors cursor-pointer focus:outline-none focus:bg-muted ${i === 0 ? 'animate-pulse' : ''}`}
                      >
                        <span className={`inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 border shrink-0 ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                          {cfg.icon} {cfg.label}
                        </span>
                        <span className="font-mono text-[10px] text-muted-foreground">{formatTime(ev.timestamp)}</span>
                        <span className={`ml-auto font-bold text-[10px] font-mono ${ev.finalScore >= 7.5 ? 'text-emerald-400' : ev.finalScore >= 5 ? 'text-amber-400' : 'text-muted-foreground'
                          }`}>{ev.finalScore?.toFixed(1)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="flex bg-muted/30 border border-border p-1 gap-1">
              {(["highlights", "events", "analytics"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 text-[10px] sm:text-sm py-2 font-medium transition-all uppercase tracking-wide cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-inset
                    ${activeTab === tab
                      ? "bg-primary text-[#07080F]"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}
                >
                  {tab === "highlights" ? `Highlights (${match.highlights.length})`
                    : tab === "events" ? (match.status === "PROCESSING" ? `Events (${liveEvents.length} live)` : `Events (${match.events.length})`)
                      : "Analytics"}
                </button>
              ))}
            </div>

            {activeTab === "highlights" && (
              <div className="space-y-3 max-h-96 overflow-y-auto pr-1">
                {match.highlightReelUrl && (
                  <div className="mb-4 p-4 bg-primary/20 border border-primary/30 relative overflow-hidden group/reel">
                    <div className="absolute top-0 right-0 p-1 bg-primary text-[8px] font-black uppercase text-background -rotate-45 translate-x-3 -translate-y-1 w-20 text-center shadow-xl">PRO REEL</div>
                    <h3 className="text-sm font-bold text-primary mb-2 flex items-center gap-2 font-heading uppercase tracking-wide">
                      <Film className="size-4" /> Broadcast Narrative Reel
                    </h3>
                    <p className="text-xs text-muted-foreground mb-3 pr-8">
                      Professional summary with synchronized AI commentary, localized voices, and smart transition physics.
                    </p>
                    <div className="flex flex-wrap gap-2">
                        <a
                        href={getAssetUrl(match.highlightReelUrl)}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 text-sm font-semibold text-background bg-primary hover:bg-primary/90 px-4 py-2 transition-all uppercase tracking-wide shadow-lg shadow-primary/20"
                        >
                        <Play className="size-4" /> Watch 16:9
                        </a>
                        <button className="inline-flex items-center gap-2 text-xs font-semibold text-primary border border-primary/40 hover:bg-primary/10 px-3 py-2 transition-all uppercase tracking-wide">
                            <Zap className="size-3" /> Gen 9:16 Vertical
                        </button>
                    </div>
                  </div>
                )}
                {!match.highlights.length && (
                  <div className="text-center text-muted-foreground/80 text-sm py-12 border border-dashed border-border-2 bg-muted/50">
                    <Film className="size-8 mx-auto mb-2 opacity-30" />
                    No highlights yet — re-upload to generate
                  </div>
                )}
                {match.highlights.map((h, i) => {
                  const cfg = EVENT_CONFIG[h.eventType ?? ""] ?? DEFAULT_EVT;
                  const isActive = activeHighlight?.id === h.id;
                  const isAccepted = acceptedHighlights.has(h.id);
                  const isEditing = editingHighlight === h.id;
                  return (
                    <div key={h.id} className={`border p-4 transition-all ${
                      isAccepted ? "border-emerald-400/50 bg-emerald-400/5" :
                      isActive ? "border-primary/50 bg-primary/5" : "border-border bg-card hover:border-border-2"
                    }`}>
                      {/* Header row: number + event badge + accept/reject + score */}
                      <div className="flex items-start justify-between gap-2 mb-2">
                        <div className="flex items-center gap-2">
                          <span className="size-6 bg-muted flex items-center justify-center text-xs text-muted-foreground font-bold shrink-0">{i + 1}</span>
                          {h.eventType && (
                            <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 border ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                              {cfg.icon} {cfg.label}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-1.5">
                          {/* Accept */}
                          <button
                            onClick={() => handleAcceptHighlight(h.id)}
                            title="Accept highlight"
                            className={`p-1 rounded transition-all cursor-pointer focus:outline-none ${
                              isAccepted
                                ? "text-emerald-400 bg-emerald-400/15"
                                : "text-muted-foreground hover:text-emerald-400 hover:bg-emerald-400/10"
                            }`}
                          >
                            <CheckCircle className="size-4" />
                          </button>
                          {/* Reject */}
                          <button
                            onClick={() => handleRejectHighlight(h.id)}
                            title="Reject & remove highlight"
                            className="p-1 rounded text-muted-foreground hover:text-red-400 hover:bg-red-400/10 transition-all cursor-pointer focus:outline-none"
                          >
                            <XCircle className="size-4" />
                          </button>
                          {/* Edit */}
                          <button
                            onClick={() => isEditing ? cancelEditHighlight() : startEditHighlight(h)}
                            title="Edit timestamps & event type"
                            className={`p-1 rounded transition-all cursor-pointer focus:outline-none ${
                              isEditing
                                ? "text-amber-400 bg-amber-400/15"
                                : "text-muted-foreground hover:text-amber-400 hover:bg-amber-400/10"
                            }`}
                          >
                            {isEditing ? <X className="size-4" /> : <Pencil className="size-4" />}
                          </button>
                          <ScoreBadge score={h.score} />
                        </div>
                      </div>

                      {isAccepted && (
                        <div className="flex items-center gap-1.5 text-[10px] text-emerald-400 font-semibold uppercase tracking-widest mb-2">
                          <CheckCircle className="size-3" /> Accepted
                        </div>
                      )}

                      <div className="h-1 bg-border mb-3 overflow-hidden">
                        <div
                          className="h-full bg-emerald-500 transition-all"
                          style={{ width: `${(h.score / 10) * 100}%` }}
                        />
                      </div>

                      {/* Editable timestamp / event type form */}
                      {isEditing ? (
                        <div className="space-y-2 mb-3 p-3 bg-muted/50 border border-border rounded">
                          <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Edit Timestamps & Type</p>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <label className="text-[10px] text-muted-foreground block mb-0.5">Start (m:ss)</label>
                              <input
                                type="text"
                                value={editForm.startTime}
                                onChange={(e) => setEditForm(f => ({ ...f, startTime: e.target.value }))}
                                placeholder="0:00"
                                className="w-full text-xs font-mono bg-background border border-border px-2 py-1.5 text-foreground rounded focus:outline-none focus:border-primary"
                              />
                            </div>
                            <div>
                              <label className="text-[10px] text-muted-foreground block mb-0.5">End (m:ss)</label>
                              <input
                                type="text"
                                value={editForm.endTime}
                                onChange={(e) => setEditForm(f => ({ ...f, endTime: e.target.value }))}
                                placeholder="0:30"
                                className="w-full text-xs font-mono bg-background border border-border px-2 py-1.5 text-foreground rounded focus:outline-none focus:border-primary"
                              />
                            </div>
                          </div>
                          <div>
                            <label className="text-[10px] text-muted-foreground block mb-0.5">Event Type</label>
                            <select
                              value={editForm.eventType}
                              onChange={(e) => setEditForm(f => ({ ...f, eventType: e.target.value }))}
                              className="w-full text-xs bg-background border border-border px-2 py-1.5 text-foreground rounded focus:outline-none focus:border-primary cursor-pointer"
                            >
                              <option value="">— select —</option>
                              <option value="GOAL">Goal / Score</option>
                              <option value="TACKLE">Tackle</option>
                              <option value="FOUL">Foul</option>
                              <option value="SAVE">Save</option>
                              <option value="Celebrate">Celebration</option>
                            </select>
                          </div>
                          <button
                            onClick={() => saveEditHighlight(h.id)}
                            className="flex items-center gap-1.5 text-xs font-medium text-emerald-400 hover:text-emerald-300 border border-emerald-400/30 hover:border-emerald-400 px-3 py-1.5 transition-all bg-emerald-400/5 hover:bg-emerald-400/10 uppercase tracking-wide cursor-pointer focus:outline-none w-full justify-center mt-1"
                          >
                            <Save className="size-3.5" /> Save Changes
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-3 text-xs text-muted-foreground mb-2">
                          <Clock className="size-3" />
                          <span className="font-mono">{formatTime(h.startTime)} → {formatTime(h.endTime)}</span>
                          <span className="text-muted-foreground/50">·</span>
                          <span>{Math.round(h.endTime - h.startTime)}s</span>
                        </div>
                      )}

                      {h.commentary && (
                        <div className="flex items-start gap-1.5 mb-3">
                          <p className="text-xs text-muted-foreground italic leading-relaxed flex-1">
                            "{h.commentary}"
                          </p>
                          <CopyButton text={h.commentary} />
                        </div>
                      )}

                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => playHighlight(h)}
                          className="flex items-center gap-1.5 text-xs font-medium text-primary hover:text-primary/80 border border-primary/30 hover:border-primary px-3 py-1.5 transition-all bg-primary/5 hover:bg-primary/10 uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset"
                        >
                          <Play className="size-3.5" /> Play Clip
                        </button>
                        {h.videoUrl && (
                          <a
                            href={getAssetUrl(h.videoUrl)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-1.5 text-xs font-medium text-blue-400 hover:text-blue-300 border border-blue-500/30 hover:border-blue-500 px-3 py-1.5 transition-all bg-blue-500/5 hover:bg-blue-500/10 uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset"
                          >
                            <Film className="size-3.5" /> View Generated Clip
                          </a>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {activeTab === "events" && (
              <div className="space-y-3">
                {allEventTypes.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      onClick={() => setEventTypeFilter("ALL")}
                      className={`text-xs px-3 py-1.5 border transition-all uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset ${eventTypeFilter === "ALL"
                        ? "bg-muted-foreground/20 border-border-2 text-foreground"
                        : "bg-transparent border-border text-muted-foreground hover:border-border-2 hover:text-foreground"
                        }`}
                    >
                      All ({match.events.length})
                    </button>
                    {allEventTypes.map((type) => {
                      const cfg = EVENT_CONFIG[type] ?? DEFAULT_EVT;
                      return (
                        <button
                          key={type}
                          onClick={() => setEventTypeFilter(type)}
                          className={`inline-flex items-center gap-1 text-xs px-3 py-1.5 border transition-all uppercase tracking-wide cursor-pointer focus:outline-none focus:ring-2 focus:ring-primary focus:ring-inset ${eventTypeFilter === type
                            ? `${cfg.bg} ${cfg.border} ${cfg.color}`
                            : "bg-transparent border-border text-muted-foreground hover:border-border-2 hover:text-foreground"
                            }`}
                        >
                          {cfg.icon} {cfg.label} ({byType[type] ?? 0})
                        </button>
                      );
                    })}
                  </div>
                )}

                <div className="space-y-1.5 max-h-80 overflow-y-auto pr-1">
                  {!filteredEvents.length && (
                    <div className="text-center text-muted-foreground/80 text-sm py-12 border border-dashed border-border-2 bg-muted/50">
                      No events detected
                    </div>
                  )}
                  <div className="relative pt-2 pl-4 border-l border-zinc-800 space-y-6">
                    {filteredEvents.map((ev, idx) => {
                      const cfg = EVENT_CONFIG[ev.type] ?? DEFAULT_EVT;
                      const isLast = idx === filteredEvents.length - 1;
                      return (
                        <div
                          key={ev.id}
                          role="button"
                          onClick={() => seekTo(ev.timestamp)}
                          className="group relative flex flex-col gap-2 cursor-pointer outline-none"
                        >
                          {/* Dot on line */}
                          <div className="absolute -left-5 top-1.5 size-3 bg-zinc-900 border border-zinc-700 rounded-full group-hover:border-primary transition-colors" />
                          
                          <div className="flex items-center gap-3">
                            <span className="text-[10px] font-mono text-primary font-bold">{formatTime(ev.timestamp)}</span>
                            <span className={`text-[10px] font-black uppercase tracking-widest px-1.5 py-0.5 border ${cfg.bg} ${cfg.border} ${cfg.color}`}>
                              {cfg.label}
                            </span>
                            <ScoreBadge score={ev.finalScore} />
                          </div>
                          
                          {ev.commentary && (
                            <div className="pl-0 border-l-0">
                                <p className="text-sm text-muted-foreground leading-relaxed group-hover:text-foreground transition-colors">
                                    {ev.commentary}
                                </p>
                            </div>
                          )}
                          {!isLast && <div className="h-px w-full bg-linear-to-r from-zinc-800/50 to-transparent mt-2" />}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
            {activeTab === "analytics" && (
              <div className="space-y-4">
                {/* Ball Speed Stat */}
                {match.topSpeedKmh && match.topSpeedKmh > 0 ? (
                  <div className="bg-card border border-border p-5">
                    <div className="flex items-center gap-2 mb-4">
                      <Zap className="size-4 text-amber-400" />
                      <span className="text-sm font-semibold text-foreground font-heading uppercase tracking-wide">Ball Speed</span>
                      <span className="ml-auto text-[10px] text-muted-foreground border border-border px-2 py-0.5 uppercase tracking-wider">Est. via YOLO Tracking</span>
                    </div>
                    <div className="flex items-end gap-3">
                      <span className="font-display text-5xl text-amber-400" style={{ textShadow: '0 0 20px rgba(251,191,36,0.4)' }}>
                        {(match as any).topSpeedKmh.toFixed(1)}
                      </span>
                      <span className="font-mono text-sm text-muted-foreground mb-1 uppercase tracking-widest">KM / H</span>
                    </div>
                    <p className="text-xs text-muted-foreground/60 mt-2">Top ball speed estimated from consecutive YOLO detections across the match footage</p>
                  </div>
                ) : (
                  <div className="bg-card border border-dashed border-border/60 p-5 text-center">
                    <Zap className="size-6 text-muted-foreground/30 mx-auto mb-2" />
                    <p className="text-xs text-muted-foreground">Ball speed data unavailable — re-analyze to generate</p>
                  </div>
                )}

                {/* Team Colors */}
                {match.teamColors && Array.isArray(match.teamColors) && (match.teamColors as number[][]).length >= 2 && (
                  <div className="bg-card border border-border p-5">
                    <div className="flex items-center gap-2 mb-4">
                      <BarChart3 className="size-4 text-primary" />
                      <span className="text-sm font-semibold text-foreground font-heading uppercase tracking-wide">Detected Team Colors</span>
                    </div>
                    <div className="flex gap-4">
                      {(match.teamColors as number[][]).slice(0, 2).map((color, idx) => {
                        const [r, g, b] = color;
                        const hex = `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
                        return (
                          <div key={idx} className="flex items-center gap-3">
                            <div
                              className="size-10 border border-border/60 shadow-lg"
                              style={{ backgroundColor: hex, boxShadow: `0 0 12px ${hex}60` }}
                            />
                            <div>
                              <p className="font-mono text-xs text-foreground uppercase tracking-widest">Team {String.fromCharCode(65 + idx)}</p>
                              <p className="font-mono text-[10px] text-muted-foreground">{hex.toUpperCase()}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Player Heatmap */}
                {(match as any).heatmapUrl ? (
                  <div className="bg-card border border-border p-5">
                    <div className="flex items-center gap-2 mb-4">
                      <TrendingUp className="size-4 text-primary" />
                      <span className="text-sm font-semibold text-foreground font-heading uppercase tracking-wide">Player Density Heatmap</span>
                      <span className="ml-auto text-[10px] text-muted-foreground">Full match coverage</span>
                    </div>
                    <div className="relative w-full overflow-hidden border border-border/50">
                      <img
                        src={getAssetUrl((match as any).heatmapUrl)}
                        alt="Player heatmap"
                        className="w-full h-auto object-contain"
                        loading="lazy"
                      />
                    </div>
                    <p className="text-xs text-muted-foreground/60 mt-2">Highlights where each team concentrated their play. Green = Team A, Red = Team B, Yellow = ball trail.</p>
                  </div>
                ) : (
                  <div className="bg-card border border-dashed border-border/60 p-8 text-center">
                    <TrendingUp className="size-8 text-muted-foreground/30 mx-auto mb-3" />
                    <p className="text-sm text-muted-foreground">Heatmap not generated yet</p>
                    <p className="text-[10px] text-muted-foreground/60 mt-1">Re-analyze the match to generate a player density heatmap</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

