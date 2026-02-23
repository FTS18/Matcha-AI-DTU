"use client";

import {
  useRef, useEffect, useState, useCallback, useMemo,
  type MouseEvent as RMouseEvent,
} from "react";
import {
  Play, Pause, Volume2, VolumeX, Maximize, Minimize,
  SkipForward, SkipBack, Gauge, Film, Cpu, ChevronLeft,
  X, List, Zap,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  formatTime, MatchEvent, Highlight, TrackFrame,
  EVENT_CONFIG, DEFAULT_EVENT_CONFIG,
} from "@matcha/shared";
import { BRAND_COLORS } from "@matcha/theme";

export interface VideoPlayerProps {
  src: string;
  events: MatchEvent[];
  highlights: Highlight[];
  onTimeUpdate?: (t: number) => void;
  seekFnRef?: React.MutableRefObject<(t: number) => void>;
  initialTeamColors?: number[][] | null;
  trackingData?: TrackFrame[] | null;
}

// ── Color helpers ─────────────────────────────────────────────────────────────
function colorDist(a: number[], b: number[]): number {
  return Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));
}
function kMeans2(samples: number[][]): [number[], number[]] {
  if (samples.length < 4) return [[220, 50, 50], [50, 100, 220]];
  let c1 = [...samples[0]];
  let c2 = [...samples.reduce((best, c) =>
    colorDist(c, c1) > colorDist(best, c1) ? c : best, samples[1])];
  for (let iter = 0; iter < 12; iter++) {
    const g1: number[][] = [], g2: number[][] = [];
    for (const c of samples) (colorDist(c, c1) <= colorDist(c, c2) ? g1 : g2).push(c);
    if (!g1.length || !g2.length) break;
    const mean = (g: number[][]) =>
      g[0].map((_, i) => Math.round(g.reduce((s, c) => s + c[i], 0) / g.length));
    c1 = mean(g1); c2 = mean(g2);
  }
  return [c1, c2];
}

let _offscreen: HTMLCanvasElement | null = null;
function sampleJersey(
  video: HTMLVideoElement, bx: number, by: number, bw: number, bh: number,
): [number, number, number] | null {
  try {
    if (!_offscreen) {
      _offscreen = document.createElement("canvas");
      _offscreen.width = 1; _offscreen.height = 1;
    }
    const ctx = _offscreen.getContext("2d", { willReadFrequently: true })!;
    const sw = bw * 0.60, sh = bh * 0.35;
    if (sw < 2 || sh < 2) return null;
    ctx.drawImage(video, bx + bw * 0.20, by + bh * 0.25, sw, sh, 0, 0, 1, 1);
    const d = ctx.getImageData(0, 0, 1, 1).data;
    return [d[0], d[1], d[2]];
  } catch { return null; }
}

const THEME_HEX: Record<string, string> = {
  success: BRAND_COLORS.success,
  warning: BRAND_COLORS.warning,
  error:   BRAND_COLORS.error,
  info:    BRAND_COLORS.info,
  accent:  BRAND_COLORS.primary,
  neutral: BRAND_COLORS.muted,
};
function eventColor(type: string): string {
  const cfg = EVENT_CONFIG[type] ?? DEFAULT_EVENT_CONFIG;
  return THEME_HEX[cfg.theme] ?? THEME_HEX.neutral;
}
function toRgba(rgb: number[], alpha = 0.85) {
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

// clip mode state
type ClipMode = { type: "main" } | { type: "clip"; idx: number; url: string; highlight: Highlight };

interface Detection { bbox: [number, number, number, number]; class: string; score: number; }

export function VideoPlayer({
  src, events, highlights, onTimeUpdate, seekFnRef, initialTeamColors, trackingData,
}: VideoPlayerProps) {
  const wrapRef     = useRef<HTMLDivElement>(null);
  const vidRef      = useRef<HTMLVideoElement>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const rafRef      = useRef<number>(0);
  const modelRef    = useRef<any>(null);
  const predsRef    = useRef<Detection[]>([]);
  const detectingRef= useRef(false);
  const frameIdx    = useRef(0);
  const jerseyBuf   = useRef<number[][]>([]);
  const teamCols    = useRef<[number[], number[]]>(
    initialTeamColors?.length === 2
      ? [initialTeamColors[0]!, initialTeamColors[1]!]
      : [[220, 50, 50], [50, 100, 220]]
  );
  const seenEvents  = useRef<Set<string>>(new Set());
  const toastTimer  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const trackIdx    = useRef(0);
  const controlTimer= useRef<ReturnType<typeof setTimeout> | null>(null);

  // Core state
  const [playing,      setPlaying]      = useState(false);
  const [current,      setCurrent]      = useState(0);
  const [duration,     setDuration]     = useState(0);
  const [volume,       setVolume]       = useState(1);
  const [muted,        setMuted]        = useState(false);
  const [speed,        setSpeed]        = useState(1);
  const [fullscreen,   setFullscreen]   = useState(false);
  const [showTracking, setShowTracking] = useState(true);
  const [showControls, setShowControls] = useState(true);
  const [showSpeed,    setShowSpeed]    = useState(false);
  const [showPlaylist, setShowPlaylist] = useState(false);
  const [toast,        setToast]        = useState<string | null>(null);
  const [modelState,   setModelState]   = useState<"loading" | "ready" | "error">("loading");
  const [clipMode,     setClipMode]     = useState<ClipMode>({ type: "main" });
  const [buffering,    setBuffering]    = useState(false);

  // Clips with videoUrl
  const clips = useMemo(
    () => highlights.filter(h => h.videoUrl),
    [highlights]
  );

  // Active src
  const activeSrc = clipMode.type === "clip" ? clipMode.url : src;
  const isYoutube  = activeSrc?.includes("youtube") || activeSrc?.includes("youtu.be");
  const safeSrc    = isYoutube ? undefined : activeSrc;

  // Load TF model
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // @ts-ignore
        await import("@tensorflow/tfjs");
        // @ts-ignore
        const cocoSsd = await import("@tensorflow-models/coco-ssd");
        const m = await (cocoSsd as any).load({ base: "lite_mobilenet_v2" });
        if (!cancelled) { modelRef.current = m; setModelState("ready"); }
      } catch { if (!cancelled) setModelState("error"); }
    })();
    return () => { cancelled = true; };
  }, []);

  // Switch clip
  const switchToClip = useCallback((idx: number) => {
    const h = clips[idx];
    if (!h?.videoUrl) return;
    setClipMode({ type: "clip", idx, url: h.videoUrl, highlight: h });
    setShowPlaylist(false);
    setPlaying(false);
    setCurrent(0);
    setDuration(0);
  }, [clips]);

  const switchToMain = useCallback(() => {
    setClipMode({ type: "main" });
    setPlaying(false);
    setCurrent(0);
    setDuration(0);
  }, []);

  const nextClip = useCallback(() => {
    if (clipMode.type === "clip") {
      const next = clipMode.idx + 1;
      if (next < clips.length) switchToClip(next);
      else switchToMain();
    }
  }, [clipMode, clips.length, switchToClip, switchToMain]);

  // Auto-hide controls
  const resetControlTimer = useCallback(() => {
    setShowControls(true);
    if (controlTimer.current) clearTimeout(controlTimer.current);
    controlTimer.current = setTimeout(() => {
      if (playing) setShowControls(false);
    }, 3000);
  }, [playing]);

  useEffect(() => {
    if (!playing) setShowControls(true);
  }, [playing]);

  // Detection
  const runDetection = useCallback(async () => {
    const video = vidRef.current;
    const model = modelRef.current;
    if (!video || !model || detectingRef.current) return;
    if (video.readyState < 2 || video.paused || video.ended) return;
    detectingRef.current = true;
    try {
      const preds: Detection[] = await model.detect(video);
      predsRef.current = preds;
      let added = 0;
      for (const p of preds) {
        if (p.class !== "person" || p.score < 0.45) continue;
        const [bx, by, bw, bh] = p.bbox;
        const col = sampleJersey(video, bx, by, bw, bh);
        if (col) { jerseyBuf.current.push(col); added++; }
      }
      if (added && jerseyBuf.current.length >= 40) {
        if (jerseyBuf.current.length > 300)
          jerseyBuf.current.splice(0, jerseyBuf.current.length - 300);
        teamCols.current = kMeans2(jerseyBuf.current);
      }
    } finally { detectingRef.current = false; }
  }, []);

  // Draw loop
  const drawLoop = useCallback(() => {
    const canvas = canvasRef.current;
    const video  = vidRef.current;
    if (!canvas || !video) { rafRef.current = requestAnimationFrame(drawLoop); return; }
    const ctx = canvas.getContext("2d");
    if (!ctx) { rafRef.current = requestAnimationFrame(drawLoop); return; }

    const rect = video.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = rect.width; canvas.height = rect.height;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    frameIdx.current++;
    if (frameIdx.current % 6 === 0 && showTracking && clipMode.type === "main") runDetection();

    if (showTracking && video.readyState >= 2) {
      const nW = video.videoWidth || rect.width;
      const nH = video.videoHeight || rect.height;
      const nR = nW / nH, eR = rect.width / rect.height;
      let dW: number, dH: number, oX: number, oY: number;
      if (nR > eR) { dW = rect.width; dH = dW / nR; oX = 0; oY = (rect.height - dH) / 2; }
      else { dH = rect.height; dW = dH * nR; oY = 0; oX = (rect.width - dW) / 2; }
      const sX = dW / nW, sY = dH / nH;

      // Coco-SSD overlay (only in main mode)
      if (clipMode.type === "main") {
        for (const p of predsRef.current) {
          const [bx, by, bw, bh] = p.bbox;
          const cx = oX + bx * sX, cy = oY + by * sY;
          const cw = bw * sX, ch = bh * sY;
          if (p.class === "person" && p.score >= 0.45) {
            const col = sampleJersey(video, bx, by, bw, bh);
            const [c0, c1] = teamCols.current;
            const team = col ? (colorDist(col, c0) <= colorDist(col, c1) ? 0 : 1) : 0;
            const stroke = toRgba(teamCols.current[team]!);
            ctx.strokeStyle = stroke; ctx.lineWidth = 2;
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(cx, cy, cw, ch, 3); else ctx.rect(cx, cy, cw, ch);
            ctx.stroke();
          }
          if (p.class === "sports ball" && p.score >= 0.30) {
            const bCx = oX + (bx + bw / 2) * sX, bCy = oY + (by + bh / 2) * sY;
            const r = Math.max(7, (bw * sX) / 2);
            ctx.fillStyle = BRAND_COLORS.primary;
            ctx.beginPath(); ctx.arc(bCx, bCy, r, 0, Math.PI * 2); ctx.fill();
          }
        }
      }

      // Backend tracking overlay
      if (trackingData && trackingData.length > 0 && clipMode.type === "main") {
        while (trackIdx.current < trackingData.length - 1 &&
               (trackingData[trackIdx.current]?.t ?? 0) < video.currentTime) {
          trackIdx.current++;
        }
        while (trackIdx.current > 0 &&
               (trackingData[trackIdx.current]?.t ?? 0) > video.currentTime) {
          trackIdx.current--;
        }
        const currentFrame = trackingData[trackIdx.current];
        if (currentFrame && Math.abs(currentFrame.t - video.currentTime) < 0.2) {
          for (const p of (currentFrame.p ?? [])) {
            const [nx, ny, nw, nh, , team] = p;
            const px = oX + nx * dW, py = oY + ny * dH;
            const pw = nw * dW, ph = nh * dH;
            const color = team === 0 ? "rgba(220,50,50,0.85)" : "rgba(50,100,220,0.85)";
            ctx.strokeStyle = color; ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.ellipse(px + pw / 2, py + ph, pw / 3, pw / 6, 0, 0, Math.PI * 2);
            ctx.stroke();
          }
          for (const b of (currentFrame.b ?? [])) {
            const [nx, ny, nw, nh] = b;
            const bx2 = oX + nx * dW, by2 = oY + ny * dH;
            const bCx = bx2 + (nw * dW) / 2, bCy = by2 + (nh * dH) / 2;
            ctx.fillStyle = BRAND_COLORS.primary;
            ctx.shadowBlur = 10; ctx.shadowColor = BRAND_COLORS.primary;
            ctx.beginPath(); ctx.arc(bCx, bCy, 6, 0, Math.PI * 2); ctx.fill();
            ctx.shadowBlur = 0;
          }
        }
      }
    }
    rafRef.current = requestAnimationFrame(drawLoop);
  }, [showTracking, runDetection, trackingData, clipMode.type]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(drawLoop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [drawLoop]);

  // Time update
  const handleTimeUpdate = useCallback(() => {
    const v = vidRef.current;
    if (!v) return;
    setCurrent(v.currentTime);
    onTimeUpdate?.(v.currentTime);
    // Toast for main match events
    if (clipMode.type === "main") {
      for (const ev of events) {
        if (!seenEvents.current.has(ev.id) &&
            ev.timestamp >= v.currentTime - 0.8 && ev.timestamp <= v.currentTime + 0.8) {
          seenEvents.current.add(ev.id);
          setToast(ev.commentary ?? `${ev.type} @ ${formatTime(ev.timestamp)}`);
          if (toastTimer.current) clearTimeout(toastTimer.current);
          toastTimer.current = setTimeout(() => setToast(null), 3500);
        }
      }
    }
  }, [events, onTimeUpdate, clipMode.type]);

  // Fullscreen
  useEffect(() => {
    const onFS = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", onFS);
    return () => document.removeEventListener("fullscreenchange", onFS);
  }, []);

  // Keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const v = vidRef.current;
      if (!v || (e.target as HTMLElement).tagName === "INPUT") return;
      if (e.key === " ") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
      if (e.key === "ArrowRight") v.currentTime += 5;
      if (e.key === "ArrowLeft")  v.currentTime -= 5;
      if (e.key === "ArrowUp")    { v.volume = Math.min(1, v.volume + 0.1); setVolume(v.volume); }
      if (e.key === "ArrowDown")  { v.volume = Math.max(0, v.volume - 0.1); setVolume(v.volume); }
      if (e.key === "m") { v.muted = !v.muted; setMuted(v.muted); }
      if (e.key === "f") {
        const el = wrapRef.current;
        if (!el) return;
        document.fullscreenElement ? document.exitFullscreen() : el.requestFullscreen();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const seekTo = useCallback((t: number) => {
    const v = vidRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(t, duration || 999));
  }, [duration]);

  useEffect(() => {
    if (seekFnRef) seekFnRef.current = seekTo;
  }, [seekTo, seekFnRef]);

  // Seek bar click
  const onSeekClick = (e: RMouseEvent<HTMLDivElement>) => {
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    seekTo(((e.clientX - rect.left) / rect.width) * duration);
  };

  const progressPct = duration > 0 ? (current / duration) * 100 : 0;

  return (
    <div
      ref={wrapRef}
      className="relative flex flex-col bg-black select-none focus:outline-none overflow-hidden"
      tabIndex={0}
      onMouseMove={resetControlTimer}
      onMouseLeave={() => playing && setShowControls(false)}
      onClick={e => {
        // close menus on backdrop click
        if (showSpeed) setShowSpeed(false);
        if (showPlaylist) setShowPlaylist(false);
      }}
    >
      {/* ── Video element ─────────────────────────────────────────────────── */}
      <div className="relative bg-black">
        <video
          key={activeSrc}
          ref={vidRef}
          src={safeSrc}
          crossOrigin="anonymous"
          className="w-full aspect-video object-contain bg-black cursor-pointer"
          onClick={() => { const v = vidRef.current; v?.paused ? v.play() : v?.pause(); }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => { setPlaying(false); if (clipMode.type === "clip") nextClip(); }}
          onLoadedMetadata={() => {
            setDuration(vidRef.current?.duration ?? 0);
            vidRef.current?.play().catch(() => {});
          }}
          onTimeUpdate={handleTimeUpdate}
          onWaiting={() => setBuffering(true)}
          onCanPlay={() => setBuffering(false)}
          preload="metadata"
        />

        {/* Canvas for AI overlay */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 pointer-events-none"
          style={{ width: "100%", height: "100%" }}
        />

        {/* Buffering spinner */}
        {buffering && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="size-10 rounded-full border-2 border-white/20 border-t-white animate-spin" />
          </div>
        )}

        {/* Big play button */}
        <AnimatePresence>
          {!playing && !buffering && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              className="absolute inset-0 flex items-center justify-center pointer-events-none"
            >
              <div className="size-16 sm:size-20 rounded-full bg-black/60 backdrop-blur-md flex items-center justify-center border border-white/20 shadow-2xl">
                <Play className="size-6 sm:size-8 text-white ml-1" />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Event toast */}
        <AnimatePresence>
          {toast && (
            <motion.div
              initial={{ y: -20, opacity: 0, x: "-50%" }}
              animate={{ y: 0, opacity: 1, x: "-50%" }}
              exit={{ y: -20, opacity: 0, x: "-50%" }}
              className="absolute top-4 left-1/2 z-30 max-w-xs
                         bg-black/80 backdrop-blur-lg border border-primary/40
                         text-white text-xs px-4 py-2 rounded-full text-center shadow-xl"
            >
              <span className="text-primary font-bold mr-1.5">EVENT</span>
              {toast}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Top-left badges (clip mode indicator) */}
        <div className="absolute top-3 left-3 flex items-center gap-2 z-20">
          {clipMode.type === "clip" && (
            <button
              onClick={e => { e.stopPropagation(); switchToMain(); }}
              className="flex items-center gap-1.5 bg-black/70 hover:bg-black/90 backdrop-blur-md
                         border border-white/20 text-white text-[10px] font-bold uppercase
                         tracking-wider px-3 py-1.5 rounded-full transition-all"
            >
              <ChevronLeft className="size-3" /> Match
            </button>
          )}
          {clipMode.type === "clip" && (
            <div className="flex items-center gap-1.5 bg-primary/80 backdrop-blur-md
                            text-black text-[10px] font-black uppercase tracking-wider
                            px-3 py-1.5 rounded-full">
              <Film className="size-3" />
              Clip {clipMode.idx + 1}/{clips.length}
            </div>
          )}
        </div>

        {/* Top-right AI badge + tracking toggle */}
        <div className="absolute top-3 right-3 flex flex-col items-end gap-2 z-20">
          <div className={`flex items-center gap-1.5 text-[10px] font-bold uppercase
            px-2.5 py-1.5 rounded-full backdrop-blur-md border
            ${modelState === "ready"
              ? "bg-emerald-950/70 border-emerald-500/30 text-emerald-400"
              : modelState === "loading"
              ? "bg-zinc-900/70 border-zinc-600/30 text-zinc-400 animate-pulse"
              : "bg-red-950/70 border-red-500/30 text-red-400"}`}
          >
            <Cpu className="size-3" />
            {modelState === "ready" ? "AI Live" : modelState === "loading" ? "Loading…" : "AI Off"}
          </div>

          <button
            onClick={e => { e.stopPropagation(); setShowTracking(v => !v); }}
            className="bg-black/60 hover:bg-black/80 backdrop-blur-md border border-white/10
                       text-[10px] font-semibold px-2.5 py-1.5 rounded-full transition-all
                       flex items-center gap-1.5 text-white/80"
          >
            <span className={`size-1.5 rounded-full ${showTracking ? "bg-primary animate-pulse" : "bg-zinc-600"}`} />
            Tracking {showTracking ? "ON" : "OFF"}
          </button>
        </div>

        {/* ── Controls overlay ──────────────────────────────────────────── */}
        <AnimatePresence>
          {showControls && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15 }}
              className="absolute bottom-0 left-0 right-0 z-20
                         bg-gradient-to-t from-black/95 via-black/60 to-transparent
                         px-3 sm:px-5 pb-3 pt-12"
              onClick={e => e.stopPropagation()}
            >
              {/* Seekbar */}
              <div
                className="relative h-1 bg-white/20 rounded-full mb-4 cursor-pointer group/seek"
                onClick={onSeekClick}
              >
                {/* Progress fill */}
                <div
                  className="absolute left-0 top-0 h-full bg-primary rounded-full"
                  style={{ width: `${progressPct}%` }}
                />

                {/* Thumb */}
                <div
                  className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2
                             size-3 bg-white rounded-full shadow-lg opacity-0
                             group-hover/seek:opacity-100 transition-opacity"
                  style={{ left: `${progressPct}%` }}
                />

                {/* Event markers */}
                {duration > 0 && events.map(ev => (
                  <button
                    key={ev.id}
                    title={`${ev.type} @ ${formatTime(ev.timestamp)}`}
                    onClick={e => { e.stopPropagation(); seekTo(ev.timestamp); }}
                    className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 size-2 rounded-full z-10"
                    style={{ left: `${(ev.timestamp / duration) * 100}%`, background: eventColor(ev.type) }}
                  />
                ))}
              </div>

              {/* Control row */}
              <div className="flex items-center gap-2 sm:gap-4">
                {/* Skip back */}
                <button
                  onClick={() => seekTo(current - 10)}
                  className="text-white/70 hover:text-white transition-colors p-1"
                >
                  <SkipBack className="size-4 sm:size-5" />
                </button>

                {/* Play/pause */}
                <button
                  onClick={() => { const v = vidRef.current; v?.paused ? v.play() : v?.pause(); }}
                  className="size-9 sm:size-10 rounded-full bg-white text-black flex items-center
                             justify-center hover:bg-zinc-200 transition-all shrink-0 shadow-lg"
                >
                  {playing
                    ? <Pause className="size-4 sm:size-5" />
                    : <Play  className="size-4 sm:size-5 ml-0.5" />}
                </button>

                {/* Skip forward */}
                <button
                  onClick={() => seekTo(current + 10)}
                  className="text-white/70 hover:text-white transition-colors p-1"
                >
                  <SkipForward className="size-4 sm:size-5" />
                </button>

                {/* Time */}
                <span className="text-xs font-mono text-white/70 tabular-nums ml-1 shrink-0">
                  {formatTime(current)} / {formatTime(duration)}
                </span>

                {/* Spacer */}
                <div className="flex-1" />

                {/* Next clip / skip (when in clip mode) */}
                {clipMode.type === "clip" && clips.length > 1 && (
                  <button
                    onClick={e => { e.stopPropagation(); nextClip(); }}
                    className="flex items-center gap-1 text-white/70 hover:text-white text-[10px]
                               uppercase tracking-wider font-bold transition-colors"
                  >
                    <Zap className="size-3.5" /> Next
                  </button>
                )}

                {/* Volume */}
                <div className="flex items-center gap-1.5 group/vol">
                  <button
                    onClick={e => {
                      e.stopPropagation();
                      const v = vidRef.current;
                      if (!v) return;
                      v.muted = !v.muted;
                      setMuted(v.muted);
                    }}
                    className="text-white/70 hover:text-white transition-colors"
                  >
                    {muted || volume === 0
                      ? <VolumeX className="size-4 sm:size-5" />
                      : <Volume2 className="size-4 sm:size-5" />}
                  </button>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={muted ? 0 : volume}
                    onChange={e => {
                      const v = vidRef.current;
                      if (!v) return;
                      v.volume = +e.target.value;
                      setVolume(+e.target.value);
                      if (+e.target.value > 0 && v.muted) { v.muted = false; setMuted(false); }
                    }}
                    className="w-0 group-hover/vol:w-16 transition-all duration-300 h-0.5 accent-primary cursor-pointer opacity-0 group-hover/vol:opacity-100"
                  />
                </div>

                {/* Speed */}
                <div className="relative">
                  <button
                    onClick={e => { e.stopPropagation(); setShowSpeed(v => !v); setShowPlaylist(false); }}
                    className="flex items-center gap-1 text-white/70 hover:text-white text-[10px]
                               font-bold uppercase tracking-wider transition-colors"
                  >
                    <Gauge className="size-3.5" /> {speed}x
                  </button>
                  <AnimatePresence>
                    {showSpeed && (
                      <motion.div
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 6 }}
                        className="absolute bottom-9 right-0 bg-zinc-900 border border-white/10
                                   rounded-xl overflow-hidden shadow-2xl z-40 min-w-20"
                        onClick={e => e.stopPropagation()}
                      >
                        {[0.5, 0.75, 1, 1.25, 1.5, 2].map(r => (
                          <button
                            key={r}
                            onClick={() => {
                              const v = vidRef.current;
                              if (v) v.playbackRate = r;
                              setSpeed(r); setShowSpeed(false);
                            }}
                            className={`w-full text-left px-4 py-2 text-xs font-semibold transition-colors
                              ${speed === r ? "bg-primary/20 text-primary" : "text-zinc-300 hover:bg-zinc-800"}`}
                          >
                            {r}x
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                {/* Playlist / clips toggle */}
                {clips.length > 0 && (
                  <button
                    onClick={e => { e.stopPropagation(); setShowPlaylist(v => !v); setShowSpeed(false); }}
                    className="relative text-white/70 hover:text-white transition-colors"
                    title="Highlight clips"
                  >
                    <List className="size-4 sm:size-5" />
                    <span className="absolute -top-1.5 -right-1.5 size-3.5 bg-primary text-black
                                     text-[8px] font-black rounded-full flex items-center justify-center">
                      {clips.length}
                    </span>
                  </button>
                )}

                {/* Fullscreen */}
                <button
                  onClick={e => {
                    e.stopPropagation();
                    const el = wrapRef.current;
                    if (!el) return;
                    document.fullscreenElement ? document.exitFullscreen() : el.requestFullscreen();
                  }}
                  className="text-white/70 hover:text-white transition-colors"
                >
                  {fullscreen
                    ? <Minimize className="size-4 sm:size-5" />
                    : <Maximize className="size-4 sm:size-5" />}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Clip playlist drawer ──────────────────────────────────────── */}
        <AnimatePresence>
          {showPlaylist && (
            <motion.div
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%" }}
              transition={{ type: "spring", damping: 28, stiffness: 280 }}
              className="absolute top-0 right-0 bottom-0 w-64 sm:w-72
                         bg-black/95 backdrop-blur-xl border-l border-white/10
                         flex flex-col z-30"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-4 py-3 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <Film className="size-4 text-primary" />
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Highlight Clips
                  </span>
                </div>
                <button
                  onClick={() => setShowPlaylist(false)}
                  className="text-white/50 hover:text-white transition-colors"
                >
                  <X className="size-4" />
                </button>
              </div>

              {/* Back to match */}
              <button
                onClick={() => switchToMain()}
                className={`flex items-center gap-3 px-4 py-3 border-b border-white/5
                            text-xs transition-all
                            ${clipMode.type === "main"
                              ? "bg-primary/10 text-primary font-semibold"
                              : "text-white/60 hover:bg-white/5 hover:text-white"}`}
              >
                <div className="size-6 rounded bg-white/10 flex items-center justify-center shrink-0">
                  <Play className="size-3" />
                </div>
                <div className="text-left">
                  <p className="font-semibold">Full Match</p>
                  <p className="text-[10px] text-white/40 uppercase">Main Video</p>
                </div>
                {clipMode.type === "main" && (
                  <div className="ml-auto size-1.5 rounded-full bg-primary animate-pulse" />
                )}
              </button>

              <div className="flex-1 overflow-y-auto">
                {clips.map((h, idx) => {
                  const cfg = EVENT_CONFIG[h.eventType ?? ""] ?? DEFAULT_EVENT_CONFIG;
                  const isActive = clipMode.type === "clip" && clipMode.idx === idx;
                  return (
                    <button
                      key={h.id}
                      onClick={() => switchToClip(idx)}
                      className={`w-full flex items-center gap-3 px-4 py-3 border-b border-white/5
                                  text-xs transition-all text-left
                                  ${isActive
                                    ? "bg-primary/10 text-primary"
                                    : "text-white/70 hover:bg-white/5 hover:text-white"}`}
                    >
                      <div className={`size-6 rounded flex items-center justify-center shrink-0 font-black text-[10px]
                                       ${'bg' in cfg ? (cfg as any).bg : 'bg-zinc-800'} ${'color' in cfg ? (cfg as any).color : 'text-white'}`}>
                        {idx + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold truncate">
                          {cfg.label} · {formatTime(h.startTime)}
                        </p>
                        <p className="text-[10px] text-white/40">
                          {Math.round(h.endTime - h.startTime)}s clip · Score {h.score.toFixed(1)}
                        </p>
                      </div>
                      {isActive && (
                        <div className="size-1.5 rounded-full bg-primary animate-pulse shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Bottom mini seekbar visible below video (not inside overlay) ── */}
      {/* already handled inside controls above — nothing extra needed */}
    </div>
  );
}
