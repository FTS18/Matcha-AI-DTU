"use client";

import {
  useRef, useEffect, useState, useCallback, useMemo, forwardRef, useImperativeHandle,
  type MouseEvent as RMouseEvent,
} from "react";
import {
  Play, Pause, Volume2, VolumeX, Maximize, Minimize,
  SkipForward, SkipBack, Settings, Film, Cpu, ChevronLeft,
  X, List, Zap,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  formatTime, MatchEvent, Highlight, TrackFrame,
  EVENT_CONFIG, DEFAULT_EVENT_CONFIG,
} from "@matcha/shared";
import { BRAND_COLORS } from "@matcha/theme";

export interface VideoPlayerRef {
  playClip: (index: number) => void;
  playMain: () => void;
}

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
  error: BRAND_COLORS.error,
  info: BRAND_COLORS.info,
  accent: BRAND_COLORS.primary,
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

export const VideoPlayer = forwardRef<VideoPlayerRef, VideoPlayerProps>(({
  src, events, highlights, onTimeUpdate, seekFnRef, initialTeamColors, trackingData,
}, ref) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const vidRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const modelRef = useRef<any>(null);
  const predsRef = useRef<Detection[]>([]);
  const detectingRef = useRef(false);
  const frameIdx = useRef(0);
  const jerseyBuf = useRef<number[][]>([]);
  const teamCols = useRef<[number[], number[]]>(
    initialTeamColors?.length === 2
      ? [initialTeamColors[0]!, initialTeamColors[1]!]
      : [[220, 50, 50], [50, 100, 220]]
  );
  const seenEvents = useRef<Set<string>>(new Set());
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const trackIdx = useRef(0);
  const controlTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const centralPlayTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Core state
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [fullscreen, setFullscreen] = useState(false);
  const [showTracking, setShowTracking] = useState(true);
  const [showControls, setShowControls] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [showPlaylist, setShowPlaylist] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [clipMode, setClipMode] = useState<ClipMode>({ type: "main" });
  const [buffering, setBuffering] = useState(false);
  const [centralPlayIcon, setCentralPlayIcon] = useState<"play" | "pause" | null>(null);

  // Clips with videoUrl
  const clips = useMemo(
    () => highlights.filter(h => h.videoUrl),
    [highlights]
  );

  // Active src
  const activeSrc = clipMode.type === "clip" ? clipMode.url : src;
  const isYoutube = activeSrc?.includes("youtube") || activeSrc?.includes("youtu.be");
  const safeSrc = isYoutube ? undefined : activeSrc;

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

  // Imperative Handle for controlling from outside (e.g., Highlights Tab)
  useImperativeHandle(ref, () => ({
    playClip: (idx: number) => {
      switchToClip(idx);
      // scroll to player
      wrapRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    },
    playMain: () => {
      switchToMain();
      wrapRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }), [clips, switchToClip, switchToMain]);

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
      if (playing && !showSettings && !showPlaylist) setShowControls(false);
    }, 2500);
  }, [playing, showSettings, showPlaylist]);

  useEffect(() => {
    if (!playing) setShowControls(true);
  }, [playing]);

  // Toggle Play Pause Helper
  const togglePlayPause = useCallback(() => {
    const v = vidRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
      setCentralPlayIcon("play");
    } else {
      v.pause();
      setCentralPlayIcon("pause");
    }
    if (centralPlayTimer.current) clearTimeout(centralPlayTimer.current);
    centralPlayTimer.current = setTimeout(() => setCentralPlayIcon(null), 800);
  }, []);

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
    const video = vidRef.current;
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
      // YouTube style player keeps AI simple and refined
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
            ctx.strokeStyle = stroke; ctx.lineWidth = 1.5;
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(cx, cy, cw, ch, 4); else ctx.rect(cx, cy, cw, ch);
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
            ctx.strokeStyle = color; ctx.lineWidth = 1.5;
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
      if (e.key === " ") { e.preventDefault(); togglePlayPause(); }
      if (e.key === "ArrowRight") { seekTo(current + 5); resetControlTimer(); }
      if (e.key === "ArrowLeft") { seekTo(current - 5); resetControlTimer(); }
      if (e.key === "ArrowUp") { v.volume = Math.min(1, v.volume + 0.1); setVolume(v.volume); resetControlTimer(); }
      if (e.key === "ArrowDown") { v.volume = Math.max(0, v.volume - 0.1); setVolume(v.volume); resetControlTimer(); }
      if (e.key === "m" || e.key === "M") { v.muted = !v.muted; setMuted(v.muted); resetControlTimer(); }
      if (e.key === "f" || e.key === "F") {
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
    setCurrent(v.currentTime); // Immediate UI update
  }, [duration]);

  useEffect(() => {
    if (seekFnRef) seekFnRef.current = seekTo;
  }, [seekTo, seekFnRef]);

  // Seek bar interaction
  const handleSeek = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect();
    let clientX = 0;
    if ("touches" in e) {
      clientX = e.touches[0].clientX;
    } else {
      clientX = (e as React.MouseEvent).clientX;
    }
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    seekTo(pct * duration);
  };

  const progressPct = duration > 0 ? (current / duration) * 100 : 0;

  return (
    <div
      ref={wrapRef}
      className={`relative flex flex-col bg-black select-none focus:outline-none overflow-hidden group ${fullscreen ? 'fixed inset-0 z-9999' : ''}`}
      tabIndex={0}
      onMouseMove={resetControlTimer}
      onMouseLeave={() => playing && setShowControls(false)}
      onClick={e => {
        // close menus on backdrop click
        if (showSettings) setShowSettings(false);
        if (showPlaylist) setShowPlaylist(false);
      }}
    >
      {/* ── Video element ─────────────────────────────────────────────────── */}
      <div className="relative bg-black w-full h-full flex items-center justify-center">
        <video
          key={activeSrc}
          ref={vidRef}
          src={safeSrc}
          crossOrigin="anonymous"
          className="w-full h-full max-h-screen object-contain bg-black cursor-pointer"
          onClick={(e) => { e.stopPropagation(); togglePlayPause(); }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => { setPlaying(false); if (clipMode.type === "clip") nextClip(); }}
          onLoadedMetadata={() => {
            setDuration(vidRef.current?.duration ?? 0);
            if (vidRef.current && (clipMode.type === "clip" || !playing)) {
              vidRef.current.play().catch(() => { });
            }
          }}
          onTimeUpdate={handleTimeUpdate}
          onWaiting={() => setBuffering(true)}
          onPlaying={() => setBuffering(false)}
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
            <div className="size-16 rounded-full border-4 border-white/20 border-t-white animate-spin drop-shadow-lg" />
          </div>
        )}

        {/* Central Play/Pause Animation like YT */}
        <AnimatePresence>
          {centralPlayIcon && (
            <motion.div
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.5 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="absolute inset-0 flex items-center justify-center pointer-events-none z-10"
            >
              <div className="size-20 rounded-full bg-black/40 backdrop-blur-sm flex items-center justify-center">
                {centralPlayIcon === "play" ? <Play className="size-10 text-white fill-white ml-1" /> : <Pause className="size-10 text-white fill-white" />}
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
              className="absolute top-4 left-1/2 z-30 max-w-sm
                         bg-zinc-900/90 backdrop-blur-md border border-white/10
                         text-white text-sm px-5 py-2.5 rounded-full text-center shadow-2xl tracking-wide"
            >
              <span className="text-primary font-bold mr-2 uppercase text-xs tracking-wider">Event</span>
              {toast}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Top Badges overlay */}
        <AnimatePresence>
          {showControls && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute top-0 left-0 right-0 p-4 flex justify-between items-start pointer-events-none z-20 bg-linear-to-b from-black/60 to-transparent pt-6 pb-12"
            >
              <div className="flex gap-2 pointer-events-auto">
                {clipMode.type === "clip" && (
                  <>
                    <button
                      onClick={e => { e.stopPropagation(); switchToMain(); }}
                      className="flex items-center gap-1.5 bg-black/50 hover:bg-black/80 backdrop-blur-md text-white text-[11px] font-bold uppercase tracking-wider px-3.5 py-1.5 rounded-full transition-colors"
                    >
                      <ChevronLeft className="size-3.5" /> Back to Match
                    </button>
                    <div className="flex items-center gap-1.5 bg-primary/90 text-black text-[11px] font-black uppercase tracking-wider px-3.5 py-1.5 rounded-full shadow-lg">
                      <Film className="size-3.5" /> Clip {clipMode.idx + 1} of {clips.length}
                    </div>
                  </>
                )}
              </div>
              <div className="flex gap-2 pointer-events-auto">
                <div className={`flex items-center gap-1.5 text-[11px] font-bold uppercase px-3 py-1.5 rounded-full backdrop-blur-md shadow-lg transition-colors
                  ${modelState === "ready" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" : modelState === "loading" ? "bg-zinc-800/80 text-zinc-400 animate-pulse border border-zinc-600/30" : "bg-red-500/10 text-red-400 border border-red-500/20"}`}
                >
                  <Cpu className="size-3.5" />
                  {modelState === "ready" ? "AI Live" : modelState === "loading" ? "Loading…" : "AI Off"}
                </div>
                <button
                  onClick={e => { e.stopPropagation(); setShowTracking(v => !v); }}
                  className="bg-black/50 hover:bg-black/80 backdrop-blur-md text-[11px] font-bold uppercase tracking-wider px-3 py-1.5 rounded-full transition-colors flex items-center gap-2 text-white/90 border border-white/10"
                >
                  <span className={`size-1.5 rounded-full ${showTracking ? "bg-primary shadow-[0_0_8px_rgba(255,255,255,0.8)]" : "bg-zinc-500"}`} />
                  Tracking {showTracking ? "On" : "Off"}
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Controls overlay ──────────────────────────────────────────── */}
        <AnimatePresence>
          {showControls && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2 }}
              className="absolute bottom-0 left-0 right-0 z-20 px-4 pb-2 pt-24 bg-linear-to-t from-black/90 via-black/40 to-transparent"
              onClick={e => e.stopPropagation()}
            >
              {/* YouTube-style Seekbar */}
              <div
                className="relative flex items-center group/seek cursor-pointer h-5 w-full mb-1"
                onClick={handleSeek}
                onMouseDown={(e) => {
                  const onMouseMove = (moveEv: MouseEvent) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const pct = Math.max(0, Math.min(1, (moveEv.clientX - rect.left) / rect.width));
                    seekTo(pct * duration);
                  };
                  const onMouseUp = () => {
                    document.removeEventListener("mousemove", onMouseMove);
                    document.removeEventListener("mouseup", onMouseUp);
                  };
                  document.addEventListener("mousemove", onMouseMove);
                  document.addEventListener("mouseup", onMouseUp);
                }}
              >
                {/* Bar Base */}
                <div className="absolute left-0 right-0 h-1 group-hover/seek:h-1.5 bg-white/20 transition-all duration-200" />

                {/* Progress Fill */}
                <div
                  className="absolute left-0 h-1 group-hover/seek:h-1.5 bg-primary transition-all duration-200"
                  style={{ width: `${progressPct}%` }}
                />

                {/* Thumb */}
                <div
                  className="absolute h-3 w-3 group-hover/seek:h-4 group-hover/seek:w-4 bg-primary rounded-full opacity-0 group-hover/seek:opacity-100 transition-all duration-200 shadow-sm"
                  style={{ left: `calc(${progressPct}% - 6px)` }}
                />

                {/* Event markers */}
                {duration > 0 && events.map(ev => (
                  <button
                    key={ev.id}
                    title={`${ev.type} @ ${formatTime(ev.timestamp)}`}
                    onClick={e => { e.stopPropagation(); seekTo(ev.timestamp); }}
                    className="absolute h-1.5 group-hover/seek:h-2 w-1 group-hover/seek:w-1.5 -translate-y-1/2 top-1/2 z-10 transition-all duration-200 hover:scale-150"
                    style={{ left: `${(ev.timestamp / duration) * 100}%`, background: eventColor(ev.type) }}
                  />
                ))}
              </div>

              {/* Control row */}
              <div className="flex items-center justify-between mt-1 text-white">
                <div className="flex items-center gap-1">
                  {/* Play/pause */}
                  <button
                    onClick={togglePlayPause}
                    className="p-2 rounded-full hover:bg-white/20 transition-colors"
                  >
                    {playing
                      ? <Pause className="size-6 fill-white" />
                      : <Play className="size-6 fill-white" />}
                  </button>

                  {/* Volume */}
                  <div className="flex items-center group/vol relative">
                    <button
                      onClick={e => {
                        e.stopPropagation();
                        const v = vidRef.current;
                        if (!v) return;
                        v.muted = !v.muted;
                        setMuted(v.muted);
                      }}
                      className="p-2 rounded-full hover:bg-white/20 transition-colors"
                    >
                      {muted || volume === 0
                        ? <VolumeX className="size-5" />
                        : <Volume2 className="size-5" />}
                    </button>
                    <div className="w-0 overflow-hidden group-hover/vol:w-16 transition-all duration-300 ease-in-out">
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
                        className="w-16 h-1 accent-white bg-white/20 rounded-full appearance-none cursor-pointer"
                        style={{ outline: 'none' }}
                      />
                    </div>
                  </div>

                  {/* Time container */}
                  <div className="text-[13px] font-medium opacity-90 ml-2 tabular-nums">
                    {formatTime(current)} <span className="opacity-60 mx-1">/</span> {formatTime(duration)}
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  {/* Next clip (if clip mode) */}
                  {clipMode.type === "clip" && clips.length > 1 && (
                    <button
                      onClick={e => { e.stopPropagation(); nextClip(); }}
                      className="flex items-center gap-1 text-[12px] font-bold uppercase tracking-wider px-3 py-1.5 rounded-full hover:bg-white/20 transition-colors mr-2"
                    >
                      <Zap className="size-4" /> Next Clip
                    </button>
                  )}

                  {/* Playlist / clips toggle */}
                  {clips.length > 0 && (
                    <button
                      onClick={e => { e.stopPropagation(); setShowPlaylist(v => !v); setShowSettings(false); }}
                      className="p-2 rounded-full hover:bg-white/20 transition-colors relative"
                      title="Highlight clips"
                    >
                      <List className="size-5" />
                      <span className="absolute top-1 right-1 size-3 bg-primary text-black text-[9px] font-bold rounded-full flex items-center justify-center border border-black">
                        {clips.length}
                      </span>
                    </button>
                  )}

                  {/* Settings */}
                  <div className="relative">
                    <button
                      onClick={e => { e.stopPropagation(); setShowSettings(v => !v); setShowPlaylist(false); }}
                      className="p-2 rounded-full hover:bg-white/20 transition-colors"
                      title="Settings"
                    >
                      <Settings className="size-5" />
                    </button>
                    <AnimatePresence>
                      {showSettings && (
                        <motion.div
                          initial={{ opacity: 0, y: 10, scale: 0.95 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.95 }}
                          className="absolute bottom-12 right-0 bg-zinc-900/95 backdrop-blur-md rounded-xl overflow-hidden shadow-2xl z-40 min-w-36 border border-white/10"
                          onClick={e => e.stopPropagation()}
                        >
                          <div className="px-4 py-2 text-xs font-semibold text-white/50 border-b border-white/10 uppercase tracking-wider">Playback Speed</div>
                          <div className="py-1">
                            {[0.5, 0.75, 1, 1.25, 1.5, 2].map(r => (
                              <button
                                key={r}
                                onClick={() => {
                                  const v = vidRef.current;
                                  if (v) v.playbackRate = r;
                                  setSpeed(r); setShowSettings(false);
                                }}
                                className="w-full text-left px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-white/10 flex items-center justify-between"
                              >
                                {r === 1 ? 'Normal' : `${r}x`}
                                {speed === r && <span className="text-primary text-lg leading-none">✓</span>}
                              </button>
                            ))}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  {/* Fullscreen */}
                  <button
                    onClick={e => {
                      e.stopPropagation();
                      const el = wrapRef.current;
                      if (!el) return;
                      document.fullscreenElement ? document.exitFullscreen() : el.requestFullscreen();
                    }}
                    className="p-2 rounded-full hover:bg-white/20 transition-colors"
                  >
                    {fullscreen
                      ? <Minimize className="size-5" />
                      : <Maximize className="size-5" />}
                  </button>
                </div>
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
              transition={{ type: "spring", damping: 25, stiffness: 250 }}
              className="absolute top-0 right-0 bottom-0 w-72 sm:w-80
                         bg-zinc-950/95 backdrop-blur-2xl border-l border-white/10
                         flex flex-col z-40 shadow-2xl"
              onClick={e => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <Film className="size-5 text-primary" />
                  <span className="text-sm font-bold text-white tracking-wide">
                    Highlights
                  </span>
                </div>
                <button
                  onClick={() => setShowPlaylist(false)}
                  className="p-2 rounded-full hover:bg-white/10 text-white/60 hover:text-white transition-colors"
                >
                  <X className="size-5" />
                </button>
              </div>

              {/* Back to match */}
              <button
                onClick={() => switchToMain()}
                className={`w-full flex items-center gap-3 px-5 py-4 border-b border-white/5
                            text-sm transition-colors
                            ${clipMode.type === "main"
                    ? "bg-primary/10 text-primary"
                    : "text-white/70 hover:bg-white/5 hover:text-white"}`}
              >
                <div className="size-8 rounded-full bg-white/10 flex items-center justify-center shrink-0">
                  <Play className="size-4" />
                </div>
                <div className="text-left flex-1">
                  <p className="font-semibold">Full Match Video</p>
                  <p className="text-xs opacity-60">Watch the entire game</p>
                </div>
                {clipMode.type === "main" && (
                  <div className="ml-2 size-2 rounded-full bg-primary shadow-[0_0_8px_rgba(255,255,255,0.8)]" />
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
                      className={`w-full flex items-center gap-3 px-5 py-3 border-b border-white/5
                                  transition-colors text-left group
                                  ${isActive
                          ? "bg-primary/5 text-primary"
                          : "text-white/80 hover:bg-white/5 hover:text-white"}`}
                    >
                      <div className={`size-8 rounded-full flex items-center justify-center shrink-0 font-black text-xs
                                       ${'bg' in cfg ? (cfg as any).bg : 'bg-zinc-800'} ${'color' in cfg ? (cfg as any).color : 'text-white'} group-hover:scale-105 transition-transform`}>
                        {idx + 1}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold text-sm truncate">
                          {cfg.label}
                        </p>
                        <p className="text-xs opacity-60 mt-0.5">
                          {formatTime(h.startTime)} • {Math.round(h.endTime - h.startTime)}s
                        </p>
                      </div>
                      {isActive && (
                        <div className="size-2 rounded-full bg-primary shadow-[0_0_8px_rgba(255,255,255,0.8)] shrink-0" />
                      )}
                    </button>
                  );
                })}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
});
