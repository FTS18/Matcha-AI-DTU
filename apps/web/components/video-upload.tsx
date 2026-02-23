"use client";

import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useDropzone } from "react-dropzone";
import { Upload, FileVideo, X, CheckCircle2, AlertCircle, Loader2, ArrowRight, Clock, Youtube, Scissors } from "lucide-react";
import { cn } from "@/lib/utils";
import { io, Socket } from "socket.io-client";
import { createApiClient, WsEvents, PIPELINE_STAGES, isYoutubeUrl, extractYoutubeId } from "@matcha/shared";

const ORCHESTRATOR_URL = process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ?? "http://localhost:4000";
const api = createApiClient(ORCHESTRATOR_URL);

// ── Duration preset options ─────────────────────────────────────────────────
const DURATION_PRESETS = [
  { label: "2 MIN",  secs: 120 },
  { label: "3 MIN",  secs: 180 },
  { label: "5 MIN",  secs: 300 },
  { label: "10 MIN", secs: 600 },
  { label: "FULL",   secs: 0 },   // 0 = no trim, process full video
];

function formatMMSS(totalSecs: number): string {
  const h = Math.floor(totalSecs / 3600);
  const m = Math.floor((totalSecs % 3600) / 60);
  const s = Math.floor(totalSecs % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export const VideoUpload = React.memo(function VideoUploadContent() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "success" | "error">("idle");
  const [matchId, setMatchId] = useState<string | null>(null);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [currentStage, setCurrentStage] = useState<string>("");

  // ── YouTube info state ──────────────────────────────────────────────────
  const [ytInfo, setYtInfo] = useState<{ title: string; duration: number; thumbnail: string; channel: string } | null>(null);
  const [ytLoading, setYtLoading] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<number>(300); // default 5 min
  const [sliderStart, setSliderStart] = useState(0); // seconds
  const fetchTimeout = useRef<NodeJS.Timeout | null>(null);

  const ytDuration = ytInfo?.duration ?? 0;

  // Computed end time
  const computedEnd = useMemo(() => {
    if (!ytDuration) return 0;
    if (selectedPreset === 0) return ytDuration; // "FULL"
    return Math.min(sliderStart + selectedPreset, ytDuration);
  }, [sliderStart, selectedPreset, ytDuration]);

  const clipLength = useMemo(() => computedEnd - sliderStart, [sliderStart, computedEnd]);

  // ── Auto-fetch YT info on valid URL ───────────────────────────────────
  useEffect(() => {
    if (fetchTimeout.current) clearTimeout(fetchTimeout.current);
    if (!isYoutubeUrl(youtubeUrl)) {
      setYtInfo(null);
      setSliderStart(0);
      return;
    }
    // Debounce 600ms
    fetchTimeout.current = setTimeout(async () => {
      setYtLoading(true);
      try {
        const info = await api.getYtInfo(youtubeUrl);
        setYtInfo(info);
        setSliderStart(0);
        // If video shorter than preset, switch to FULL
        if (info.duration && info.duration < selectedPreset) {
          setSelectedPreset(0);
        }
      } catch (e) {
        console.warn("Could not fetch YT info:", e);
        setYtInfo(null);
      } finally {
        setYtLoading(false);
      }
    }, 600);
    return () => { if (fetchTimeout.current) clearTimeout(fetchTimeout.current); };
  }, [youtubeUrl]);

  useEffect(() => {
    const newSocket = io(ORCHESTRATOR_URL);
    setSocket(newSocket);
    return () => { newSocket.disconnect(); };
  }, []);

  // Socket.IO live progress
  useEffect(() => {
    if (!socket || !matchId) return;
    socket.emit(WsEvents.JOIN_MATCH, matchId);

    const onProgress = (data: { progress: number; stage?: string }) => {
      if (data.progress === -1) { setStatus("error"); return; }
      setProcessingProgress(Math.min(data.progress, 99));
      if (data.stage) setCurrentStage(data.stage);
      if (data.progress >= 100) { setProcessingProgress(100); setStatus("success"); }
    };
    const onComplete = () => {
      setProcessingProgress(100);
      setStatus("success");
      setTimeout(() => { if (matchId) router.push(`/matches/${matchId}`); }, 1500);
    };

    socket.on(WsEvents.PROGRESS, onProgress);
    socket.on(WsEvents.COMPLETE, onComplete);
    return () => { socket.off(WsEvents.PROGRESS, onProgress); socket.off(WsEvents.COMPLETE, onComplete); };
  }, [socket, matchId, router]);

  // HTTP polling fallback
  useEffect(() => {
    if (!matchId || status === "success" || status === "error" || status === "idle" || status === "uploading") return;
    const poll = async () => {
      try {
        const m = await api.getMatch(matchId);
        if (!m) return;
        if (m.status === "COMPLETED") { setProcessingProgress(100); setStatus("success"); setTimeout(() => router.push(`/matches/${matchId}`), 1500); }
        else if (m.status === "FAILED") { setStatus("error"); }
        else if (typeof m.progress === "number" && m.progress > 0) { setProcessingProgress((prev) => Math.max(prev, Math.min(m.progress ?? 0, 99))); }
      } catch { }
    };
    poll();
    const iv = setInterval(poll, 3000);
    return () => clearInterval(iv);
  }, [matchId, status, router]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles?.length) { setFile(acceptedFiles[0]); setStatus("idle"); setUploadProgress(0); setProcessingProgress(0); }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "video/*": [".mp4", ".mov", ".avi", ".mkv"] },
    maxFiles: 1,
    multiple: false,
  });

  const uploadFile = async () => {
    if (!file) return;
    setUploading(true);
    try {
      setStatus("uploading");
      setUploadProgress(0);
      const data = await api.uploadVideo(file, (pct) => setUploadProgress(pct));
      setUploadProgress(100);
      setMatchId(data.id);
      setStatus("processing");
      window.dispatchEvent(new CustomEvent("matcha:refresh"));
    } catch (error) {
      console.error(error);
      setStatus("error");
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const uploadYoutube = async () => {
    if (!youtubeUrl) return;
    setUploading(true);
    setStatus("uploading");
    setUploadProgress(100);
    try {
      const startSec = sliderStart > 0 ? sliderStart : undefined;
      const endSec = selectedPreset === 0 ? undefined : computedEnd > 0 ? computedEnd : undefined;
      const data = await api.uploadYoutube(youtubeUrl, startSec, endSec);
      setMatchId(data.id);
      setStatus("processing");
      window.dispatchEvent(new CustomEvent("matcha:refresh"));
    } catch (error) {
      console.error(error);
      setStatus("error");
      setUploadProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setYoutubeUrl("");
    setYtInfo(null);
    setSliderStart(0);
    setSelectedPreset(300);
    setStatus("idle");
    setUploadProgress(0);
    setProcessingProgress(0);
    setCurrentStage("");
  };

  const isYoutube = isYoutubeUrl(youtubeUrl);

  // Slider percentage for the selected range overlay
  const rangeStartPct = ytDuration > 0 ? (sliderStart / ytDuration) * 100 : 0;
  const rangeWidthPct = ytDuration > 0 ? (clipLength / ytDuration) * 100 : 0;

  return (
    <div className="w-full flex flex-col gap-6">
      {/* ── YouTube URL Input ───────────────────────────────────────────── */}
      <div className="flex flex-col gap-3">
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="PASTE YOUTUBE URL HERE..."
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            disabled={status !== "idle" || file !== null}
            className="flex-1 bg-background border border-border px-4 py-3 font-mono text-[11px] uppercase tracking-widest text-foreground focus:outline-none focus:border-primary transition-colors disabled:opacity-50"
          />
          <button
            onClick={uploadYoutube}
            disabled={!isYoutube || status !== "idle" || file !== null || uploading}
            className="font-mono text-[10px] uppercase tracking-widest px-6 py-3 transition-all duration-200 hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary cursor-pointer bg-primary text-[#07080F] font-medium disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap flex items-center gap-2"
          >
            {uploading ? (
              <><Loader2 className="size-3 animate-spin" /> ANALYSING...</>
            ) : (
              "ANALYSE URL"
            )}
          </button>
        </div>

        {/* ── YouTube clip selector ──────────────────────────────────────── */}
        {isYoutube && status === "idle" && (
          <div className="flex flex-col gap-3 animate-fade-in border border-border/60 bg-background/60 backdrop-blur-sm p-4 rounded-sm">

            {/* Loading state */}
            {ytLoading && (
              <div className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" />
                <span className="font-mono text-[10px] uppercase tracking-widest">Fetching video info...</span>
              </div>
            )}

            {/* Video info card */}
            {ytInfo && !ytLoading && (
              <>
                <div className="flex gap-3 items-center">
                  {ytInfo.thumbnail && (
                    <div className="relative w-20 h-12 rounded-sm overflow-hidden shrink-0 border border-border/30">
                      <img src={ytInfo.thumbnail} alt="" className="object-cover w-full h-full" />
                      <div className="absolute bottom-0.5 right-0.5 bg-black/80 px-1 py-0.5 rounded-sm">
                        <span className="font-mono text-[8px] text-white">{formatMMSS(ytDuration)}</span>
                      </div>
                    </div>
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="font-mono text-[10px] text-foreground truncate leading-tight">{ytInfo.title}</p>
                    <p className="font-mono text-[8px] text-muted-foreground mt-0.5">{ytInfo.channel}</p>
                  </div>
                </div>

                {/* Duration presets */}
                <div className="flex flex-col gap-2">
                  <div className="flex items-center gap-2">
                    <Scissors className="size-3 text-muted-foreground" />
                    <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-widest">Clip duration</span>
                  </div>
                  <div className="flex gap-1.5 flex-wrap">
                    {DURATION_PRESETS.map((p) => {
                      const disabled = p.secs > 0 && ytDuration > 0 && p.secs > ytDuration;
                      const active = selectedPreset === p.secs;
                      return (
                        <button
                          key={p.label}
                          disabled={disabled}
                          onClick={() => {
                            setSelectedPreset(p.secs);
                            // If start + duration exceeds video, clamp start
                            if (p.secs > 0 && sliderStart + p.secs > ytDuration) {
                              setSliderStart(Math.max(0, ytDuration - p.secs));
                            }
                          }}
                          className={cn(
                            "font-mono text-[9px] uppercase tracking-widest px-3 py-1.5 border transition-all cursor-pointer",
                            active
                              ? "bg-primary/20 border-primary/50 text-primary"
                              : "border-border/50 text-muted-foreground hover:border-border hover:text-foreground",
                            disabled && "opacity-30 cursor-not-allowed"
                          )}
                        >
                          {p.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Range slider (only if not FULL) */}
                {selectedPreset !== 0 && ytDuration > 0 && (
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Clock className="size-3 text-muted-foreground" />
                        <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-widest">Start point</span>
                      </div>
                      <span className="font-mono text-[10px] text-primary font-bold">
                        {formatMMSS(sliderStart)} – {formatMMSS(computedEnd)}
                      </span>
                    </div>

                    {/* Custom range bar */}
                    <div className="relative h-7 select-none group">
                      {/* Track background */}
                      <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-1.5 bg-border/40 rounded-full" />
                      {/* Selected range overlay */}
                      <div
                        className="absolute top-1/2 -translate-y-1/2 h-1.5 bg-primary/50 rounded-full transition-all"
                        style={{ left: `${rangeStartPct}%`, width: `${rangeWidthPct}%` }}
                      />
                      {/* Native range input */}
                      <input
                        type="range"
                        min={0}
                        max={Math.max(0, ytDuration - selectedPreset)}
                        step={1}
                        value={sliderStart}
                        onChange={(e) => setSliderStart(Number(e.target.value))}
                        className="absolute inset-0 w-full opacity-0 cursor-pointer z-10"
                        aria-label="Start point slider"
                      />
                      {/* Thumb indicator */}
                      <div
                        className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 size-3.5 rounded-full bg-primary border-2 border-background shadow-lg pointer-events-none transition-all"
                        style={{ left: `${rangeStartPct}%` }}
                      />
                    </div>

                    {/* Time labels */}
                    <div className="flex justify-between">
                      <span className="font-mono text-[8px] text-muted-foreground">{formatMMSS(0)}</span>
                      <span className="font-mono text-[8px] text-muted-foreground">{formatMMSS(ytDuration)}</span>
                    </div>
                  </div>
                )}

                {/* Summary line */}
                <div className="flex items-center gap-2 pt-1 border-t border-border/30">
                  <Youtube className="size-3 text-red-500" />
                  <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-widest">
                    {selectedPreset === 0
                      ? `FULL VIDEO · ${formatMMSS(ytDuration)}`
                      : `${formatMMSS(clipLength)} CLIP · ${formatMMSS(sliderStart)} → ${formatMMSS(computedEnd)}`}
                  </span>
                </div>
              </>
            )}

            {/* Fallback when info not loaded yet — plain text inputs */}
            {!ytInfo && !ytLoading && (
              <div className="flex gap-4 items-center">
                <div className="flex-1 flex gap-2">
                  <input
                    type="text"
                    placeholder="START (eg. 30:00)"
                    value={sliderStart > 0 ? formatMMSS(sliderStart) : ""}
                    onChange={(e) => {
                      const parts = e.target.value.split(":").map(Number);
                      let secs = 0;
                      if (parts.length === 3) secs = parts[0] * 3600 + parts[1] * 60 + parts[2];
                      else if (parts.length === 2) secs = parts[0] * 60 + parts[1];
                      else secs = parts[0] || 0;
                      setSliderStart(Math.max(0, secs));
                    }}
                    className="w-1/2 bg-background border border-border px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-foreground focus:outline-none focus:border-primary transition-colors"
                  />
                  <input
                    type="text"
                    placeholder="END (eg. 45:00)"
                    onChange={() => {}}
                    className="w-1/2 bg-background border border-border px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-foreground focus:outline-none focus:border-primary transition-colors"
                  />
                </div>
                <span className="font-mono text-[8px] text-muted-foreground uppercase tracking-widest whitespace-nowrap">
                  LEAVE BLANK FOR FULL VIDEO
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1 h-px bg-border/50" />
        <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-widest">OR</span>
        <div className="flex-1 h-px bg-border/50" />
      </div>

      <div
        {...getRootProps()}
        className={[
          "drop-zone bracket relative p-10 transition-colors duration-200  focus:outline-none focus-visible:ring-2 focus-visible:ring-primary",
          status !== "idle" || !!youtubeUrl ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
          isDragActive ? "active" : "",
          file ? "has-file" : "",
        ].join(" ")}
        aria-label="Upload tactical Match video"
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center text-center gap-5">

          {/* Icon */}
          <div
            className={`size-14 flex items-center justify-center border transition-colors ${file ? 'border-primary bg-primary/10' : 'border-border-2 bg-muted'}`}
          >
            {status === "processing" ? (
              <svg className="animate-spin size-5.5 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 12a9 9 0 11-6.219-8.56" />
              </svg>
            ) : status === "success" ? (
              <svg className="size-5.5 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            ) : status === "error" ? (
              <svg className="size-5.5 text-destructive" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            ) : file ? (
              <svg className="size-5.5 text-primary" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <rect x="2" y="2" width="20" height="20" /><path d="M8 10l4-4 4 4M12 6v9" /><path d="M6 18h12" />
              </svg>
            ) : (
              <svg className="size-5.5 text-muted-foreground" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            )}
          </div>

          {/* Label */}
          <div>
            <p className={`font-display text-[28px] tracking-[0.05em] ${file ? 'text-primary' : 'text-foreground'}`}>
              {isDragActive ? "DROP TO ANALYSE" : file ? file.name.toUpperCase() : "DROP FOOTAGE HERE"}
            </p>
            <p className="font-mono mt-1 text-[9px] text-muted-foreground uppercase tracking-[0.12em]">
              {file
                ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · MP4 / MOV / AVI / MKV`
                : "Click to select · MP4 · MOV · AVI · MKV"}
            </p>
          </div>

          {/* Action */}
          {file && status === "idle" && (
            <div className="flex gap-2">
              <button
                onClick={(e) => { e.stopPropagation(); uploadFile(); }}
                disabled={uploading}
                className="font-mono text-[10px] uppercase tracking-widest px-7 py-2.5 transition-all duration-200 hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary cursor-pointer bg-primary text-[#07080F] font-medium flex items-center gap-2"
                aria-label="Analyze Match"
              >
                {uploading ? (
                  <>
                    <Loader2 className="size-3 animate-spin" />
                    ANALYSING...
                  </>
                ) : (
                  "▸ ANALYSE MATCH"
                )}
              </button>
              <button
                onClick={removeFile}
                className="font-mono text-[10px] uppercase tracking-widest px-3 py-2.5 border border-border-2 text-muted-foreground transition-colors duration-200 hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-border-2 cursor-pointer"
                aria-label="Remove File"
              >
                ✕
              </button>
            </div>
          )}

          {/* Upload progress */}
          {status === "uploading" && (
            <div className="w-full max-w-75">
              <div className="flex justify-between mb-2">
                <span className="font-mono text-[9px] text-primary uppercase tracking-[0.12em]">UPLOADING</span>
                <span className="font-mono text-[9px] text-primary">{uploadProgress}%</span>
              </div>
              <div className="h-0.5 bg-border overflow-hidden">
                <div className="h-full bg-primary transition-[width] duration-300 ease-out" style={{ width: `${uploadProgress}%` }} />
              </div>
            </div>
          )}

          {/* Processing progress */}
          {status === "processing" && (
            <div className="w-full max-w-75">
              <div className="flex justify-between mb-1">
                <span className="font-mono text-[9px] text-primary uppercase tracking-[0.12em] animate-blink">
                  {currentStage ? (PIPELINE_STAGES[currentStage] ?? currentStage).toUpperCase() : "ANALYSING MATCH"}
                </span>
                <span className="font-mono text-[9px] text-primary tabular-nums">{processingProgress}%</span>
              </div>
              <div className="h-1 bg-border overflow-hidden rounded-full">
                <div className="h-full bg-linear-to-r from-primary via-emerald-400 to-cyan-400 transition-[width] duration-500 ease-out rounded-full" style={{ width: `${processingProgress}%` }} />
              </div>
              {currentStage && (
                <p className="font-mono text-[8px] text-muted-foreground mt-1.5 uppercase tracking-widest animate-pulse">
                  {PIPELINE_STAGES[currentStage] ?? currentStage}
                </p>
              )}
            </div>
          )}

          {/* Success */}
          {status === "success" && (
            <div className="flex flex-col items-center gap-3">
              <p className="font-display text-[28px] text-primary tracking-[0.05em]">ANALYSIS COMPLETE</p>
              {matchId && (
                <button
                  onClick={() => router.push(`/matches/${matchId}`)}
                  className="font-mono text-[10px] uppercase tracking-widest px-7 py-2.5 transition-opacity hover:opacity-75 bg-primary text-[#07080F] font-medium"
                >
                  VIEW RESULTS ▸
                </button>
              )}
            </div>
          )}

          {/* Error */}
          {status === "error" && (
            <p className="font-mono text-[9px] text-destructive uppercase tracking-[0.12em]">
              ✕ UPLOAD FAILED — TRY AGAIN
            </p>
          )}
        </div>
      </div>
    </div>
  );
});
