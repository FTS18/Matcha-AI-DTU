import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Scissors, PlayCircle } from "lucide-react";
import { MiniHeatmap } from "./mini-heatmap";
import { formatTime, timeAgo } from "@matcha/shared";
import { CardStats } from "./card-stats";
import { CardActions } from "./card-actions";
import { CardProgress } from "./card-progress";

interface MatchCardProps {
  match: any;
  statusConfig: any;
  progress: number;
  stage: string;
  isConfirming: boolean;
  isDeleting: boolean;
  reanalyzingId: string | null;
  onConfirmDelete: (id: string | null) => void;
  onDelete: (id: string) => void;
  onReanalyze: (id: string) => void;
  apiBase: string;
}

export const MatchCard = ({
  match: m,
  statusConfig,
  progress,
  stage,
  isConfirming,
  isDeleting,
  reanalyzingId,
  onConfirmDelete,
  onDelete,
  onReanalyze,
  apiBase,
}: MatchCardProps) => {
  const cfg = statusConfig[m.status] ?? statusConfig.UPLOADED;
  const isProcessing = m.status === "PROCESSING" || m.status === "UPLOADED";
  const safeProgress = Math.max(0, Math.min(99, Math.round(progress)));

  const accentColor =
    m.status === "COMPLETED"
      ? "var(--green)"
      : m.status === "PROCESSING"
        ? "oklch(60% 0.15 250)"
        : m.status === "FAILED"
          ? "var(--red)"
          : "var(--amber-dim)";

  const formattedDate = new Date(m.createdAt).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });

  const formattedTime = new Date(m.createdAt).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });

  const thumbnailUrl = m.thumbnailUrl || m.heatmapUrl;
  const fullThumbnailUrl = thumbnailUrl?.startsWith("http")
    ? thumbnailUrl
    : thumbnailUrl
      ? `${apiBase}${thumbnailUrl}`
      : null;

  return (
    <motion.div
      layout
      variants={{
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0 },
      }}
      exit={{
        opacity: 0,
        scale: 0.98,
        filter: "brightness(0.5)",
        transition: { duration: 0.2 },
      }}
      className="card relative group bg-card/40 backdrop-blur-md border border-white/5 transition-all duration-300 hover:bg-card/60 hover:border-white/10 overflow-hidden"
    >
      {/* Left accent bar (desktop only) */}
      <div
        className="hidden lg:block absolute left-0 top-0 bottom-0 w-1 transition-all duration-500 group-hover:w-1.5"
        style={{
          backgroundColor: accentColor,
          opacity: 0.8,
          boxShadow: `4px 0 20px -4px ${accentColor}`,
        }}
      />

      {/* ════ MOBILE LAYOUT ════ */}
      <div className="lg:hidden">
        <Link href={`/matches/${m.id}`} className="block relative w-full h-40 overflow-hidden bg-black/70">
          {fullThumbnailUrl ? (
            <img
              src={fullThumbnailUrl}
              alt="Match preview"
              className={`w-full h-full transition-all duration-700 group-hover:scale-105 ${m.thumbnailUrl ? "object-cover saturate-75 group-hover:saturate-100" : "object-contain p-6 opacity-40"}`}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <PlayCircle className="size-14 text-white/10" />
            </div>
          )}
          <div className="absolute inset-0 bg-linear-to-t from-black/90 via-black/20 to-transparent" />
          <div
            className="absolute inset-0"
            style={{
              background: `linear-gradient(135deg, ${accentColor}18 0%, transparent 60%)`,
            }}
          />
          <div className="absolute top-3 left-3">
            <div
              className={`flex items-center gap-1 px-2 py-1 border backdrop-blur-md font-mono text-[9px] uppercase tracking-widest font-bold bg-black/60 ${cfg.color} ${m.status === "PROCESSING" ? "animate-pulse" : ""}`}
            >
              {cfg.icon}&nbsp;{cfg.label}
            </div>
          </div>
          <div className="absolute top-3 right-3">
            <span className="font-mono text-[8px] text-white/40 uppercase tracking-widest bg-black/50 backdrop-blur-sm px-2 py-1 border border-white/5">
              {timeAgo(m.createdAt)}
            </span>
          </div>
          <div className="absolute bottom-0 left-0 right-0 px-3 pb-3 pt-8">
            <h4 className="font-display text-sm tracking-[0.05em] text-white font-semibold truncate">
              {formattedDate} — Analysis
            </h4>
            <p className="font-mono text-[8px] text-white/30 uppercase tracking-widest mt-0.5">
              {formattedTime} · {m.id.split("-")[0]}
            </p>
          </div>
        </Link>

        {isProcessing && (
          <div className="px-4 py-2.5 bg-blue-500/5 border-b border-blue-500/15">
            <CardProgress status={m.status} stage={stage} progress={progress} />
          </div>
        )}

        <div className="flex items-stretch divide-x divide-white/5 bg-black/30 border-t border-white/5">
          <Link href={`/matches/${m.id}`} className="flex-1 flex items-center justify-around px-2 py-3">
            <div className="flex flex-col items-center gap-0.5">
              <span className="font-mono text-[7px] text-white/25 uppercase tracking-widest">Duration</span>
              <span className="font-mono text-xs text-white/60 tabular-nums">
                {m.duration ? formatTime(m.duration) : "--:--"}
              </span>
            </div>
            <div className="w-px h-6 bg-white/10" />
            <div className="flex flex-col items-center gap-0.5">
              <span className="font-mono text-[7px] text-white/25 uppercase tracking-widest">Events</span>
              <span className="font-display text-sm text-accent">
                {m.status === "COMPLETED" ? m._count.events : "--"}
              </span>
            </div>
            <div className="w-px h-6 bg-white/10" />
            <div className="flex flex-col items-center gap-0.5">
              <span className="font-mono text-[7px] text-white/25 uppercase tracking-widest">Clips</span>
              <span className="font-display text-sm text-primary">
                {m.status === "COMPLETED" ? m._count.highlights : "--"}
              </span>
            </div>
          </Link>
          <div className="flex items-stretch divide-x divide-white/5">
            <Link
              href={`/matches/${m.id}#highlights`}
              className="flex items-center justify-center w-11 text-accent/60 hover:text-accent hover:bg-accent/10 transition-colors"
              title="Highlights"
            >
              <Scissors className="size-4" />
            </Link>
            <div className="flex items-center justify-center px-2">
              <CardActions
                matchId={m.id}
                status={m.status}
                isConfirming={isConfirming}
                isDeleting={isDeleting}
                reanalyzingId={reanalyzingId}
                onConfirmDelete={onConfirmDelete}
                onDelete={onDelete}
                onReanalyze={onReanalyze}
              />
            </div>
          </div>
        </div>
      </div>

      {/* ════ DESKTOP LAYOUT ════ */}
      <div className="hidden lg:flex lg:h-18 items-stretch relative overflow-hidden">
        <div className="flex flex-1 items-stretch">
          <Link
            href={`/matches/${m.id}`}
            className="w-30 h-full shrink-0 relative overflow-hidden group/thumb border-r border-white/10 bg-black/40"
          >
            {m.thumbnailUrl ? (
              <img
                src={fullThumbnailUrl!}
                alt="Preview"
                className="w-full h-full object-cover opacity-60 group-hover/thumb:opacity-100 transition-all duration-700 scale-110 group-hover/thumb:scale-100 saturate-50 group-hover/thumb:saturate-100"
              />
            ) : (
              <div className="w-full h-full p-2">
                <MiniHeatmap matches={[]} />
              </div>
            )}
            <div className="absolute inset-0 bg-linear-to-r from-black/60 via-transparent to-transparent opacity-60" />
          </Link>
          <Link
            href={`/matches/${m.id}`}
            className="flex-1 flex flex-col justify-center px-6 py-2 min-w-0 focus:outline-none group/id"
          >
            <div className="flex items-center gap-2 mb-1">
              <div
                className={`px-1.5 py-0.5 border font-mono text-[7px] uppercase tracking-widest font-bold shrink-0 ${cfg.color} ${m.status === "PROCESSING" ? "animate-pulse" : ""}`}
              >
                {cfg.label}
              </div>
              <h4 className="font-display text-base tracking-[0.05em] text-foreground group-hover/id:text-white transition-colors truncate">
                {formattedDate} — Analysis
              </h4>
            </div>
            <p className="font-mono text-[9px] text-muted-foreground/40 uppercase tracking-widest truncate">
              {formattedTime} • ID: {m.id.split("-")[0]}
            </p>
            <CardProgress status={m.status} stage={stage} progress={progress} />
          </Link>
        </div>
        <div className="flex items-stretch border-l border-white/5">
          <CardStats
            duration={m.duration}
            eventCount={m._count.events}
            highlightCount={m._count.highlights}
            status={m.status}
          />
          <div className="flex items-center justify-center w-30 border-l border-white/5">
            <Link
              href={`/matches/${m.id}#highlights`}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-accent/5 hover:bg-accent/15 border border-accent/20 hover:border-accent/40 text-accent transition-all rounded-sm"
            >
              <Scissors className="size-3" />
              <span className="font-mono text-[8px] uppercase tracking-widest font-bold">Highlights</span>
            </Link>
          </div>
          <div className="flex items-center justify-center w-30 border-l border-white/10 bg-white/2">
            <CardActions
              matchId={m.id}
              status={m.status}
              isConfirming={isConfirming}
              isDeleting={isDeleting}
              reanalyzingId={reanalyzingId}
              onConfirmDelete={onConfirmDelete}
              onDelete={onDelete}
              onReanalyze={onReanalyze}
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
};
