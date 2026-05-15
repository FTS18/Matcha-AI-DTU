import React from "react";
import { PIPELINE_STAGES } from "@matcha/shared";

interface CardProgressProps {
  status: string;
  stage: string;
  progress: number;
}

export const CardProgress = ({ status, stage, progress }: CardProgressProps) => {
  const isProcessing = status === "PROCESSING" || status === "UPLOADED";
  if (!isProcessing) return null;

  const safeProgress = Math.max(0, Math.min(99, Math.round(progress)));

  return (
    <div className="mt-1.5 space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-[7px] uppercase tracking-[0.15em] text-muted-foreground/70">
          {stage
            ? (PIPELINE_STAGES[stage] ?? stage)
            : status === "UPLOADED" && safeProgress === 0
              ? "Queued"
              : "Processing"}
        </span>
        <span className="font-mono text-[9px] tabular-nums text-blue-300">{safeProgress}%</span>
      </div>
      <div className="h-1 w-full bg-white/10 overflow-hidden rounded-sm">
        <div
          className="h-full bg-linear-to-r from-blue-500 to-cyan-300 transition-all duration-500"
          style={{ width: `${safeProgress}%` }}
        />
      </div>
    </div>
  );
};
