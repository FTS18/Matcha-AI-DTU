import React from "react";
import { AlertTriangle } from "lucide-react";

interface EmptyStateProps {
  hasMatches: boolean;
}

export const EmptyState = ({ hasMatches }: EmptyStateProps) => (
  <div className="flex flex-col items-center justify-center py-20 border border-dashed border-border/60 bg-[radial-gradient(ellipse_at_center,var(--surface-2)_0%,transparent_100%)]">
    <AlertTriangle className="size-8 text-muted-foreground/50 mb-4" />
    <h3 className="font-display text-xl tracking-widest text-muted-foreground uppercase opacity-80">
      {hasMatches ? "NO SESSIONS RECORDED" : "NO MATCH DATA"}
    </h3>
    <p className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.15em] mt-2 opacity-60">
      {hasMatches ? "ADJUST FILTER PARAMETERS" : "UPLOAD A VIDEO TO BEGIN AUTOMATED ANALYSIS"}
    </p>
  </div>
);
