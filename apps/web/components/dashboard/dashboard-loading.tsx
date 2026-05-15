import React from "react";
import { Loader2 } from "lucide-react";

export const DashboardLoading = () => (
  <div className="flex flex-col items-center justify-center h-64 border border-dashed border-border/50 bg-card/30">
    <Loader2 className="size-6 text-accent animate-spin mb-4" />
    <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.2em]">
      INITIALIZING ANALYTICS ENGINE...
    </span>
  </div>
);
