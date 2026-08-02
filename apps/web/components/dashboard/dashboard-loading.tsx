import React from "react";
import { MatchCardSkeleton } from "./match-card-skeleton";

export const DashboardLoading = () => (
  <div className="space-y-4">
    <div className="flex items-center gap-2 px-1">
      <div className="size-3 border-2 border-primary border-t-transparent rounded-full animate-spin" />
      <span className="font-mono text-[9px] text-muted-foreground uppercase tracking-[0.2em]">
        FETCHING ANALYTICS FROM DATABASE...
      </span>
    </div>
    <div className="grid grid-cols-1 gap-4">
      <MatchCardSkeleton />
      <MatchCardSkeleton />
      <MatchCardSkeleton />
    </div>
  </div>
);

