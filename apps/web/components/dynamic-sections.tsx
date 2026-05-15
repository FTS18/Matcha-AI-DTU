"use client";

// ssr: false is required because VideoUpload and MatchDashboard both use
// socket.io-client, which accesses `window` at import time and crashes during SSR.
// next/dynamic with ssr: false is only legal inside Client Components (Next.js 15).

import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

export const VideoUpload = dynamic(() => import("@/components/video-upload").then((mod) => mod.VideoUpload), {
  ssr: false,
  loading: () => (
    <div className="flex flex-col items-center justify-center p-12 border border-dashed border-border rounded-xl">
      <Loader2 className="size-6 text-muted-foreground animate-spin mb-4" />
      <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.14em]">
        INITIALIZING UPLOAD CORE...
      </span>
    </div>
  ),
});

export const MatchDashboard = dynamic(() => import("@/components/match-dashboard").then((mod) => mod.MatchDashboard), {
  ssr: false,
  loading: () => (
    <div className="flex flex-col items-center justify-center h-48 border border-border rounded-xl bg-card/50">
      <Loader2 className="size-6 text-muted-foreground animate-spin mb-4" />
      <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.14em]">
        CONNECTING TO ORCHESTRATOR...
      </span>
    </div>
  ),
});
