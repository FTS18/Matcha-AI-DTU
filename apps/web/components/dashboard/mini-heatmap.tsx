import React from "react";

export const MiniHeatmap = React.memo(({ matches }: { matches: any }) => (
  <svg
    viewBox="0 0 100 60"
    className="w-full h-full opacity-60 group-hover:opacity-100 transition-opacity duration-500"
  >
    <rect x="0" y="0" width="100" height="60" fill="var(--muted)" fillOpacity="0.1" rx="2" />
    <circle cx="20" cy="30" r="15" fill="var(--accent)" fillOpacity="0.2" filter="blur(4px)" />
    <circle cx="80" cy="20" r="10" fill="var(--primary)" fillOpacity="0.2" filter="blur(4px)" />
    <path d="M50 0 L50 60" stroke="currentColor" strokeOpacity="0.1" strokeWidth="0.5" />
    <circle cx="50" cy="30" r="8" fill="none" stroke="currentColor" strokeOpacity="0.1" strokeWidth="0.5" />
  </svg>
));

MiniHeatmap.displayName = "MiniHeatmap";
