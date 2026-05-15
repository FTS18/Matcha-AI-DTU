import React from "react";
import { formatTime } from "@matcha/shared";

interface CardStatsProps {
  duration: number;
  eventCount: number;
  highlightCount: number;
  status: string;
}

export const CardStats = ({ duration, eventCount, highlightCount, status }: CardStatsProps) => {
  const stats = [
    {
      v: duration ? formatTime(duration) : "--:--",
      cls: "font-mono text-[10px] text-white/70 tabular-nums",
    },
    {
      v: status === "COMPLETED" ? eventCount.toString().padStart(2, "0") : "--",
      cls: "font-display text-[14px] text-accent drop-shadow-[0_0_8px_rgba(var(--color-accent),0.4)]",
    },
    {
      v: status === "COMPLETED" ? highlightCount.toString().padStart(2, "0") : "--",
      cls: "font-display text-[14px] text-primary drop-shadow-[0_0_8px_rgba(var(--color-primary),0.4)]",
    },
  ];

  return (
    <div className="flex items-stretch divide-x divide-white/5">
      {stats.map((stat, i) => (
        <div key={i} className="w-20 flex items-center justify-center bg-white/1">
          <span className={stat.cls}>{stat.v}</span>
        </div>
      ))}
    </div>
  );
};
