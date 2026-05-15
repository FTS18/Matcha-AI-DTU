"use client";

import React from "react";

export interface ScoreBadgeProps {
  score: number;
  className?: string;
}

export function ScoreBadge({ score, className = "" }: ScoreBadgeProps) {
  const color = score >= 7.5 ? "text-emerald-400" : score >= 5 ? "text-amber-400" : "text-zinc-500";
  return <span className={`font-mono text-sm font-bold ${color} ${className}`}>{score.toFixed(1)}</span>;
}
