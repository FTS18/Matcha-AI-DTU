"use client";

import React, { useState, useCallback } from "react";
import Link from "next/link";
import {
  CheckCircle2,
  Loader2,
  Upload,
  XCircle,
} from "lucide-react";
import { useMatches } from "@/hooks/useMatches";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { STATUS_CONFIG as SHARED_STATUS_CONFIG } from "@matcha/shared";

import { MatchCard } from "./dashboard/match-card";
import { FilterTabs, FilterOption } from "./dashboard/filter-tabs";
import { EmptyState } from "./dashboard/empty-state";

const API_BASE =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ||
  process.env.NEXT_PUBLIC_API_URL?.replace("/api/v1", "") ||
  "http://localhost:4000";

const THEME_MAP: Record<string, { color: string }> = {
  success: {
    color: "text-emerald-400 bg-emerald-500/15 border-emerald-500/40",
  },
  info: { color: "text-blue-400 bg-blue-500/15 border-blue-500/40" },
  warning: { color: "text-amber-400 bg-amber-500/15 border-amber-500/40" },
  error: { color: "text-red-400 bg-red-500/15 border-red-500/40" },
};

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  COMPLETED: {
    ...SHARED_STATUS_CONFIG.COMPLETED,
    ...THEME_MAP[SHARED_STATUS_CONFIG.COMPLETED.theme],
    icon: <CheckCircle2 className="w-3 h-3" />,
  },
  PROCESSING: {
    ...SHARED_STATUS_CONFIG.PROCESSING,
    ...THEME_MAP[SHARED_STATUS_CONFIG.PROCESSING.theme],
    icon: <Loader2 className="w-3 h-3 animate-spin" />,
  },
  UPLOADED: {
    ...SHARED_STATUS_CONFIG.UPLOADED,
    ...THEME_MAP[SHARED_STATUS_CONFIG.UPLOADED.theme],
    icon: <Upload className="w-3 h-3" />,
  },
  FAILED: {
    ...SHARED_STATUS_CONFIG.FAILED,
    ...THEME_MAP[SHARED_STATUS_CONFIG.FAILED.theme],
    icon: <XCircle className="w-3 h-3" />,
  },
};

export const MatchDashboard = React.memo(function MatchDashboardContent() {
  const { matches, loading, progressMap, stageMap, deleteMatch, reanalyzeMatch, refetch } = useMatches();
  const [filter, setFilter] = useState<FilterOption>("ALL");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [reanalyzingId, setReanalyzingId] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await refetch();
    setTimeout(() => setIsRefreshing(false), 1000);
  }, [refetch]);

  const handleDelete = useCallback(
    async (id: string) => {
      setDeletingId(id);
      await deleteMatch(id);
      setDeletingId(null);
      setConfirmId(null);
    },
    [deleteMatch],
  );

  const handleReanalyze = useCallback(
    async (id: string) => {
      setReanalyzingId(id);
      await reanalyzeMatch(id);
      setReanalyzingId(null);
    },
    [reanalyzeMatch],
  );

  const visible = Array.isArray(matches)
    ? filter === "ALL"
      ? matches
      : matches.filter((m) => m.status?.toUpperCase() === filter.toUpperCase())
    : [];

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 border border-dashed border-border/50 bg-card/30">
        <Loader2 className="size-6 text-accent animate-spin mb-4" />
        <span className="font-mono text-[10px] text-muted-foreground uppercase tracking-[0.2em]">
          INITIALIZING ANALYTICS ENGINE...
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <FilterTabs
        filter={filter}
        setFilter={setFilter}
        matches={matches}
        isRefreshing={isRefreshing}
        handleRefresh={handleRefresh}
        statusConfig={STATUS_CONFIG}
      />

      {!visible.length && <EmptyState hasMatches={matches.length > 0} />}

      <motion.div
        className="grid grid-cols-1 gap-4"
        initial="hidden"
        animate="show"
        variants={{
          hidden: { opacity: 0 },
          show: {
            opacity: 1,
            transition: {
              staggerChildren: 0.05,
            },
          },
        }}
      >
        <AnimatePresence mode="popLayout">
          {visible.map((m) => (
            <MatchCard
              key={m.id}
              match={m}
              statusConfig={STATUS_CONFIG}
              progress={progressMap[m.id] ?? m.progress ?? 0}
              stage={stageMap[m.id]}
              isConfirming={confirmId === m.id}
              isDeleting={deletingId === m.id}
              reanalyzingId={reanalyzingId}
              onConfirmDelete={setConfirmId}
              onDelete={handleDelete}
              onReanalyze={handleReanalyze}
              apiBase={API_BASE}
            />
          ))}
        </AnimatePresence>
      </motion.div>
    </div>
  );
});
