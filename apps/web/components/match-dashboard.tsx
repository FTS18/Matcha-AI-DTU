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
import { DashboardLoading } from "./dashboard/dashboard-loading";
import { STATUS_CONFIG } from "./dashboard/dashboard-config";

const API_BASE =
  process.env.NEXT_PUBLIC_ORCHESTRATOR_URL ||
  process.env.NEXT_PUBLIC_API_URL?.replace("/api/v1", "") ||
  "http://localhost:4000";

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
    return <DashboardLoading />;
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
