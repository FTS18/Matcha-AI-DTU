import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { io, Socket } from "socket.io-client";
import type { MatchSummary, ProgressMap, StageMap } from "@matcha/shared";
import { createApiClient, WsEvents } from "@matcha/shared";

export function useMatches(baseUrl: string = "http://localhost:4000") {
  const [matches, setMatches] = useState<MatchSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [progressMap, setProgressMap] = useState<ProgressMap>({});
  const [stageMap, setStageMap] = useState<StageMap>({});

  const socketRef = useRef<Socket | null>(null);
  const client = useMemo(() => createApiClient(baseUrl), [baseUrl]);

  const fetchMatches = useCallback(async () => {
    try {
      const data = await client.getMatches();
      setMatches(data);
      // Seed progress map from API data — API is the source of truth
      const serverProgress: ProgressMap = {};
      data.forEach(m => {
        if (m.status === "PROCESSING" || m.status === "UPLOADED") {
          serverProgress[m.id] = m.progress ?? 0;
        }
      });
      // Merge: use higher of server vs local for active matches (WebSocket may be ahead)
      setProgressMap(prev => {
        const merged: ProgressMap = { ...serverProgress };
        Object.keys(prev).forEach(id => {
          if (merged[id] !== undefined && prev[id] > merged[id]) {
            merged[id] = prev[id]; // WebSocket was ahead
          }
        });
        return merged;
      });
    } catch {
      // API Offline
    } finally {
      setLoading(false);
    }
  }, [client]);

  // WebSocket connection — stable across renders
  useEffect(() => {
    const socket = io(baseUrl, { transports: ["websocket", "polling"] });
    socketRef.current = socket;

    socket.on(WsEvents.PROGRESS, (data: { matchId: string; progress: number; stage?: string }) => {
      if (data.progress === -1) {
        // Match failed — refetch to get updated status
        fetchMatches();
        return;
      }
      setProgressMap(prev => ({ ...prev, [data.matchId]: data.progress }));
      if (data.stage) {
        setStageMap((prev: StageMap) => ({ ...prev, [data.matchId]: data.stage! }));
      }
      if (data.progress >= 100) {
        setTimeout(fetchMatches, 500);
      }
    });

    socket.on(WsEvents.COMPLETE, (data: { matchId: string }) => {
      setProgressMap(prev => ({ ...prev, [data.matchId]: 100 }));
      fetchMatches();
    });

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [baseUrl, fetchMatches]);

  // Join rooms for all active matches whenever the match list changes
  useEffect(() => {
    if (!socketRef.current || !Array.isArray(matches)) return;
    const active = matches.filter(m => m.status === "PROCESSING" || m.status === "UPLOADED");
    active.forEach(m => {
      socketRef.current?.emit(WsEvents.JOIN_MATCH, m.id);
    });
  }, [matches]);

  // Polling + global refresh listener
  useEffect(() => {
    fetchMatches();
    const interval = setInterval(fetchMatches, 4000);

    const handleRefresh = () => fetchMatches();
    window.addEventListener("matcha:refresh", handleRefresh);

    return () => {
      clearInterval(interval);
      window.removeEventListener("matcha:refresh", handleRefresh);
    };
  }, [baseUrl, fetchMatches]);

  const deleteMatch = async (id: string) => {
    try {
      await client.deleteMatch(id);
      setMatches(prev => prev.filter(m => m.id !== id));
      return true;
    } catch {
      return false;
    }
  };

  const reanalyzeMatch = async (id: string) => {
    try {
      await client.reanalyze(id);
      await fetchMatches();
      return true;
    } catch {
      return false;
    }
  };

  return { matches, loading, progressMap, stageMap, deleteMatch, reanalyzeMatch, refetch: fetchMatches };
}
