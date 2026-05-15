import { io, Socket } from "socket.io-client";
import type { MatchEvent } from "./types";
import { WsEvents } from "./constants";

export interface MatchSocketHandlers {
  onEvent?: (event: MatchEvent) => void;
  onProgress?: (matchId: string, progress: number) => void;
  onComplete?: (matchId: string) => void;
}

/**
 * Connects to the orchestrator, joins a match room, and subscribes to real-time events.
 * Returns a cleanup function — call it on unmount to disconnect.
 */
export function createMatchSocket(url: string, matchId: string, handlers: MatchSocketHandlers): () => void {
  const socket: Socket = io(url, { transports: ["websocket"] });
  socket.emit(WsEvents.JOIN_MATCH, matchId);

  socket.on(WsEvents.MATCH_EVENT, (payload: { matchId: string; event: MatchEvent }) => {
    if (payload.matchId !== matchId) return;
    handlers.onEvent?.(payload.event);
  });

  socket.on(WsEvents.PROGRESS, (data: { matchId: string; progress: number }) => {
    handlers.onProgress?.(data.matchId, data.progress);
  });

  socket.on(WsEvents.COMPLETE, (data: { matchId: string }) => {
    handlers.onComplete?.(data.matchId);
  });

  return () => socket.disconnect();
}
