import type { MatchEvent } from "./types";
export interface MatchSocketHandlers {
  onEvent?: (event: MatchEvent) => void;
  onProgress?: (matchId: string, progress: number) => void;
  onComplete?: (matchId: string) => void;
}
/**
 * Connects to the orchestrator, joins a match room, and subscribes to real-time events.
 * Returns a cleanup function — call it on unmount to disconnect.
 */
export declare function createMatchSocket(url: string, matchId: string, handlers: MatchSocketHandlers): () => void;
//# sourceMappingURL=socket.d.ts.map
