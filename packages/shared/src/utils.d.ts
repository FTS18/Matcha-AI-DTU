import type { MatchEvent, EmotionScore } from "./types";
/** Sort events by finalScore descending, return top N. */
export declare function getTop5Moments(events: MatchEvent[], n?: number): MatchEvent[];
/** Count events by type — { GOAL: 2, TACKLE: 5, ... } */
export declare function countEventsByType(events: MatchEvent[]): Record<string, number>;
/** Filter events by type string, or return all if type is "ALL". */
export declare function filterEventsByType(events: MatchEvent[], type: string): MatchEvent[];
/** Find the motion score nearest to currentTime in the emotion score array. */
export declare function getLiveIntensity(scores: EmotionScore[], currentTime: number): number;
/** Average confidence of all events, 0 if none. */
export declare function avgConfidence(events: MatchEvent[]): number;
/** Highest finalScore across all events, 0 if none. */
export declare function maxScore(events: MatchEvent[]): number;
/** Format seconds → "m:ss" */
export declare function formatTime(secs: number | null): string;
/** Relative time formatter: "5m ago", "just now", etc. */
export declare function timeAgo(iso: string | null): string;
/** Regex for all standard YouTube URL formats */
export declare const YOUTUBE_REGEX: RegExp;
/** Validate if a string is a valid YouTube URL */
export declare function isYoutubeUrl(url: string): boolean;
/** Extract video ID from any YouTube URL */
export declare function extractYoutubeId(url: string): string | null;
/** Fetch with automatic retries and exponential backoff */
export declare function fetchWithRetry(
  url: string,
  options?: RequestInit,
  retries?: number,
  backoff?: number,
): Promise<Response>;
//# sourceMappingURL=utils.d.ts.map
