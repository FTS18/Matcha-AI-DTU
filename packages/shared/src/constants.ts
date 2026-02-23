import type { MatchStatus } from "./types";

/** Logic-to-Theme mapping: used to pick standard Tailwind styles in frontend. */
export type ThemeColor = "success" | "warning" | "error" | "info" | "neutral" | "accent";

export const STATUS_CONFIG: Record<MatchStatus, { label: string; theme: ThemeColor }> = {
  COMPLETED:  { label: "Completed",  theme: "success" },
  PROCESSING: { label: "Processing", theme: "info" },
  UPLOADED:   { label: "Uploaded",   theme: "warning" },
  FAILED:     { label: "Failed",     theme: "error" },
};

export const EVENT_CONFIG: Record<string, { label: string; theme: ThemeColor }> = {
  GOAL:      { label: "Goal",    theme: "success" },
  TACKLE:    { label: "Tackle",  theme: "warning" },
  FOUL:      { label: "Foul",    theme: "error" },
  SAVE:      { label: "Save",    theme: "info" },
  Celebrate: { label: "Celeb",   theme: "accent" },
};

export const DEFAULT_EVENT_CONFIG = { label: "Event", theme: "neutral" as ThemeColor };

/** WebSocket Event Names */
export enum WsEvents {
  JOIN_MATCH = "joinMatch",
  MATCH_EVENT = "matchEvent",
  PROGRESS = "progress",
  COMPLETE = "complete",
  TRACKING_UPDATE = "trackingUpdate",
}

/** Pipeline stage labels — sent with progress updates for live logging */
export const PIPELINE_STAGES: Record<string, string> = {
  queued: "Queued for analysis",
  downloading: "Downloading video",
  compressing: "Compressing video",
  scanning: "Scanning frames",
  tracking: "Tracking players & ball",
  motion: "Analysing motion",
  events: "Detecting events",
  cv_physics: "CV physics analysis",
  scoring: "Scoring events",
  commentary: "Generating AI commentary",
  highlights: "Selecting highlights",
  reel: "Generating highlight reel",
  heatmap: "Generating heatmap & analytics",
  tactics: "Tactical analysis",
  summary: "Generating AI summary",
  thumbnail: "Generating thumbnail",
  saving: "Saving results",
  done: "Analysis complete",
};
