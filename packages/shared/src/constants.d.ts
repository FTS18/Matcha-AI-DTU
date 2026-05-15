import type { MatchStatus } from "./types";
/** Logic-to-Theme mapping: used to pick standard Tailwind styles in frontend. */
export type ThemeColor = "success" | "warning" | "error" | "info" | "neutral" | "accent";
export declare const STATUS_CONFIG: Record<
  MatchStatus,
  {
    label: string;
    theme: ThemeColor;
  }
>;
export declare const EVENT_CONFIG: Record<
  string,
  {
    label: string;
    theme: ThemeColor;
  }
>;
export declare const DEFAULT_EVENT_CONFIG: {
  label: string;
  theme: ThemeColor;
};
/** WebSocket Event Names */
export declare enum WsEvents {
  JOIN_MATCH = "joinMatch",
  MATCH_EVENT = "matchEvent",
  PROGRESS = "progress",
  COMPLETE = "complete",
}
//# sourceMappingURL=constants.d.ts.map
