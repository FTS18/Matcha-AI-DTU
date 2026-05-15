import type { MatchSummary, ProgressMap } from "@matcha/shared";
export function useMatches() {
  return {
    matches: [],
    loading: true,
    progressMap: {},
    deleteMatch: async (_id: string) => {},
  };
}
