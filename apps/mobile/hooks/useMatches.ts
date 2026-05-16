// No shared types needed currently in this hook
export function useMatches() {
  return {
    matches: [],
    loading: true,
    progressMap: {},
    deleteMatch: async (_id: string) => {},
  };
}
