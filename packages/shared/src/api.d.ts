/** HTTP API client for the Matcha AI orchestrator.
 *  Import this in apps/web and apps/mobile — pass the base URL from your env/constants.
 */
import type { MatchSummary, MatchDetail } from "./types";
export declare function createApiClient(baseUrl: string): {
    /** Helper to resolve relative asset paths (e.g. /uploads/...) to full URLs */
    getAssetUrl: (path: string | null) => string;
    getMatches: () => Promise<MatchSummary[]>;
    getMatch: (id: string) => Promise<MatchDetail>;
    deleteMatch: (id: string) => Promise<Response>;
    reanalyze: (id: string) => Promise<Response>;
    generateReel: (id: string, aspectRatio: "16:9" | "9:16") => Promise<Response>;
    uploadVideo: (file: File | Blob, onProgress?: (pct: number) => void) => Promise<MatchSummary>;
    uploadYoutube: (url: string, startTime?: number, endTime?: number) => Promise<MatchSummary>;
    login: (body: any) => Promise<{
        access_token: string;
        user: any;
    }>;
    register: (body: any) => Promise<{
        access_token: string;
        user: any;
    }>;
    getMe: () => Promise<any>;
    /** Update a single highlight (timestamps, eventType, etc.) */
    updateHighlight: (matchId: string, highlightId: string, data: {
        startTime?: number;
        endTime?: number;
        eventType?: string;
        score?: number;
        commentary?: string;
    }) => Promise<any>;
    /** Delete (reject) a single highlight */
    deleteHighlight: (matchId: string, highlightId: string) => Promise<{ ok: boolean }>;
};
export type ApiClient = ReturnType<typeof createApiClient>;
//# sourceMappingURL=api.d.ts.map