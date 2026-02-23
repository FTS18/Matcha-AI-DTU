/** All shared TypeScript types for the Matcha AI platform.
 *  Update this file when the orchestrator API schema changes.
 *  Both apps/web and apps/mobile import from here.
 */
import { CreateMatchInput, LoginInput, RegisterInput } from "@matcha/contracts";
export type { CreateMatchInput, LoginInput, RegisterInput };
export interface MatchEvent {
    id: string;
    timestamp: number;
    type: string;
    confidence: number;
    finalScore: number;
    commentary: string | null;
}
export interface Highlight {
    id: string;
    startTime: number;
    endTime: number;
    score: number;
    eventType: string | null;
    commentary: string | null;
    videoUrl: string | null;
}
export interface EmotionScore {
    timestamp: number;
    audioScore: number;
    motionScore: number;
    finalScore: number;
}
/** Tracking frame: {t: seconds, b: [[nx,ny,nw,nh,conf],...], p: [[nx,ny,nw,nh,tid,team],...]} */
export type TrackFrame = {
    t: number;
    b: number[][];
    p: number[][];
};
export interface MatchDetail {
    id: string;
    status: MatchStatus;
    progress: number;
    duration: number | null;
    uploadUrl: string;
    createdAt: string;
    summary: string | null;
    highlightReelUrl: string | null;
    highlightReelPortraitUrl: string | null;
    trackingData: TrackFrame[] | null;
    teamColors: number[][] | null;
    heatmapUrl: string | null;
    topSpeedKmh: number | null;
    thumbnailUrl: string | null;
    events: MatchEvent[];
    highlights: Highlight[];
    emotionScores: EmotionScore[];
}
export interface MatchSummary {
    id: string;
    status: MatchStatus;
    progress: number;
    duration: number | null;
    createdAt: string;
    uploadUrl: string;
    heatmapUrl: string | null;
    topSpeedKmh: number | null;
    thumbnailUrl: string | null;
    _count: {
        events: number;
        highlights: number;
    };
}
export interface AnalysisPayload {
    events: {
        timestamp: number;
        type: string;
        confidence: number;
        finalScore?: number;
        commentary?: string;
    }[];
    highlights?: {
        startTime: number;
        endTime: number;
        score: number;
        eventType?: string;
        commentary?: string;
        videoUrl?: string;
    }[];
    emotionScores?: {
        timestamp: number;
        audioScore: number;
        motionScore: number;
        contextWeight: number;
        finalScore: number;
    }[];
    duration?: number;
    summary?: string;
    highlightReelUrl?: string;
    highlightReelPortraitUrl?: string;
    trackingData?: TrackFrame[];
    teamColors?: number[][];
    heatmapUrl?: string;
    thumbnailUrl?: string;
    topSpeedKmh?: number;
    videoUrl?: string;
}
export type MatchStatus = "UPLOADED" | "PROCESSING" | "COMPLETED" | "FAILED";
export type ProgressMap = Record<string, number>;
//# sourceMappingURL=types.d.ts.map