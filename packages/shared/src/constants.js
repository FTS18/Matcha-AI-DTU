export const STATUS_CONFIG = {
    COMPLETED: { label: "Completed", theme: "success" },
    PROCESSING: { label: "Processing", theme: "info" },
    UPLOADED: { label: "Uploaded", theme: "warning" },
    FAILED: { label: "Failed", theme: "error" },
};
export const EVENT_CONFIG = {
    GOAL: { label: "Goal", theme: "success" },
    TACKLE: { label: "Tackle", theme: "warning" },
    FOUL: { label: "Foul", theme: "error" },
    SAVE: { label: "Save", theme: "info" },
    Celebrate: { label: "Celeb", theme: "accent" },
};
export const DEFAULT_EVENT_CONFIG = { label: "Event", theme: "neutral" };
/** WebSocket Event Names */
export var WsEvents;
(function (WsEvents) {
    WsEvents["JOIN_MATCH"] = "joinMatch";
    WsEvents["MATCH_EVENT"] = "matchEvent";
    WsEvents["PROGRESS"] = "progress";
    WsEvents["COMPLETE"] = "complete";
    WsEvents["TRACKING_UPDATE"] = "trackingUpdate";
})(WsEvents || (WsEvents = {}));

/** Pipeline stage labels */
export const PIPELINE_STAGES = {
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
