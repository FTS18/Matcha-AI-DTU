import logging
from typing import List, Dict, Any


def time_context_weight(timestamp: float, duration: float) -> float:
    """Assign higher weight to late-game events."""
    if not duration:
        return 0.5
    progress = timestamp / duration
    if progress > 0.85:
        return 1.2
    if progress < 0.10:
        return 0.8
    return 1.0


def calculate_emotion_scores(
    motion_windows: list, duration: float
) -> List[Dict[str, Any]]:
    """Generate emotion score timeline."""
    return [
        {
            "timestamp": w["timestamp"],
            "audioScore": w["audioScore"],
            "motionScore": w["motionScore"],
            "contextWeight": round(time_context_weight(w["timestamp"], duration), 3),
            "finalScore": round(
                (
                    w["audioScore"] * 0.3
                    + w["motionScore"] * 0.5
                    + time_context_weight(w["timestamp"], duration) * 0.2
                )
                * 10,
                2,
            ),
        }
        for w in motion_windows
    ]


_FALLBACK = {
    "GOAL": {
        "high": "GOOOAL! Sensational — the crowd erupts!",
        "mid": "Goal! Crucial finish puts them ahead!",
        "low": "Goal scored.",
    },
    "TACKLE": {
        "high": "FEROCIOUS TACKLE! Incredible commitment!",
        "mid": "Strong challenge wins the ball back.",
        "low": "Tackle wins possession.",
    },
    "FOUL": {
        "high": "DEFINITE FOUL! Referee steps in immediately!",
        "mid": "Free kick awarded — bodies flying here.",
        "low": "Foul given.",
    },
    "SAVE": {
        "high": "UNBELIEVABLE SAVE! Superhuman goalkeeping!",
        "mid": "Good stop from the keeper — keeping them in it.",
        "low": "Save made.",
    },
    "CELEBRATION": {
        "high": "INCREDIBLE SCENES! The players are losing their minds!",
        "mid": "Celebrations break out on the pitch!",
        "low": "The players celebrate.",
    },
    "HIGHLIGHT": {
        "high": "WHAT A MOMENT! Crucial action in this match!",
        "mid": "Important moment of play here.",
        "low": "Key moment of play.",
    },
}


def get_fallback_commentary(
    event_type: str, final_score: float, timestamp: float, duration: float
) -> str:
    """Generate simple fallback commentary if AI fails."""
    minute = max(1, int(timestamp / 60))
    late = duration > 0 and (timestamp / duration) > 0.85
    energy = "high" if final_score >= 7.5 else ("mid" if final_score >= 5 else "low")

    text = _FALLBACK.get(event_type, {}).get(
        energy, f"{event_type} at minute {minute}."
    )

    if "minute" not in text.lower():
        text = text.rstrip("!.") + f" at minute {minute}."
    if late and final_score >= 7:
        text = "LATE DRAMA! " + text

    return text


def compile_final_payload(
    scored_events: list,
    highlights: list,
    emotion_scores: list,
    duration: float,
    summary: str,
    urls: dict,
    tracking_data: list,
    team_colors: list,
    extra_stats: dict,
) -> dict:
    """Package everything for the orchestrator."""
    import numpy as np

    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (float, np.floating)):
            return float(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    payload = {
        "events": convert_numpy(scored_events),
        "highlights": convert_numpy(highlights),
        "emotionScores": convert_numpy(emotion_scores),
        "duration": round(float(duration), 1),
        "summary": summary,
        "highlightReelUrl": urls.get("landscape"),
        "highlightReelPortraitUrl": urls.get("portrait"),
        "thumbnailUrl": urls.get("thumbnail"),
        "heatmapUrl": urls.get("heatmap"),
        "videoUrl": urls.get("video"),
        "trackingData": convert_numpy(tracking_data),
        "teamColors": convert_numpy(team_colors),
        "topSpeedKmh": round(float(extra_stats.get("topSpeed", 0.0)), 1),
        "goalpostDetections": convert_numpy(extra_stats.get("goalposts", [])),
        "advancedStats": convert_numpy(extra_stats.get("advanced", {})),
        "formationData": extra_stats.get("formation", {}),
        "trajectoryData": extra_stats.get("trajectory", {}),
        "audioVolumes": extra_stats.get("audioVolumes", {}),
    }
    return payload
