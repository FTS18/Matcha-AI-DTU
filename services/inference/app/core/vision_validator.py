import logging
import cv2
from typing import Optional, Dict, List
from collections import Counter
from app.core.llm import analyze_frames_batch

logger = logging.getLogger(__name__)

# Track consecutive Vision AI failures for fallback
_vision_failures = 0
_MAX_VISION_FAILURES = 5


def validate_candidate_moment(
    cap, timestamp: float, fps: float, duration: float
) -> Optional[Dict]:
    """
    Validate a candidate moment by analyzing multiple frames around it.
    Sends all 3 frames in ONE Gemini batch call.
    """
    frames_with_ts: list = []
    for offset in [-0.5, 0.0, 0.5]:
        t = timestamp + offset
        if t < 0 or t > duration:
            continue
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        frames_with_ts.append((frame, t))

    if not frames_with_ts:
        return None

    batch_results = analyze_frames_batch(frames_with_ts)
    results = [
        r for r in batch_results if r["event_type"] != "NONE" and r["confidence"] >= 0.5
    ]

    if not results:
        return None

    event_counts = Counter(r["event_type"] for r in results)
    most_common_event, count = event_counts.most_common(1)[0]

    # Goals are dramatic enough that 1 confident detection is enough
    if most_common_event != "GOAL" and count < 2:
        return None

    matching = [r for r in results if r["event_type"] == most_common_event]
    avg_confidence = sum(r["confidence"] for r in matching) / len(matching)
    best_result = max(matching, key=lambda r: r["confidence"])

    return {
        "event_type": most_common_event,
        "confidence": round(avg_confidence, 3),
        "timestamp": round(timestamp, 2),
        "description": best_result["description"],
        "frame_votes": count,
    }


def fallback_heuristic_event(
    motion_score: float, timestamp: float, duration: float
) -> Optional[Dict]:
    """Fallback event detection when Vision AI is unavailable."""
    late_game = duration > 0 and (timestamp / duration) > 0.75

    if motion_score >= 0.7:
        event_type = "HIGHLIGHT"
        confidence = min(0.6, motion_score * 0.8)
        desc = "High action moment detected (Vision AI fallback)"
    elif motion_score >= 0.55 and late_game:
        event_type = "HIGHLIGHT"
        confidence = 0.5
        desc = "Late-game action moment (Vision AI fallback)"
    else:
        return None

    return {
        "event_type": event_type,
        "confidence": round(confidence, 3),
        "timestamp": round(timestamp, 2),
        "description": desc,
    }


def validate_candidate_with_fallback(
    cap, timestamp: float, fps: float, duration: float, motion_score: float
) -> Optional[Dict]:
    """Validate a candidate moment, with fallback to heuristics if Vision AI fails."""
    global _vision_failures

    if _vision_failures >= _MAX_VISION_FAILURES:
        return fallback_heuristic_event(motion_score, timestamp, duration)

    result = validate_candidate_moment(cap, timestamp, fps, duration)

    if result is None:
        if motion_score >= 0.65:
            _vision_failures += 1
            if _vision_failures >= _MAX_VISION_FAILURES:
                logger.warning(
                    f"Vision AI failed {_vision_failures} times - switching to fallback mode"
                )
                return fallback_heuristic_event(motion_score, timestamp, duration)
        return None

    _vision_failures = 0
    return result


def find_motion_peaks(motion_windows: list, threshold: float, min_gap: float) -> list:
    """Identify motion peaks above threshold with minimum temporal gap."""
    if not isinstance(motion_windows, list) or not motion_windows:
        return []

    candidates = []
    last_peak = -999

    for w in motion_windows:
        if not isinstance(w, dict) or "motionScore" not in w or "timestamp" not in w:
            continue
        if w["motionScore"] >= threshold:
            t = w["timestamp"]
            if t - last_peak >= min_gap:
                candidates.append(t)
                last_peak = t

    return candidates
