import logging

logger = logging.getLogger(__name__)

# Weights for scoring
W1, W2, W3, W4 = 0.40, 0.20, 0.25, 0.15

EVENT_WEIGHTS = {
    "GOAL": 10.0,
    "PENALTY": 9.5,
    "RED_CARD": 9.0,
    "SAVE": 8.0,
    "YELLOW_CARD": 7.0,
    "CELEBRATION": 6.5,
    "FOUL": 6.0,
    "HIGHLIGHT": 5.5,
    "TACKLE": 5.0,
    "CORNER": 4.0,
    "OFFSIDE": 2.5,
}


def time_context_weight(timestamp: float, duration: float) -> float:
    """Late-game moments carry more weight."""
    if duration <= 0:
        return 0.70
    pct = timestamp / duration
    if pct > 0.92:
        return 1.00  # injury time / dying minutes
    if pct > 0.85:
        return 0.95  # final 10 min
    if pct > 0.70:
        return 0.85  # last quarter
    if pct > 0.50:
        return 0.75  # second half
    if pct > 0.45:
        return 0.60  # around half-time
    return 0.65  # first half


def score_goal(
    motion_score: float, timestamp: float, duration: float, confidence: float
) -> float:
    """Specific scoring logic for goals."""
    base_score = compute_context_score(
        "GOAL", motion_score, timestamp, duration, confidence
    )
    if duration > 0 and (timestamp / duration) > 0.85:
        return round(min(base_score * 1.2, 10.0), 2)  # Extra bump for late goals
    return base_score


def score_save(
    motion_score: float, timestamp: float, duration: float, confidence: float
) -> float:
    """Specific scoring logic for saves."""
    base_score = compute_context_score(
        "SAVE", motion_score, timestamp, duration, confidence
    )
    if duration > 0 and (timestamp / duration) < 0.08:
        return round(min(base_score * 1.3, 10.0), 2)  # Frantic early-game save bonus
    return base_score


def score_foul(
    motion_score: float, timestamp: float, duration: float, confidence: float
) -> float:
    """Specific scoring logic for fouls."""
    return compute_context_score("FOUL", motion_score, timestamp, duration, confidence)


def compute_context_score(
    event_type: str,
    motion_score: float,
    timestamp: float,
    duration: float,
    confidence: float,
) -> float:
    """
    Core contextual scoring engine.
    """
    ew = EVENT_WEIGHTS.get(event_type, 4.0) / 10.0
    audio = min(motion_score * 1.3, 1.0)
    tw = time_context_weight(timestamp, duration)

    base = (ew * W1) + (audio * W2) + (motion_score * W3) + (tw * W4)
    score = base * (0.5 + 0.5 * confidence)

    # Legacy logic maintained for GOAL and SAVE in the base function
    if duration > 0 and (timestamp / duration) > 0.85 and event_type == "GOAL":
        score *= 2.0
    if (
        duration > 0
        and (timestamp / duration) < 0.08
        and event_type in ("SAVE", "TACKLE")
    ):
        score *= 1.3

    return round(min(score * 10.0, 10.0), 2)
