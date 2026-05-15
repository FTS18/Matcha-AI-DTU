import logging

logger = logging.getLogger(__name__)


def score_save(
    motion_score: float,
    timestamp: float,
    duration: float,
    confidence: float,
    compute_context_score_fn: callable,
) -> float:
    """Specific scoring logic for saves."""
    base_score = compute_context_score_fn(
        "SAVE", motion_score, timestamp, duration, confidence
    )
    if duration > 0 and (timestamp / duration) < 0.08:
        return round(min(base_score * 1.3, 10.0), 2)  # Frantic early-game save bonus
    return base_score
