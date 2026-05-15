import logging

logger = logging.getLogger(__name__)


def score_foul(
    motion_score: float,
    timestamp: float,
    duration: float,
    confidence: float,
    compute_context_score_fn: callable,
) -> float:
    """Specific scoring logic for fouls."""
    return compute_context_score_fn(
        "FOUL", motion_score, timestamp, duration, confidence
    )
