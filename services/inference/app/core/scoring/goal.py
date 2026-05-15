import logging
from typing import Optional

logger = logging.getLogger(__name__)


def score_goal(
    motion_score: float,
    timestamp: float,
    duration: float,
    confidence: float,
    compute_context_score_fn: callable,
) -> float:
    """Specific scoring logic for goals."""
    base_score = compute_context_score_fn(
        "GOAL", motion_score, timestamp, duration, confidence
    )
    if duration > 0 and (timestamp / duration) > 0.85:
        return round(min(base_score * 1.2, 10.0), 2)  # Extra bump for late goals
    return base_score
