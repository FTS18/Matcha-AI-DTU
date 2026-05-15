import logging
import requests
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:4000/api/v1")


def emit_progress(match_id: str, pct: int, stage: str = ""):
    """Send progress updates to the orchestrator."""
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
            json={"progress": min(pct, 99), "stage": stage},
            timeout=1,
        )
        if stage:
            logger.info(f"Progress {pct}%: {stage}")
    except Exception as e:
        logger.debug(f"Failed to emit progress: {e}")


def emit_live_event(match_id: str, event: Dict[str, Any]):
    """Send a live event to the orchestrator for WebSocket broadcast."""
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/live-event",
            json=event,
            timeout=2,
        )
    except Exception as e:
        logger.debug(f"Failed to emit live event: {e}")


def emit_tracking_frames(match_id: str, frames: List[Dict[str, Any]]):
    """Send tracking data batches to the orchestrator."""
    if not frames:
        return
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/tracking-update",
            json={"frames": frames},
            timeout=3,
        )
    except Exception as e:
        logger.debug(f"Failed to emit tracking update: {e}")


def report_failure(match_id: str):
    """Notify orchestrator that analysis has failed."""
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
            json={"progress": -1, "stage": "failed"},
            timeout=3,
        )
    except Exception as e:
        logger.debug(f"Failed to report failure: {e}")
