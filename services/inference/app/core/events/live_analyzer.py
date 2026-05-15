import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def analyze_live_window(
    frame: Any, timestamp: float, m_score: float, language: str, analyze_frame_fn: Any
) -> Optional[Dict[str, Any]]:
    """Perform sliding window vision analysis on high-intensity moments."""
    try:
        analysis_res = analyze_frame_fn(
            frame, timestamp, context=f"Language: {language}"
        )
        if (
            analysis_res.get("event_type", "NONE") != "NONE"
            and analysis_res.get("confidence", 0) >= 0.65
        ):
            return {
                "timestamp": round(timestamp, 2),
                "type": analysis_res["event_type"],
                "confidence": round(analysis_res["confidence"], 3),
                "commentary": analysis_res.get("description", ""),
                "finalScore": round(m_score * 10, 1),
                "source": "live_sliding_window",
            }
    except Exception as e:
        logger.debug(f"Live window analysis error: {e}")

    return None
