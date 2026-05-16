import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def spot_transformer_actions(
    frame_buffer: List[Any], timestamp: float, fps: float, spotter: Any
) -> Optional[Dict[str, Any]]:
    """Analyze a window of frames using Vision Transformer for action spotting."""
    if len(frame_buffer) < 16 or spotter is None:
        return None

    try:
        vit_res = spotter.spot_actions(list(frame_buffer), fps)
        if vit_res and vit_res.get("confidence", 0) >= 0.65:
            vit_label = vit_res["label"].lower()
            our_type = "HIGHLIGHT"
            if "goal" in vit_label or "kick" in vit_label:
                our_type = "GOAL"
            elif "tackle" in vit_label:
                our_type = "TACKLE"

            return {
                "timestamp": round(timestamp, 2),
                "type": our_type,
                "confidence": vit_res["confidence"],
                "commentary": f"Transformer analysis: {vit_res['label']}",
                "finalScore": round(vit_res["confidence"] * 10, 1),
                "source": "vision_transformer",
            }
    except Exception:
        logger.debug("ViT analysis failed (silent)")

    return None
