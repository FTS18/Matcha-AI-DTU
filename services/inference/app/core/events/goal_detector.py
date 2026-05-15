import cv2
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def detect_goals_in_video(
    video_path: str, config: Dict[str, Any], engine_class: Any
) -> List[Dict[str, Any]]:
    """
    Detect goals in video using vision-based goal-line crossing detection.
    """
    if not engine_class:
        return []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video for goal detection: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

        # Initialize goal detector with Roboflow support
        rf_cfg = {
            "api_key": config.get("ROBOFLOW_API_KEY"),
            "workspace": config.get("ROBOFLOW_WORKSPACE", "matcha-ai"),
            "project": config.get("ROBOFLOW_PROJECT", "soccer-ball-detection"),
            "version": int(config.get("ROBOFLOW_VERSION", "1")),
        }
        goal_engine = engine_class(roboflow_cfg=rf_cfg)
        goal_engine.init(frame_width, frame_height, fps)

        logger.info(f"Goal detection: {frame_width}×{frame_height} @ {fps:.1f}fps")

        goals_detected = []
        frame_idx = 0
        frame_step = max(1, int(fps / 5.0))  # Process at ~5 FPS

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_step != 0:
                continue

            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))

            goal_event = goal_engine.process_frame(frame)
            if goal_event:
                goals_detected.append(
                    {
                        "timestamp": round(goal_event.timestamp, 2),
                        "type": "GOAL",
                        "confidence": round(goal_event.confidence, 3),
                        "description": f"Goal detected ({goal_event.direction})",
                        "source": "goal_detection",
                    }
                )
                logger.info(
                    f" GOAL at {goal_event.timestamp:.1f}s | confidence: {goal_event.confidence:.2f}"
                )

        cap.release()
        logger.info(f"Goal detection completed: {len(goals_detected)} goals found")
        return goals_detected

    except Exception as e:
        logger.error(f"Goal detection failed: {e}")
        return []
