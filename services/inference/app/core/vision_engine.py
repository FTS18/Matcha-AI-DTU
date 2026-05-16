import logging
import torch
from typing import Optional, Tuple, Any
from app.core.soccer_analysis.config import CONFIG

logger = logging.getLogger(__name__)


class VisionEngine:
    def __init__(self):
        self.model = None
        self.ball_model = None
        self.gpu_available = False

    def init_models(self):
        """Initialize YOLO models for pose/person and ball detection."""
        try:
            from ultralytics import YOLO

            # Check GPU availability
            self.gpu_available = (
                CONFIG.get("ENABLE_GPU_ACCELERATION", True)
                and torch.cuda.is_available()
            )
            device = 0 if self.gpu_available else "cpu"

            # Person / Pose model
            self.model = YOLO("yolov8n-pose.pt")

            # Ball model (often same as generic yolov8n but tuned)
            self.ball_model = YOLO("yolov8n.pt")

            if self.gpu_available:
                logger.info(
                    f" VisionEngine: GPU acceleration enabled ({torch.cuda.get_device_name(0)})"
                )
                self.model.to(device)
                self.ball_model.to(device)
            else:
                logger.info(" VisionEngine: Using CPU for inference")

            return True
        except Exception as e:
            logger.error(f" VisionEngine: Failed to initialize models: {e}")
            return False

    def get_models(self) -> Tuple[Any, Any]:
        return self.model, self.ball_model
