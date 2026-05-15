import os
import logging
from typing import Optional, List, Dict
import numpy as np
from .utils import get_center_of_bbox, get_foot_position

logger = logging.getLogger(__name__)


class SoccerTracker:
    """
    Wraps YOLO detection + ByteTrack for consistent player/ball/referee tracking.
    """

    def __init__(self, model_path: Optional[str] = None):
        from ultralytics import YOLO
        import supervision as sv

        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Soccer tracker using custom model: {model_path}")
        else:
            self.model = YOLO("yolov8n.pt")
            logger.info("Soccer tracker using yolov8n.pt (generic)")

        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames: list, batch_size: int = 20) -> list:
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(
                frames[i : i + batch_size], conf=0.1, verbose=False
            )
            detections += batch
        return detections

    def get_object_tracks(self, frames: list) -> dict:
        import supervision as sv

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)

            # Convert goalkeeper → player
            if "goalkeeper" in cls_names_inv and "player" in cls_names_inv:
                for idx, class_id in enumerate(detection_sv.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_sv.class_id[idx] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # Players / Referees / General person class
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                label = cls_names.get(cls_id, "")

                if label in ("player", "person"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif label == "referee":
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Ball (untracked — use detection directly)
            for frame_detection in detection_sv:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                label = cls_names.get(cls_id, "")
                if label in ("ball", "sports ball"):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        return tracks

    @staticmethod
    def add_position_to_tracks(tracks: dict):
        for obj_name, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    position = (
                        get_center_of_bbox(bbox)
                        if obj_name == "ball"
                        else get_foot_position(bbox)
                    )
                    tracks[obj_name][frame_num][track_id]["position"] = position

    @staticmethod
    def interpolate_ball_positions(ball_positions: list) -> list:
        """Fill gaps in ball detection using linear interpolation."""
        import pandas as pd

        raw = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df = pd.DataFrame(raw, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate()
        df = df.bfill()
        df = df.ffill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]
