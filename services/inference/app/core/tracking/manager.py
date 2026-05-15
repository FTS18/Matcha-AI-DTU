import cv2
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class TrackingManager:
    def __init__(self, model: Any, ball_model: Any, config: Dict[str, Any]):
        self.model = model
        self.ball_model = ball_model
        self.config = config
        self.gpu_available = getattr(model, "device", "cpu") != "cpu"

    def process_frame(
        self, frame: np.ndarray, frame_w: int, frame_h: int, cache: Any = None
    ) -> Tuple[List, List, List]:
        """Run YOLO on a single frame and return ball, person, and color data."""
        frame_balls = []
        frame_persons = []
        jersey_colours = []

        target_height = self.config["YOLO_DOWNSCALE_HEIGHT"]
        yolo_frame = frame
        if frame.shape[0] > target_height:
            scale = target_height / frame.shape[0]
            yolo_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Cache check
        cached = (
            cache.get(yolo_frame)
            if cache and self.config["ENABLE_INFERENCE_CACHING"]
            else None
        )
        if cached:
            results, ball_results = cached
        else:
            try:
                device = 0 if self.gpu_available else "cpu"
                results = self.model(yolo_frame, verbose=False, device=device)
                ball_results = self.ball_model(
                    yolo_frame, classes=[32], verbose=False, device=device
                )
                if cache and self.config["ENABLE_INFERENCE_CACHING"]:
                    cache.put(yolo_frame, (results, ball_results))
            except Exception as e:
                logger.warning(f"YOLO failed: {e}")
                return [], [], []

        scale_f = frame.shape[0] / yolo_frame.shape[0]

        # Process Ball
        for r in ball_results:
            if not r.boxes:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = [v * scale_f for v in box.xyxy[0].tolist()]
                frame_balls.append(
                    [
                        round(x1 / frame_w, 4),
                        round(y1 / frame_h, 4),
                        round((x2 - x1) / frame_w, 4),
                        round((y2 - y1) / frame_h, 4),
                        round(float(box.conf[0]), 3),
                    ]
                )

        # Process Persons
        from app.core.team_detector import _crop_jersey, _dominant_colour

        for r in results:
            if not r.boxes:
                continue
            for i, box in enumerate(r.boxes):
                if self.model.names[int(box.cls[0])] != "person":
                    continue

                x1, y1, x2, y2 = [v * scale_f for v in box.xyxy[0].tolist()]
                nx, ny, nw, nh = (
                    x1 / frame_w,
                    y1 / frame_h,
                    (x2 - x1) / frame_w,
                    (y2 - y1) / frame_h,
                )

                tid = (
                    int(box.id[0]) if hasattr(box, "id") and box.id is not None else -1
                )

                crop = _crop_jersey(frame, nx, ny, nx + nw, ny + nh)
                col = _dominant_colour(crop) or [128, 128, 128]
                jersey_colours.append(col)

                kps = []
                if (
                    hasattr(r, "keypoints")
                    and r.keypoints is not None
                    and r.keypoints.xyn is not None
                ):
                    if len(r.keypoints.xyn) > i:
                        kps = [
                            round(float(v), 4)
                            for v in r.keypoints.xyn[i].flatten().tolist()
                        ]

                frame_persons.append(
                    [
                        round(nx, 4),
                        round(ny, 4),
                        round(nw, 4),
                        round(nh, 4),
                        tid,
                        col[0],
                        col[1],
                        col[2],
                    ]
                    + kps
                )

        return frame_balls, frame_persons, jersey_colours

    def refine_tracks(
        self, track_frames: list, jersey_colours: list, team_colors_fallback: list
    ) -> Tuple[list, list]:
        """Clustering teams and smoothing ball trajectory."""
        from app.core.team_detector import _cluster_teams

        team_colors = team_colors_fallback
        if len(jersey_colours) >= 4:
            try:
                centroids, _ = _cluster_teams(jersey_colours, n=2)
                team_colors = centroids
            except Exception as e:
                logger.warning(f"Clustering failed: {e}")

        def _assign_team(r, g, b):
            col = np.array([r, g, b])
            dists = [np.linalg.norm(col - np.array(c)) for c in team_colors]
            return int(np.argmin(dists))

        for tf in track_frames:
            labelled = []
            for p in tf.get("p", []):
                if len(p) >= 8:
                    team = _assign_team(p[5], p[6], p[7])
                    labelled.append(p[:5] + [team])
                elif len(p) >= 5:
                    labelled.append(p[:5] + [0])
                else:
                    labelled.append(p)
            tf["p"] = labelled

        if self.config["ENHANCED_BALL_TRACKING"]:
            from app.core.soccer.spatial import smooth_ball_trajectory

            track_frames = smooth_ball_trajectory(
                track_frames, window_size=self.config["BALL_SMOOTHING_WINDOW"]
            )

        return track_frames, team_colors
