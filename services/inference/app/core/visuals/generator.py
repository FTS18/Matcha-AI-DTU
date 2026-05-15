import cv2
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class VisualsManager:
    def __init__(self, match_id: str, uploads_dir: Path):
        self.match_id = match_id
        self.uploads_dir = uploads_dir

    def generate_thumbnail(self, video_path: str, total_frames: int) -> Optional[str]:
        """Extract a midpoint frame as thumbnail."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            midpoint = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
            ret, frame = cap.read()

            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret:
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (1280, int(h * scale)))

                filename = f"thumbnail_{self.match_id}.jpg"
                path = self.uploads_dir / filename
                cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cap.release()
                return f"/uploads/{filename}"

            cap.release()
        except Exception as e:
            logger.warning(f"Thumbnail generation failed: {e}")
        return None

    def generate_heatmap(
        self, track_frames: list, team_colors: list, heatmap_fn: Any
    ) -> Optional[str]:
        """Generate intensity heatmap."""
        if not heatmap_fn or not track_frames:
            return None
        try:
            filename = f"heatmap_{self.match_id}.png"
            path = str(self.uploads_dir / filename)
            if heatmap_fn(
                track_frames=track_frames, output_path=path, team_colors_rgb=team_colors
            ):
                return f"/uploads/{filename}"
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
        return None

    def generate_tactical_stats(
        self,
        track_frames: list,
        fps: float,
        team_colors: list,
        metrics_fn: Any = None,
        possession_fn: Any = None,
        dominance_fn: Any = None,
        radar_fn: Any = None,
    ) -> Dict[str, Any]:
        """Run advanced tactical analysis."""
        stats = {}
        if not track_frames:
            return stats

        try:
            if metrics_fn:
                stats["playerMetrics"] = metrics_fn(track_frames, fps)
            if possession_fn:
                stats["possession"] = possession_fn(track_frames)
            if dominance_fn:
                stats["dominance"] = dominance_fn(track_frames)
            if radar_fn:
                radar_filename = f"radar_{self.match_id}.png"
                radar_path = str(self.uploads_dir / radar_filename)
                radar_fn(track_frames, radar_path, team_colors)
                stats["radarUrl"] = f"/uploads/{radar_filename}"
        except Exception as e:
            logger.warning(f"Tactical analysis failed: {e}")

        return stats

    def generate_context_data(
        self,
        track_frames: list,
        team_colors: list,
        fps: float,
        formation_fn: Any = None,
        trajectory_fn: Any = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Analyze team formation and ball trajectory."""
        formation_data = {}
        trajectory_data = {}

        try:
            if formation_fn:
                formation_data = formation_fn(track_frames, team_colors)

            if trajectory_fn:
                all_balls = []
                for frame in track_frames:
                    if frame.get("b"):
                        all_balls.extend(frame["b"])
                if all_balls:
                    trajectory_data = trajectory_fn(all_balls, fps=fps)
        except Exception as e:
            logger.warning(f"Context analysis failed: {e}")

        return formation_data, trajectory_data
