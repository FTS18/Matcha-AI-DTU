"""
Soccer Analysis Engine — adapted from Soccer_Analysis/football_analysis.

Provides broadcast-quality overlays for highlight videos:
  • Player tracking with team-coloured ellipses & jersey numbers
  • Ball tracking with green triangle marker
  • Referee tracking with yellow ellipses
  • Per-player speed (km/h) and distance (m)
  • Ball control percentage per team
  • Camera movement compensation & perspective transform

This module is designed to process highlight *clips* (typically 10-30s each)
rather than full-match video, so it runs fresh YOLO detection on each clip
instead of relying on pre-baked stubs.
"""

import cv2
import logging
import numpy as np
import os
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# ── Lazy-loaded heavy dependencies ─────────────────────────────────────────
_supervision_available: Optional[bool] = None
_sklearn_available: Optional[bool] = None

def _check_supervision():
    global _supervision_available
    if _supervision_available is None:
        try:
            import supervision  # noqa: F401
            _supervision_available = True
        except ImportError:
            _supervision_available = False
            logger.warning("supervision not installed — soccer analysis overlay disabled")
    return _supervision_available

def _check_sklearn():
    global _sklearn_available
    if _sklearn_available is None:
        try:
            from sklearn.cluster import KMeans  # noqa: F401
            _sklearn_available = True
        except ImportError:
            _sklearn_available = False
            logger.warning("scikit-learn not installed — team colour clustering disabled")
    return _sklearn_available


# ═══════════════════════════════════════════════════════════════════════════
#  Geometry / Bbox Utilities  (from Soccer_Analysis/utils/bbox_utils.py)
# ═══════════════════════════════════════════════════════════════════════════

def get_center_of_bbox(bbox: list) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_bbox_width(bbox: list) -> float:
    return bbox[2] - bbox[0]

def get_foot_position(bbox: list) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)

def measure_distance(p1, p2) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def measure_xy_distance(p1, p2) -> Tuple[float, float]:
    return p1[0] - p2[0], p1[1] - p2[1]


# ═══════════════════════════════════════════════════════════════════════════
#  Team Assigner  (from Soccer_Analysis/team_assigner)
# ═══════════════════════════════════════════════════════════════════════════

class TeamAssigner:
    """K-Means based team colour assignment from jersey crops."""

    def __init__(self):
        self.team_colors: Dict[int, Any] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans = None

    def _get_clustering_model(self, image: np.ndarray):
        from sklearn.cluster import KMeans
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if image.size == 0:
            return np.array([128, 128, 128])
        top_half = image[0:int(image.shape[0] / 2), :]
        if top_half.size == 0:
            return np.array([128, 128, 128])
        kmeans = self._get_clustering_model(top_half)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [
            clustered_image[0, 0], clustered_image[0, -1],
            clustered_image[-1, 0], clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame: np.ndarray, player_detections: dict):
        from sklearn.cluster import KMeans
        player_colors = []
        for _, det in player_detections.items():
            bbox = det["bbox"]
            try:
                c = self.get_player_color(frame, bbox)
                player_colors.append(c)
            except Exception:
                pass
        if len(player_colors) < 2:
            self.team_colors = {1: np.array([220, 60, 60]), 2: np.array([60, 100, 220])}
            return
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame: np.ndarray, player_bbox: list, player_id: int) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        if self.kmeans is None:
            return 1
        try:
            player_color = self.get_player_color(frame, player_bbox)
            team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        except Exception:
            team_id = 1
        self.player_team_dict[player_id] = team_id
        return team_id


# ═══════════════════════════════════════════════════════════════════════════
#  Player-Ball Assigner  (from Soccer_Analysis/player_ball_assigner)
# ═══════════════════════════════════════════════════════════════════════════

class PlayerBallAssigner:
    def __init__(self, max_distance: float = 70.0):
        self.max_player_ball_distance = max_distance

    def assign_ball_to_player(self, players: dict, ball_bbox: list) -> int:
        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = 99999.0
        assigned_player = -1
        for player_id, player in players.items():
            player_bbox = player["bbox"]
            dist_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            dist_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(dist_left, dist_right)
            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id
        return assigned_player


# ═══════════════════════════════════════════════════════════════════════════
#  Camera Movement Estimator  (from Soccer_Analysis/camera_movement_estimator)
# ═══════════════════════════════════════════════════════════════════════════

class CameraMovementEstimator:
    def __init__(self, first_frame: np.ndarray):
        self.minimum_distance = 5.0
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        h, w = first_gray.shape
        mask_features = np.zeros_like(first_gray)
        # Use left and right edges for feature tracking
        mask_features[:, 0:min(20, w)] = 1
        edge_start = max(0, w - 150)
        mask_features[:, edge_start:w] = 1
        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def get_camera_movement(self, frames: list) -> list:
        camera_movement = [[0.0, 0.0]] * len(frames)
        if not frames:
            return camera_movement
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        if old_features is None:
            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )
            if new_features is None:
                old_gray = frame_gray.copy()
                continue

            max_dist = 0.0
            cam_x, cam_y = 0.0, 0.0
            for new, old in zip(new_features, old_features):
                new_pt = new.ravel()
                old_pt = old.ravel()
                d = measure_distance(new_pt, old_pt)
                if d > max_dist:
                    max_dist = d
                    cam_x, cam_y = measure_xy_distance(old_pt, new_pt)

            if max_dist > self.minimum_distance:
                camera_movement[frame_num] = [cam_x, cam_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
                if old_features is None:
                    old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

            old_gray = frame_gray.copy()

        return camera_movement


# ═══════════════════════════════════════════════════════════════════════════
#  View Transformer  (from Soccer_Analysis/view_transformer)
# ═══════════════════════════════════════════════════════════════════════════

class ViewTransformer:
    """
    Transforms pixel coords to real-world metres using perspective homography.

    The pixel_vertices are auto-calibrated from the first frame using pitch
    line detection (white line Hough), or fall back to a sensible default for
    standard broadcast 16:9 footage.
    """

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080):
        court_width = 68.0   # metres (FIFA standard)
        court_length = 23.32  # visible portion

        # Default pixel vertices (standard broadcast view)
        # Scale proportionally to actual frame size
        sx = frame_w / 1920.0
        sy = frame_h / 1080.0
        self.pixel_vertices = np.array([
            [int(110 * sx), int(1035 * sy)],
            [int(265 * sx), int(275 * sy)],
            [int(910 * sx), int(260 * sy)],
            [int(1640 * sx), int(915 * sy)],
        ], dtype=np.float32)

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width],
        ], dtype=np.float32)

        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point: np.ndarray):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None
        reshaped = point.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.perspective_transform)
        return transformed.reshape(-1, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  Speed & Distance Estimator  (from Soccer_Analysis/speed_and_distance_estimator)
# ═══════════════════════════════════════════════════════════════════════════

class SpeedAndDistanceEstimator:
    def __init__(self, frame_rate: float = 24.0):
        self.frame_window = 5
        self.frame_rate = max(1.0, frame_rate)

    def add_speed_and_distance_to_tracks(self, tracks: dict):
        total_distance: Dict[str, Dict[int, float]] = {}
        for obj_name, obj_tracks in tracks.items():
            if obj_name in ("ball", "referees"):
                continue
            n_frames = len(obj_tracks)
            for frame_num in range(0, n_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, n_frames - 1)
                for track_id in obj_tracks[frame_num]:
                    if track_id not in obj_tracks[last_frame]:
                        continue
                    start_pos = obj_tracks[frame_num][track_id].get("position_transformed")
                    end_pos = obj_tracks[last_frame][track_id].get("position_transformed")
                    if start_pos is None or end_pos is None:
                        continue
                    dist = measure_distance(start_pos, end_pos)
                    elapsed = (last_frame - frame_num) / self.frame_rate
                    if elapsed <= 0:
                        continue
                    speed_kmh = (dist / elapsed) * 3.6

                    if obj_name not in total_distance:
                        total_distance[obj_name] = {}
                    if track_id not in total_distance[obj_name]:
                        total_distance[obj_name][track_id] = 0.0
                    total_distance[obj_name][track_id] += dist

                    for fn in range(frame_num, last_frame):
                        if track_id not in tracks[obj_name][fn]:
                            continue
                        tracks[obj_name][fn][track_id]["speed"] = speed_kmh
                        tracks[obj_name][fn][track_id]["distance"] = total_distance[obj_name][track_id]


# ═══════════════════════════════════════════════════════════════════════════
#  YOLO Tracker (from Soccer_Analysis/trackers/tracker.py)
# ═══════════════════════════════════════════════════════════════════════════

class SoccerTracker:
    """
    Wraps YOLO detection + ByteTrack for consistent player/ball/referee tracking.
    Uses the same YOLO model already loaded by the Matcha inference service
    (yolov8n), or loads a custom football-trained model if available.
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
            batch = self.model.predict(frames[i:i + batch_size], conf=0.1, verbose=False)
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
                    position = get_center_of_bbox(bbox) if obj_name == "ball" else get_foot_position(bbox)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Drawing Functions
# ═══════════════════════════════════════════════════════════════════════════

def draw_player_box(frame: np.ndarray, bbox: list, color: tuple = (0, 0, 255), track_id: int = None) -> np.ndarray:
    """Draw a red bounding-box rectangle around a player with optional track ID label."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
        return frame

    # Red rectangle around the player
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if track_id is not None:
        # Label background
        label = str(track_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lbl_x1 = x1
        lbl_y1 = max(y1 - th - 8, 0)
        lbl_x2 = x1 + tw + 8
        lbl_y2 = y1
        cv2.rectangle(frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, cv2.FILLED)
        cv2.putText(
            frame, label,
            (lbl_x1 + 4, lbl_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return frame


def draw_ellipse(frame: np.ndarray, bbox: list, color: tuple, track_id: int = None) -> np.ndarray:
    """Draw the characteristic Soccer_Analysis ellipse under a player."""
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    if width < 5:
        return frame

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    if track_id is not None:
        rect_w, rect_h = 40, 20
        x1_r = x_center - rect_w // 2
        x2_r = x_center + rect_w // 2
        y1_r = y2 - rect_h // 2 + 15
        y2_r = y2 + rect_h // 2 + 15

        cv2.rectangle(frame, (int(x1_r), int(y1_r)), (int(x2_r), int(y2_r)), color, cv2.FILLED)

        x1_text = x1_r + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame, f"{track_id}",
            (int(x1_text), int(y1_r + 15)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
        )

    return frame


def draw_triangle(frame: np.ndarray, bbox: list, color: tuple) -> np.ndarray:
    """Draw a triangle marker above an object (ball / ball carrier)."""
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)
    pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
    cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
    return frame


def draw_team_ball_control(
    frame: np.ndarray, frame_num: int, team_ball_control: np.ndarray,
    frame_w: int, frame_h: int,
) -> np.ndarray:
    """Draw a semi-transparent ball-control overlay in the bottom-right."""
    overlay = frame.copy()
    # Position relative to frame size
    x1 = int(frame_w * 0.70)
    y1 = int(frame_h * 0.88)
    x2 = frame_w - 10
    y2 = frame_h - 10
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    ctrl = team_ball_control[: frame_num + 1]
    t1_count = int(np.sum(ctrl == 1))
    t2_count = int(np.sum(ctrl == 2))
    total = t1_count + t2_count
    if total == 0:
        return frame
    t1_pct = t1_count / total
    t2_pct = t2_count / total

    font_scale = max(0.45, min(0.8, frame_w / 1600))
    cv2.putText(
        frame, f"Team 1 Ball Control: {t1_pct * 100:.1f}%",
        (x1 + 10, y1 + int((y2 - y1) * 0.4)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2,
    )
    cv2.putText(
        frame, f"Team 2 Ball Control: {t2_pct * 100:.1f}%",
        (x1 + 10, y1 + int((y2 - y1) * 0.8)),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2,
    )
    return frame


def draw_speed_and_distance(frame: np.ndarray, tracks: dict, frame_num: int) -> np.ndarray:
    """Overlay speed (km/h) and distance (m) below each tracked player."""
    for obj_name, obj_tracks in tracks.items():
        if obj_name in ("ball", "referees"):
            continue
        if frame_num >= len(obj_tracks):
            continue
        for _, track_info in obj_tracks[frame_num].items():
            speed = track_info.get("speed")
            distance = track_info.get("distance")
            if speed is None or distance is None:
                continue
            bbox = track_info["bbox"]
            pos = list(get_foot_position(bbox))
            pos[1] += 40
            pos = tuple(map(int, pos))
            cv2.putText(frame, f"{speed:.2f} km/h", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, f"{distance:.2f} m", (pos[0], pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame


# ═══════════════════════════════════════════════════════════════════════════
#  Main Pipeline – process a clip & return annotated frames
# ═══════════════════════════════════════════════════════════════════════════

def process_clip_frames(
    frames: List[np.ndarray],
    fps: float = 24.0,
    custom_model_path: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Full Soccer_Analysis pipeline on a list of BGR frames.

    Returns annotated frames with:
     – Player ellipses (team-coloured) with jersey numbers
     – Ball triangle (green)
     – Referee ellipses (yellow)
     – Speed & distance labels per player
     – Ball control % overlay

    This is called by video_utils.create_highlight_reel for each clip
    *before* FFmpeg encoding.
    """
    if not frames:
        return frames

    if not _check_supervision():
        logger.warning("Skipping soccer analysis overlay — supervision not available")
        return frames

    if not _check_sklearn():
        logger.warning("Skipping soccer analysis overlay — scikit-learn not available")
        return frames

    try:
        frame_h, frame_w = frames[0].shape[:2]

        # ── 1) Track players, referees, ball ─────────────────────────────
        logger.info(f"Soccer analysis: tracking {len(frames)} frames at {fps} fps")
        tracker = SoccerTracker(model_path=custom_model_path)
        tracks = tracker.get_object_tracks(frames)

        # ── 2) Add foot/centre positions ─────────────────────────────────
        tracker.add_position_to_tracks(tracks)

        # ── 3) Camera movement estimation & adjust positions ─────────────
        cam_estimator = CameraMovementEstimator(frames[0])
        camera_movement = cam_estimator.get_camera_movement(frames)

        # Adjust positions for camera movement
        for obj_name, obj_tracks in tracks.items():
            for fn, track in enumerate(obj_tracks):
                for tid, info in track.items():
                    if "position" in info:
                        pos = info["position"]
                        cm = camera_movement[fn]
                        info["position_adjusted"] = (pos[0] - cm[0], pos[1] - cm[1])
                    else:
                        info["position_adjusted"] = info.get("position", (0, 0))

        # ── 4) View transformer (pixel → metres) ────────────────────────
        view_tf = ViewTransformer(frame_w, frame_h)
        for obj_name, obj_tracks in tracks.items():
            for fn, track in enumerate(obj_tracks):
                for tid, info in track.items():
                    pos_adj = info.get("position_adjusted", (0, 0))
                    pos_arr = np.array(pos_adj)
                    transformed = view_tf.transform_point(pos_arr)
                    if transformed is not None:
                        info["position_transformed"] = transformed.squeeze().tolist()
                    else:
                        info["position_transformed"] = None

        # ── 5) Interpolate ball positions ────────────────────────────────
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

        # ── 6) Speed & distance estimation ───────────────────────────────
        speed_estimator = SpeedAndDistanceEstimator(frame_rate=fps)
        speed_estimator.add_speed_and_distance_to_tracks(tracks)

        # ── 7) Team assignment ───────────────────────────────────────────
        team_assigner = TeamAssigner()
        if tracks["players"] and tracks["players"][0]:
            team_assigner.assign_team_color(frames[0], tracks["players"][0])
            for fn, player_track in enumerate(tracks["players"]):
                for pid, info in player_track.items():
                    team = team_assigner.get_player_team(frames[fn], info["bbox"], pid)
                    info["team"] = team
                    info["team_color"] = tuple(int(c) for c in team_assigner.team_colors.get(team, (128, 128, 128)))

        # ── 8) Ball possession assignment ────────────────────────────────
        ball_assigner = PlayerBallAssigner()
        team_ball_control_list: list = []
        for fn, player_track in enumerate(tracks["players"]):
            ball_data = tracks["ball"][fn].get(1, {})
            ball_bbox = ball_data.get("bbox", [])
            if ball_bbox:
                assigned = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
                if assigned != -1:
                    tracks["players"][fn][assigned]["has_ball"] = True
                    team_ball_control_list.append(
                        tracks["players"][fn][assigned].get("team", 1)
                    )
                else:
                    team_ball_control_list.append(
                        team_ball_control_list[-1] if team_ball_control_list else 1
                    )
            else:
                team_ball_control_list.append(
                    team_ball_control_list[-1] if team_ball_control_list else 1
                )
        team_ball_control = np.array(team_ball_control_list)

        # ── 9) Draw annotations on every frame ──────────────────────────
        output_frames: List[np.ndarray] = []
        for fn, frame in enumerate(frames):
            out = frame.copy()

            # Draw players — red bounding boxes
            player_dict = tracks["players"][fn] if fn < len(tracks["players"]) else {}
            for tid, player in player_dict.items():
                out = draw_player_box(out, player["bbox"], color=(0, 0, 255), track_id=tid)
                if player.get("has_ball", False):
                    out = draw_triangle(out, player["bbox"], (0, 0, 255))

            # Draw referees
            ref_dict = tracks["referees"][fn] if fn < len(tracks["referees"]) else {}
            for _, referee in ref_dict.items():
                out = draw_ellipse(out, referee["bbox"], (0, 255, 255))

            # Draw ball
            ball_dict = tracks["ball"][fn] if fn < len(tracks["ball"]) else {}
            for _, ball in ball_dict.items():
                out = draw_triangle(out, ball["bbox"], (0, 255, 0))

            # Draw speed & distance
            out = draw_speed_and_distance(out, tracks, fn)

            # Draw team ball control
            if len(team_ball_control) > 0:
                out = draw_team_ball_control(out, fn, team_ball_control, frame_w, frame_h)

            output_frames.append(out)

        logger.info(f"Soccer analysis overlay complete: {len(output_frames)} frames annotated")
        return output_frames

    except Exception as e:
        logger.error(f"Soccer analysis pipeline failed: {e}", exc_info=True)
        return frames  # Return original frames on failure — never break the highlight reel


def is_available() -> bool:
    """Check if all dependencies for soccer analysis overlay are present."""
    return _check_supervision() and _check_sklearn()
