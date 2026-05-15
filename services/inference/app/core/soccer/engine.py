import logging
from typing import List, Optional
import numpy as np
from .tracker import SoccerTracker
from .camera import CameraMovementEstimator
from .view_transformer import ViewTransformer
from .estimators import SpeedAndDistanceEstimator
from .team_assigner import TeamAssigner
from .ball_assigner import PlayerBallAssigner
from .draw_utils import (
    draw_player_box,
    draw_triangle,
    draw_ellipse,
    draw_speed_and_distance,
    draw_team_ball_control,
)

logger = logging.getLogger(__name__)


def _check_supervision():
    try:
        import supervision  # noqa: F401

        return True
    except ImportError:
        return False


def _check_sklearn():
    try:
        from sklearn.cluster import KMeans  # noqa: F401

        return True
    except ImportError:
        return False


def is_available() -> bool:
    """Check if all dependencies for soccer analysis overlay are present."""
    return _check_supervision() and _check_sklearn()


def process_clip_frames(
    frames: List[np.ndarray],
    fps: float = 24.0,
    custom_model_path: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Full Soccer_Analysis pipeline on a list of BGR frames.
    """
    if not frames:
        return frames

    if not is_available():
        logger.warning("Skipping soccer analysis overlay — dependencies not available")
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
                    info["team_color"] = tuple(
                        int(c)
                        for c in team_assigner.team_colors.get(team, (128, 128, 128))
                    )

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
                out = draw_player_box(
                    out, player["bbox"], color=(0, 0, 255), track_id=tid
                )
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
                out = draw_team_ball_control(
                    out, fn, team_ball_control, frame_w, frame_h
                )

            output_frames.append(out)

        logger.info(
            f"Soccer analysis overlay complete: {len(output_frames)} frames annotated"
        )
        return output_frames

    except Exception as e:
        logger.error(f"Soccer analysis pipeline failed: {e}", exc_info=True)
        return frames
