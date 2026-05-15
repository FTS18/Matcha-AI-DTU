import logging
import numpy as np

logger = logging.getLogger(__name__)


def smooth_ball_trajectory(track_frames: list, window_size: int = 3) -> list:
    """Smoothing ball positions using a sliding window average to reduce jitter."""
    if not track_frames or window_size < 1:
        return track_frames

    smoothed = []
    for i, frame in enumerate(track_frames):
        if not frame.get("b") or len(frame["b"]) == 0:
            smoothed.append(frame)
            continue

        start = max(0, i - window_size // 2)
        end = min(len(track_frames), i + window_size // 2 + 1)
        window_frames = track_frames[start:end]

        all_balls = []
        for wf in window_frames:
            all_balls.extend(wf.get("b", []))

        if all_balls:
            smoothed_ball = [
                round(np.mean([b[j] for b in all_balls]), 4)
                for j in range(len(all_balls[0]))
            ]
            frame_copy = frame.copy()
            frame_copy["b"] = [smoothed_ball[:4]]
            smoothed.append(frame_copy)
        else:
            smoothed.append(frame)

    return smoothed


def predict_ball_trajectory(balls: list, fps: float = 30.0) -> dict:
    """Predict ball movement trajectory based on historical positions."""
    if len(balls) < 2:
        return {"direction": "unknown", "speed": 0.0, "confidence": 0.0}

    try:
        recent = balls[-3:] if len(balls) >= 3 else balls
        if len(recent) < 2:
            return {"direction": "unknown", "speed": 0.0, "confidence": 0.0}

        x_vel = recent[-1][0] - recent[-2][0]
        y_vel = recent[-1][1] - recent[-2][1]

        speed = np.sqrt(x_vel**2 + y_vel**2)
        direction = np.degrees(np.arctan2(y_vel, x_vel))
        confidence = min(speed * 2, 1.0)

        return {
            "direction": f"{direction:.1f}°",
            "speed": round(speed, 4),
            "confidence": round(confidence, 3),
            "velocity": [round(x_vel, 4), round(y_vel, 4)],
        }
    except Exception as e:
        logger.debug(f"Trajectory prediction failed: {e}")
        return {"direction": "unknown", "speed": 0.0, "confidence": 0.0}


def analyze_team_formation(track_frames: list, team_colors: list) -> dict:
    """Analyze team formation and positioning from tracking data."""
    if not track_frames:
        return {"formation": "unknown", "spacing": 0.0, "cohesion": 0.0}

    try:
        recent_frames = track_frames[-5:]
        all_positions = []

        for frame in recent_frames:
            for person in frame.get("p", [])[:11]:
                if len(person) >= 3:
                    all_positions.append((person[0], person[1]))

        if len(all_positions) < 4:
            return {"formation": "unknown", "spacing": 0.0, "cohesion": 0.0}

        positions = np.array(all_positions)
        distances = np.sqrt(
            ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
        )
        spacing = np.mean(distances[distances > 0.01])
        cohesion = 1.0 / (1.0 + spacing)

        if spacing < 0.15:
            formation = "compact"
        elif spacing < 0.25:
            formation = "balanced"
        else:
            formation = "spread"

        return {
            "formation": formation,
            "spacing": round(spacing, 3),
            "cohesion": round(cohesion, 3),
            "player_count": len(all_positions),
        }
    except Exception as e:
        logger.debug(f"Formation analysis failed: {e}")
        return {"formation": "unknown", "spacing": 0.0, "cohesion": 0.0}
