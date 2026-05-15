import cv2
import numpy as np
from .utils import get_center_of_bbox, get_bbox_width, get_foot_position


def draw_player_box(
    frame: np.ndarray, bbox: list, color: tuple = (0, 0, 255), track_id: int = None
) -> np.ndarray:
    """Draw a red bounding-box rectangle around a player with optional track ID label."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    if abs(x2 - x1) < 3 or abs(y2 - y1) < 3:
        return frame

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if track_id is not None:
        label = str(track_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        lbl_x1 = x1
        lbl_y1 = max(y1 - th - 8, 0)
        lbl_x2 = x1 + tw + 8
        lbl_y2 = y1
        cv2.rectangle(frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, cv2.FILLED)
        cv2.putText(
            frame,
            label,
            (lbl_x1 + 4, lbl_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return frame


def draw_ellipse(
    frame: np.ndarray, bbox: list, color: tuple, track_id: int = None
) -> np.ndarray:
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

        cv2.rectangle(
            frame, (int(x1_r), int(y1_r)), (int(x2_r), int(y2_r)), color, cv2.FILLED
        )

        x1_text = x1_r + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_r + 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
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
    frame: np.ndarray,
    frame_num: int,
    team_ball_control: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    """Draw a semi-transparent ball-control overlay in the bottom-right."""
    overlay = frame.copy()
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
        frame,
        f"Team 1 Ball Control: {t1_pct * 100:.1f}%",
        (x1 + 10, y1 + int((y2 - y1) * 0.4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Team 2 Ball Control: {t2_pct * 100:.1f}%",
        (x1 + 10, y1 + int((y2 - y1) * 0.8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        2,
    )
    return frame


def draw_speed_and_distance(
    frame: np.ndarray, tracks: dict, frame_num: int
) -> np.ndarray:
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
            cv2.putText(
                frame,
                f"{speed:.2f} km/h",
                pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"{distance:.2f} m",
                (pos[0], pos[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
    return frame
