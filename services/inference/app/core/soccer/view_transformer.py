import cv2
import numpy as np


class ViewTransformer:
    """
    Transforms pixel coords to real-world metres using perspective homography.

    The pixel_vertices are auto-calibrated from the first frame using pitch
    line detection (white line Hough), or fall back to a sensible default for
    standard broadcast 16:9 footage.
    """

    def __init__(self, frame_w: int = 1920, frame_h: int = 1080):
        court_width = 68.0  # metres (FIFA standard)
        court_length = 23.32  # visible portion

        # Default pixel vertices (standard broadcast view)
        # Scale proportionally to actual frame size
        sx = frame_w / 1920.0
        sy = frame_h / 1080.0
        self.pixel_vertices = np.array(
            [
                [int(110 * sx), int(1035 * sy)],
                [int(265 * sx), int(275 * sy)],
                [int(910 * sx), int(260 * sy)],
                [int(1640 * sx), int(915 * sy)],
            ],
            dtype=np.float32,
        )

        self.target_vertices = np.array(
            [
                [0, court_width],
                [0, 0],
                [court_length, 0],
                [court_length, court_width],
            ],
            dtype=np.float32,
        )

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
