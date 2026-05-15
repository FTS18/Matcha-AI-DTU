import cv2
import numpy as np
from .utils import measure_distance, measure_xy_distance


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
        mask_features[:, 0 : min(20, w)] = 1
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

            old_gray = frame_gray.copy()

        return camera_movement
