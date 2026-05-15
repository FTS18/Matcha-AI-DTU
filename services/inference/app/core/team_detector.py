import cv2
import numpy as np
from typing import Optional, List


def _crop_jersey(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> np.ndarray:
    """Return the torso crop (middle 40% height, inner 60% width) of a person box."""
    h, w = frame.shape[:2]
    bx1, by1 = int(x1 * w), int(y1 * h)
    bx2, by2 = int(x2 * w), int(y2 * h)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 4 or bh < 10:
        return np.array([])
    cy1 = by1 + int(bh * 0.30)
    cy2 = by1 + int(bh * 0.70)
    cx1 = bx1 + int(bw * 0.20)
    cx2 = bx1 + int(bw * 0.80)
    crop = frame[cy1:cy2, cx1:cx2]
    return crop if crop.size else np.array([])


def _dominant_colour(crop: np.ndarray) -> Optional[List[int]]:
    """Return [R, G, B] dominant colour of a BGR crop via median."""
    if crop is None or crop.size < 3:
        return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    return [int(v) for v in np.median(pixels, axis=0).tolist()]


def _cluster_teams(
    colours: list[list[int]], n: int = 2
) -> tuple[list[list[int]], list[int]]:
    """
    K-means cluster colours into n teams.
    Returns (centroids, labels) where centroids = [[R,G,B], ...]
    """
    if len(colours) < n:
        defaults = [[220, 50, 50], [50, 100, 220]]
        return defaults[:n], [i % n for i in range(len(colours))]

    data = np.array(colours, dtype=np.float32)
    centres = data[np.random.choice(len(data), n, replace=False)]

    for _ in range(20):
        dists = np.stack([np.linalg.norm(data - c, axis=1) for c in centres], axis=1)
        labels = np.argmin(dists, axis=1)
        new_c = np.stack(
            [
                data[labels == k].mean(axis=0) if np.any(labels == k) else centres[k]
                for k in range(n)
            ]
        )
        if np.allclose(centres, new_c, atol=1.0):
            break
        centres = new_c

    final_labels = np.argmin(
        np.stack([np.linalg.norm(data - c, axis=1) for c in centres], axis=1), axis=1
    )
    return [[int(v) for v in c.tolist()] for c in centres], final_labels.tolist()
