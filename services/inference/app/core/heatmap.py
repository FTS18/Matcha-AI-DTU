"""
heatmap.py — Player Heatmap & Ball Speed Estimation
=====================================================
This module post-processes the `track_frames` list (already collected during
the main YOLO loop in analysis.py) to produce:

 1. A pitch heatmap PNG saved to disk (uses OpenCV + NumPy — no matplotlib).
 2. A top ball speed estimate in km/h (uses consecutive ball positions + FPS).

Both outputs are attached to the orchestrator completion payload.
"""

import cv2
import numpy as np
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Pitch palette constants ───────────────────────────────────────────────────
PITCH_W, PITCH_H = 800, 520 # pixels for the output heatmap image
TEAM_COLORS_BGR = {
 0: (60, 220, 60), # Team A — green
 1: (60, 60, 220), # Team B — red (BGR)
 "ball": (0, 200, 255),
}

# Calibration: assumes a 105m × 68m pitch maps to the video frame
# 1 pixel ≈ 0.13 m at 800px wide (105/800)
PIXELS_PER_METRE = 800 / 105.0 # ~7.6 px/m

# Approximate broadcast pitch polygon (normalized coords)
SRC_POLY_NORM = np.array([
 [0.10, 0.35], # top-left
 [0.90, 0.35], # top-right
 [1.00, 0.95], # bottom-right
 [0.00, 0.95] # bottom-left
], dtype=np.float32)

# Top-down pitch rectangle corners
DST_POLY = np.array([
 [0, 0],
 [PITCH_W, 0],
 [PITCH_W, PITCH_H],
 [0, PITCH_H]
], dtype=np.float32)

# Calculate fixed Homography matrix
H_SRC = SRC_POLY_NORM.copy()
H_MATRIX, _ = cv2.findHomography(H_SRC, DST_POLY)


def generate_heatmap(track_frames: list, output_path: str,
 team_colors_rgb: Optional[List] = None) -> bool:
 """
 Build a combined player position heatmap PNG from YOLO tracking data.

 Parameters
 ----------
 track_frames : list of dicts with keys "t", "b" (balls), "p" (persons)
 Each person entry: [nx, ny, nw, nh, track_id, team_idx]
 output_path : absolute path to write the PNG file
 team_colors_rgb : optional [[R,G,B], [R,G,B]] for team 0 and team 1

 Returns True on success.
 """
 if not track_frames:
 logger.warning("generate_heatmap: no tracking frames available")
 return False

 # ── Accumulate density grids per team ────────────────────────────────────
 grid_a = np.zeros((PITCH_H, PITCH_W), dtype=np.float32)
 grid_b = np.zeros((PITCH_H, PITCH_W), dtype=np.float32)
 ball_positions: List[Tuple[int, int]] = []

 for tf in track_frames:
 for p in tf.get("p", []):
 if len(p) < 6:
 continue
 nx, ny, nw, nh = float(p[0]), float(p[1]), float(p[2]), float(p[3])
 team = int(p[5]) if len(p) > 5 else 0

 # Use bottom-center of bounding box for ground position
 px = nx + nw / 2
 py = ny + nh
 
 # Map using homography
 pt = np.array([[[px, py]]], dtype=np.float32)
 td = cv2.perspectiveTransform(pt, H_MATRIX)[0][0]
 cx, cy = int(td[0]), int(td[1])
 
 cx = max(0, min(PITCH_W - 1, cx))
 cy = max(0, min(PITCH_H - 1, cy))

 if team == 0:
 grid_a[cy, cx] += 1
 else:
 grid_b[cy, cx] += 1

 for b in tf.get("b", []):
 if len(b) < 4:
 continue
 nx, ny, nw, nh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
 px = nx + nw / 2
 py = ny + nh / 2
 pt = np.array([[[px, py]]], dtype=np.float32)
 td = cv2.perspectiveTransform(pt, H_MATRIX)[0][0]
 cx, cy = int(td[0]), int(td[1])
 cx = max(0, min(PITCH_W - 1, cx))
 cy = max(0, min(PITCH_H - 1, cy))
 ball_positions.append((cx, cy))

 if grid_a.max() == 0 and grid_b.max() == 0:
 logger.warning("generate_heatmap: all grids empty — no detections")
 return False

 # ── Gaussian blur for smooth density ─────────────────────────────────────
 blur_k = 51 # kernel size — must be odd
 grid_a = cv2.GaussianBlur(grid_a, (blur_k, blur_k), 0)
 grid_b = cv2.GaussianBlur(grid_b, (blur_k, blur_k), 0)

 # ── Determine team colours ────────────────────────────────────────────────
 if team_colors_rgb and len(team_colors_rgb) >= 2:
 ra, ga, ba = team_colors_rgb[0]
 rb, gb, bb = team_colors_rgb[1]
 col_a = (int(ba), int(ga), int(ra)) # convert RGB → BGR
 col_b = (int(bb), int(gb), int(rb))
 else:
 col_a = (60, 220, 60) # green (BGR)
 col_b = (60, 60, 220) # red

 # ── Draw pitch background ─────────────────────────────────────────────────
 canvas = _draw_pitch(PITCH_W, PITCH_H)

 # ── Overlay heatmap for each team ─────────────────────────────────────────
 for grid, color in [(grid_a, col_a), (grid_b, col_b)]:
 if grid.max() == 0:
 continue
 norm = grid / grid.max()

 # Coloured semi-transparent overlay
 heat_color = np.zeros((PITCH_H, PITCH_W, 3), dtype=np.uint8)
 heat_color[:, :] = color # solid colour fill
 alpha_ch = (norm * 200).astype(np.uint8) # 0-200 opacity

 mask = np.stack([alpha_ch, alpha_ch, alpha_ch], axis=-1).astype(np.float32) / 255.0
 canvas = (canvas.astype(np.float32) * (1 - mask) +
 heat_color.astype(np.float32) * mask).astype(np.uint8)

 # ── Draw ball trail ───────────────────────────────────────────────────────
 for i, (bx, by) in enumerate(ball_positions):
 radius = max(2, min(8, len(ball_positions) // 20 + 2))
 alpha = max(0.15, i / max(len(ball_positions), 1))
 overlay = canvas.copy()
 cv2.circle(overlay, (bx, by), radius, (0, 200, 255), -1)
 cv2.addWeighted(overlay, alpha * 0.4, canvas, 1 - alpha * 0.4, 0, canvas)

 # ── Legend ────────────────────────────────────────────────────────────────
 _draw_legend(canvas, col_a, col_b)

 # ── Save ──────────────────────────────────────────────────────────────────
 try:
 Path(output_path).parent.mkdir(parents=True, exist_ok=True)
 cv2.imwrite(output_path, canvas)
 logger.info(f"Heatmap saved → {output_path}")
 return True
 except Exception as e:
 logger.error(f"Heatmap save failed: {e}")
 return False


def _draw_pitch(w: int, h: int) -> np.ndarray:
 """Draw a simple top-down football pitch on a dark green canvas."""
 canvas = np.full((h, w, 3), (30, 80, 30), dtype=np.uint8) # dark green
 lc = (200, 230, 200) # line colour (light green-white)
 t = 2 # line thickness

 # Outer boundary
 cv2.rectangle(canvas, (10, 10), (w - 10, h - 10), lc, t)
 # Centre line
 cv2.line(canvas, (w // 2, 10), (w // 2, h - 10), lc, t)
 # Centre circle
 cv2.circle(canvas, (w // 2, h // 2), h // 5, lc, t)
 # Centre spot
 cv2.circle(canvas, (w // 2, h // 2), 4, lc, -1)

 # Penalty areas
 pa_w, pa_h = int(w * 0.14), int(h * 0.50)
 pa_top = (h - pa_h) // 2
 # Left
 cv2.rectangle(canvas, (10, pa_top), (10 + pa_w, pa_top + pa_h), lc, t)
 # Right
 cv2.rectangle(canvas, (w - 10 - pa_w, pa_top), (w - 10, pa_top + pa_h), lc, t)

 # 6-yard boxes
 ga_w, ga_h = int(w * 0.045), int(h * 0.23)
 ga_top = (h - ga_h) // 2
 cv2.rectangle(canvas, (10, ga_top), (10 + ga_w, ga_top + ga_h), lc, t)
 cv2.rectangle(canvas, (w - 10 - ga_w, ga_top), (w - 10, ga_top + ga_h), lc, t)

 return canvas


def _draw_legend(canvas: np.ndarray, col_a: tuple, col_b: tuple) -> None:
 """Draw a small legend in the corner."""
 h, w = canvas.shape[:2]
 pad = 10
 box_size = 14
 font = cv2.FONT_HERSHEY_SIMPLEX
 scale = 0.45
 thickness = 1

 labels = [("Team A", col_a), ("Team B", col_b), ("Ball", (0, 200, 255))]
 for i, (label, color) in enumerate(labels):
 y = h - pad - (len(labels) - i - 1) * (box_size + 6)
 cv2.rectangle(canvas, (pad, y - box_size + 4), (pad + box_size, y + 4), color, -1)
 cv2.putText(canvas, label, (pad + box_size + 6, y + 2),
 font, scale, (230, 230, 230), thickness, cv2.LINE_AA)


# ── Ball speed estimation ─────────────────────────────────────────────────────

def estimate_ball_speed(track_frames: list, fps: float) -> float:
 """
 Estimate top ball speed in km/h from YOLO ball positions across frames.

 Uses consecutive ball detections to compute pixel displacement per second,
 then converts to km/h via a rough calibration (pitch width = 105 metres).

 Returns the 95th-percentile speed to prevent noise spikes from dominating.
 """
 if not track_frames or fps <= 0:
 return 0.0

 positions: List[Tuple[float, float, float]] = [] # (t, cx_norm, cy_norm)

 for tf in track_frames:
 t = float(tf.get("t", 0))
 balls = tf.get("b", [])
 if balls:
 b = balls[0] # take the primary ball
 if len(b) >= 4:
 cx = float(b[0]) + float(b[2]) / 2
 cy = float(b[1]) + float(b[3]) / 2
 positions.append((t, cx, cy))

 if len(positions) < 2:
 return 0.0

 speeds_mps: list[float] = []

 for i in range(1, len(positions)):
 t1, x1, y1 = positions[i - 1]
 t2, x2, y2 = positions[i]
 dt = t2 - t1
 if dt <= 0 or dt > 2.0: # skip large gaps (ball was lost)
 continue

 # Pixel distance (normalised 0..1) → metres
 # Assuming pitch width ≈ 105 m maps to normalised width 1.0
 dx_m = (x2 - x1) * 105.0
 dy_m = (y2 - y1) * 68.0 # pitch height ≈ 68 m
 dist_m = (dx_m**2 + dy_m**2) ** 0.5
 speed_mps = dist_m / dt
 speeds_mps.append(speed_mps)

 if not speeds_mps:
 return 0.0

 # 95th percentile — ignores occasional noise spikes
 speeds_arr = np.array(speeds_mps)
 top_speed_mps = float(np.percentile(speeds_arr, 95))
 top_speed_kmh = round(top_speed_mps * 3.6, 1)

 # Clamp to plausible ball speeds (0–200 km/h)
 return float(np.clip(top_speed_kmh, 0.0, 200.0))
