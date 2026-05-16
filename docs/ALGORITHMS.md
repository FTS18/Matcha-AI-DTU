# Algorithms Behind the Pipeline

A full breakdown of every algorithm used in the Matcha-AI soccer analysis engine.

---

## 1. Object Detection — YOLOv8 (Ultralytics)

- Models: `yolov8n.pt` / `yolov8s-pose.pt`
- Detects **players, referees, ball, goalkeepers** in every frame
- Runs in batches for efficiency
- Goalkeepers are remapped → `player` class

---

## 2. Object Tracking — ByteTrack (via Supervision)

- After detection, `sv.ByteTracker` assigns persistent **track IDs** to players across frames
- Enables consistent per-player identity across the whole clip

---

## 3. Top Speed & Distance Calculation

**Files:** `performance_metrics.py`, `soccer_analysis.py` (`SpeedAndDistanceEstimator`)

**How Top Speed works:**

1. Player **foot position** (bottom-center of bbox) is recorded each frame
2. Euclidean distance between consecutive frames: `√((x2-x1)² + (y2-y1)²)`
3. Divided by time delta → speed in m/s → converted to **km/h × 3.6**
4. Speeds > 40 km/h are discarded as noise
5. **Top Speed = 95th percentile** of all speed samples (not raw max, to filter outliers)
6. **Sprints** = count of frames where speed > 20 km/h

---

## 4. Perspective Transform (Pixel → Real-World Metres) — Homography

**Files:** `soccer_analysis.py` (`ViewTransformer`), `dynamic_calibration.py`

- **Hough Line Transform** detects white pitch lines
- **Convex Hull + `approxPolyDP`** finds the 4-corner pitch boundary
- **`cv2.findHomography`** computes a 3×3 perspective matrix mapping pixel coords → real metres (FIFA 105×68m pitch)
- Without this, speed would be in pixels/frame, not km/h

---

## 5. Camera Movement Compensation — Lucas-Kanade Optical Flow

**File:** `soccer_analysis.py` (`CameraMovementEstimator`)

- **`cv2.goodFeaturesToTrack`** finds stable corner features on left/right frame edges (static background)
- **`cv2.calcOpticalFlowPyrLK`** tracks those features between frames (pyramidal LK optical flow)
- Detected camera pan/tilt is **subtracted** from all player positions before speed calculation
- Prevents camera panning from inflating player speed numbers

---

## 6. Team Assignment — K-Means Clustering

**File:** `soccer_analysis.py` (`TeamAssigner`)

- Jersey crop (top half of player bbox) is extracted
- **K-Means (k=2)** clusters pixel colors into 2 groups: player jersey vs. background
- Corner pixels = background cluster → remaining = jersey color
- All jersey colors → global **K-Means (k=2, k-means++)** to split into Team 1 vs. Team 2

---

## 7. Ball Possession Assignment — Nearest Foot Distance

**File:** `soccer_analysis.py` (`PlayerBallAssigner`)

- For each frame, measures distance from ball center to **both feet** (bottom-left, bottom-right) of every player
- Player with minimum distance within threshold (70px) = ball possessor
- Cumulative possession → **Team 1 / Team 2 ball control %**

---

## 8. Event Detection — Physics Heuristics (CV)

**File:** `cv_detector.py` (`PhysicsDetector`)

| Event             | Algorithm                                                           |
| ----------------- | ------------------------------------------------------------------- |
| **GOAL**          | Ball dwells in goal zone (x < 10% or x > 90%) for 1+ seconds        |
| **SHOT**          | Ball sudden acceleration (velocity > 0.4) from near ankle keypoints |
| **SAVE / TACKLE** | Ball sharp deceleration (was fast → nearly stopped) near a player   |
| **CORNER**        | 6+ players cluster at corner coordinates (x < 10%, y < 20%)         |
| **FOUL**          | 6+ players cluster near penalty arc without a preceding goal        |
| **KICKOFF**       | Ball at center (45–55% of frame), 2 teams clearly separated         |
| **CELEBRATION**   | Tight player cluster within 15s after a GOAL event                  |

- Ball IoU overlap between player bboxes → **Tackle/collision** detection
- Events within 5 seconds of same type → **merged** (highest confidence kept)

---

## 9. Ball Interpolation — Pandas Linear Interpolation

- Ball detections have gaps (occlusion, fast movement)
- Missing frames filled using `DataFrame.interpolate()` + `bfill/ffill`

---

## 10. Activity Profile

- High-intensity effort % = frames where speed > 15 km/h / total frames
- Profile classified as: `Walking/Static` → `Moderate` → `High Intensity`

---

## Pipeline Flow Summary

```
Video Frames

│

▼
YOLOv8 Detection (players, ball, referee)

│

▼
ByteTrack (persistent track IDs)

│

▼
Camera Movement Compensation (Lucas-Kanade Optical Flow)

│

▼
Perspective Transform (Homography → real-world metres)

│

▼
Speed & Distance Estimation (95th percentile top speed)

│

▼
Team Assignment (K-Means on jersey color)

│

▼
Ball Possession Assignment (nearest foot distance)

│

▼
Event Detection (physics heuristics: GOAL, SHOT, TACKLE, etc.)

│

▼
Annotated Output Frames + Performance Metrics JSON
```
