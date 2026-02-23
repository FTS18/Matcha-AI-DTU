"""
Vision-Only Goal Detection Engine
==================================
Architecture:
  1. Ball Detection  -- YOLOv8 (sports ball class, per-frame bbox)
  2. Kalman Tracking -- smooth trajectory, occlusion-tolerant prediction
  3. Camera-Cut Detector -- histogram shift resets tracker
  4. Goal-Line Geometry -- homography-based perspective to top-down field view
  5. Goal Logic FSM  -- OUTSIDE->INSIDE polygon + velocity towards goal + N-frame dwell
  6. Physics Guards  -- min ball size, speed threshold, post-event cooldown

No player detection. No audio. No external tracking libraries.
Pure: OpenCV + NumPy + YOLO.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Cfg:
    YOLO_CONF             = 0.10  # Lowered from 0.15 to catch small balls
    BALL_MIN_PX           = 3     # Lowered from 4 for very small balls
    BALL_MAX_PX           = 200   # Increased from 180
    BALL_ASPECT_MAX       = 2.5
    KALMAN_PROCESS_NOISE  = 1e-2
    KALMAN_MEAS_NOISE     = 1e-1
    MAX_OCCLUSION_FRAMES  = 20    # Increased from 15 for tracking smoothness
    IOU_MATCH_THRESH      = 0.15  # Lowered from 0.20 for better matching
    HIST_DIFF_THRESHOLD   = 0.55
    DEFAULT_GOAL_POLY_NORM = None   
    DWELL_FRAMES          = 3
    VELOCITY_DOT_MIN      = 0.05    # Slightly stricter
    MIN_BALL_SPEED_PX     = 2.5
    GOAL_COOLDOWN_SEC     = 30.0
    TOPDOWN_W             = 200
    TOPDOWN_H             = 300

# Goal polygon: approximate goal-mouth area in broadcast camera view
# Tighter than before to avoid false positives from mid-field activity
# Left/right 5% of frame, vertical center 30-70%
Cfg.DEFAULT_GOAL_POLY_NORM = np.array([
    [0.00, 0.30],
    [0.08, 0.30],
    [0.08, 0.70],
    [0.00, 0.70],
], dtype=np.float32)

# Right-side goal polygon (mirrored)
Cfg.DEFAULT_GOAL_POLY_NORM_RIGHT = np.array([
    [0.92, 0.30],
    [1.00, 0.30],
    [1.00, 0.70],
    [0.92, 0.70],
], dtype=np.float32)


@dataclass
class BallObs:
    cx: float
    cy: float
    w:  float
    h:  float
    conf: float
    frame_id: int

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.cx - self.w/2, self.cy - self.h/2,
                self.cx + self.w/2, self.cy + self.h/2)


@dataclass
class TrackState:
    kf: cv2.KalmanFilter
    last_obs_frame: int
    frame_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=30))
    lost_frames: int = 0

    def predicted_center(self) -> Tuple[float, float]:
        s = self.kf.statePre
        return float(s[0]), float(s[1])

    def corrected_center(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[0]), float(s[1])

    def velocity(self) -> Tuple[float, float]:
        s = self.kf.statePost
        return float(s[2]), float(s[3])

    def speed(self) -> float:
        vx, vy = self.velocity()
        return float(np.hypot(vx, vy))


@dataclass
class GoalEvent:
    frame: int
    timestamp: float
    confidence: float
    ball_pos: Tuple[float, float]
    ball_speed_px: float
    direction: str


class KalmanBallTracker:
    """Constant-velocity Kalman tracker. State: [cx,cy,vx,vy]. Measurement: [cx,cy]."""

    def __init__(self):
        self._track: Optional[TrackState] = None
        self._frame = 0

    def reset(self):
        self._track = None

    def update(self, obs: Optional[BallObs]) -> Optional[Tuple[float, float]]:
        self._frame += 1
        if obs is not None:
            if self._track is None:
                self._track = self._init_track(obs)
            else:
                self._track.kf.predict()
                meas = np.array([[obs.cx], [obs.cy]], dtype=np.float32)
                self._track.kf.correct(meas)
                self._track.last_obs_frame = self._frame
                self._track.lost_frames = 0
            cx, cy = self._track.corrected_center()
            self._track.history.append((cx, cy))
            self._track.frame_id = self._frame
            return cx, cy
        else:
            if self._track is None:
                return None
            self._track.lost_frames += 1
            if self._track.lost_frames > Cfg.MAX_OCCLUSION_FRAMES:
                self._track = None
                return None
            self._track.kf.predict()
            cx, cy = self._track.predicted_center()
            self._track.history.append((cx, cy))
            return cx, cy

    @property
    def track(self) -> Optional[TrackState]:
        return self._track

    @staticmethod
    def _init_track(obs: BallObs) -> TrackState:
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
        
        # Smoothing parameters: Reduced process noise, increased measurement noise for stability
        # This makes the filter trust previous estimates more, reducing jitter
        kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 5e-3  # Lowered from 1e-2
        kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 2e-1  # Increased from 1e-1 for smoothing
        kf.errorCovPost       = np.eye(4, dtype=np.float32) * 0.1
        kf.statePost = np.array([[obs.cx],[obs.cy],[0.0],[0.0]], dtype=np.float32)
        ts = TrackState(kf=kf, last_obs_frame=0, frame_id=0)
        ts.history.append((obs.cx, obs.cy))
        return ts


class CameraCutDetector:
    """Histogram correlation drop -> scene cut -> reset tracker."""

    def __init__(self, threshold: float = Cfg.HIST_DIFF_THRESHOLD):
        self.threshold = threshold
        self._prev_hist: Optional[np.ndarray] = None

    def is_cut(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten().astype(np.float32)
        cv2.normalize(hist, hist)
        if self._prev_hist is None:
            self._prev_hist = hist
            return False
        corr = float(cv2.compareHist(
            self._prev_hist.reshape(-1,1), hist.reshape(-1,1), cv2.HISTCMP_CORREL
        ))
        self._prev_hist = hist
        return corr < self.threshold

    def reset(self):
        self._prev_hist = None


class GoalGeometry:
    """
    Goal-mouth polygon with optional homography to top-down view.
    Removes perspective distortion so goal line is a straight 2D boundary.
    """

    def __init__(self, frame_w: int, frame_h: int):
        self.fw = frame_w
        self.fh = frame_h
        self._poly_px: Optional[np.ndarray] = None
        self._homography: Optional[np.ndarray] = None
        self._topdown_poly: Optional[np.ndarray] = None
        self._use_topdown = False
        self._goal_center_x: float = frame_w / 2.0

    def set_goal_polygon(self, pts_norm: np.ndarray, build_homography: bool = True):
        pts = pts_norm.astype(np.float32).copy()
        pts[:, 0] *= self.fw
        pts[:, 1] *= self.fh
        self._poly_px = pts
        self._goal_center_x = float(np.mean(pts[:, 0]))
        if build_homography and len(pts) == 4:
            # Top-down view mapping: 18-yard box perspective
            dst = np.array([[0,0],[Cfg.TOPDOWN_W,0],
                            [Cfg.TOPDOWN_W,Cfg.TOPDOWN_H],[0,Cfg.TOPDOWN_H]], dtype=np.float32)
            H, _ = cv2.findHomography(pts, dst)
            if H is not None:
                self._homography = H
                self._topdown_poly = dst
                self._use_topdown = True

    def _detect_pitch_lines(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Find candidate goal-lines and 18-yard box lines using Hough Transform."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                detected.append((float(x1), float(y1), float(x2), float(y2)))
        return detected

    def auto_calibrate(self, frame: Optional[np.ndarray] = None):
        """
        Calibrate goal geometry. If a frame is provided, attempts to find 
        pitch lines to adjust the default polygon to the actual perspective.
        """
        if frame is not None:
            lines = self._detect_pitch_lines(frame)
            if len(lines) > 2:
                # Logic to find vertical-ish lines (goal posts) and horizontal (goal line)
                # For now, we use lines to refine the default poly if they are near the edges
                logger.info(f"GoalGeometry: detected {len(lines)} pitch lines for calibration")
                # TODO: Implement robust line-to-quad fitting
                # Fallback to default for now but with the infrastructure ready
        
        self.set_goal_polygon(Cfg.DEFAULT_GOAL_POLY_NORM, build_homography=True)
        logger.info("GoalGeometry: calibrated to broadcast perspective")

    def point_in_goal(self, cx: float, cy: float) -> bool:
        if self._poly_px is None:
            return False
        if self._use_topdown and self._homography is not None:
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            td = cv2.perspectiveTransform(pt, self._homography)[0][0]
            r  = cv2.pointPolygonTest(self._topdown_poly, (float(td[0]), float(td[1])), False)
        else:
            r = cv2.pointPolygonTest(self._poly_px, (cx, cy), False)
        return r >= 0

    def velocity_toward_goal(self, vx: float, vy: float, cx: float) -> bool:
        dot = vx * (self._goal_center_x - cx)
        return dot >= Cfg.VELOCITY_DOT_MIN

    @property
    def goal_center(self) -> Tuple[float, float]:
        if self._poly_px is not None:
            return float(np.mean(self._poly_px[:,0])), float(np.mean(self._poly_px[:,1]))
        return self.fw / 2.0, self.fh / 2.0


class BallDetector:
    """YOLOv8 wrapper. Detects sports ball only. Returns best per-frame BallObs."""

    BALL_LABELS = {"sports ball", "ball", "soccer ball", "football"}

    def __init__(self, model_path: str = "yolov8n.pt"):
        self._model = None
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(model_path)
            logger.info(f"BallDetector: YOLO loaded ({model_path})")
        except Exception as e:
            logger.warning(f"BallDetector: YOLO unavailable -- {e}")

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray, frame_id: int) -> Optional[BallObs]:
        if not self._model:
            return None
        try:
            results = self._model(frame, conf=Cfg.YOLO_CONF, verbose=False)
        except Exception as e:
            logger.debug(f"YOLO error: {e}")
            return None

        best: Optional[BallObs] = None
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                label = self._model.names[int(box.cls[0])].lower()
                if label not in self.BALL_LABELS:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bw, bh = x2 - x1, y2 - y1
                if bw < Cfg.BALL_MIN_PX or bh < Cfg.BALL_MIN_PX:
                    continue
                if bw > Cfg.BALL_MAX_PX or bh > Cfg.BALL_MAX_PX:
                    continue
                ratio = max(bw, bh) / max(min(bw, bh), 1e-6)
                if ratio > Cfg.BALL_ASPECT_MAX:
                    continue
                obs = BallObs(cx=(x1+x2)/2, cy=(y1+y2)/2,
                              w=bw, h=bh, conf=conf, frame_id=frame_id)
                if best is None or conf > best.conf:
                    best = obs
        return best


class RoboflowDetector:
    """Roboflow API wrapper for specialized ball/event detection."""

    def __init__(self, api_key: str, workspace: str, project: str, version: int = 1):
        self._model = None
        try:
            from roboflow import Roboflow
            rf = Roboflow(api_key=api_key)
            project_obj = rf.workspace(workspace).project(project)
            self._model = project_obj.model(version)
            logger.info(f"RoboflowDetector: Model loaded ({workspace}/{project}/{version})")
        except Exception as e:
            logger.warning(f"RoboflowDetector: Failed to initialize -- {e}")

    @property
    def available(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray, frame_id: int) -> Optional[BallObs]:
        if not self._model:
            return None
        
        try:
            # Roboflow inference
            prediction = self._model.predict(frame, confidence=int(Cfg.YOLO_CONF * 100)).json()
            
            best: Optional[BallObs] = None
            for p in prediction.get("predictions", []):
                # Filter for ball-like objects if not specific
                if p["class"].lower() not in ["ball", "soccer-ball", "football"]:
                    continue
                
                conf = p["confidence"]
                cx, cy = p["x"], p["y"]
                bw, bh = p["width"], p["height"]
                
                obs = BallObs(cx=cx, cy=cy, w=bw, h=bh, conf=conf, frame_id=frame_id)
                if best is None or conf > best.conf:
                    best = obs
            return best
        except Exception as e:
            logger.error(f"Roboflow detection error: {e}")
            return None


class _State(Enum):
    OUTSIDE  = auto()
    ENTERING = auto()
    COOLDOWN = auto()


class GoalFSM:
    """
    3-state FSM: OUTSIDE -> ENTERING -> COOLDOWN (goal event emitted).
    Guards: dwell frames + velocity direction + minimum speed.
    """

    def __init__(self, geometry: GoalGeometry, fps: float):
        self._geo   = geometry
        self._fps   = max(fps, 1.0)
        self._state = _State.OUTSIDE
        self._dwell = 0
        self._cooldown_left = 0.0

    def reset(self):
        self._state = _State.OUTSIDE
        self._dwell = 0
        self._cooldown_left = 0.0

    def step(self, cx: float, cy: float, vx: float, vy: float,
             speed: float, frame_id: int, timestamp: float,
             ball_conf: float) -> Optional[GoalEvent]:

        if self._state == _State.COOLDOWN:
            self._cooldown_left -= 1.0 / self._fps
            if self._cooldown_left <= 0:
                self._state = _State.OUTSIDE
            return None

        in_goal = self._geo.point_in_goal(cx, cy)

        if self._state == _State.OUTSIDE:
            if in_goal and speed >= Cfg.MIN_BALL_SPEED_PX:
                self._state = _State.ENTERING
                self._dwell = 1

        elif self._state == _State.ENTERING:
            if in_goal:
                self._dwell += 1
                if self._dwell >= Cfg.DWELL_FRAMES:
                    toward = self._geo.velocity_toward_goal(vx, vy, cx)
                    if toward or self._dwell >= Cfg.DWELL_FRAMES + 2:
                        # -- GOAL confirmed --
                        self._state = _State.COOLDOWN
                        self._cooldown_left = Cfg.GOAL_COOLDOWN_SEC
                        conf = min(ball_conf + min((self._dwell - Cfg.DWELL_FRAMES)*0.05, 0.15)
                                   + min(speed/60.0, 0.10), 0.99)
                        gcx, _ = self._geo.goal_center
                        direction = ("left_goal"  if gcx < self._geo.fw * 0.4 else
                                     "right_goal" if gcx > self._geo.fw * 0.6 else "center")
                        logger.info(f"GOAL at t={timestamp:.2f}s dwell={self._dwell} speed={speed:.1f} conf={conf:.3f}")
                        return GoalEvent(frame=frame_id, timestamp=timestamp,
                                         confidence=round(conf,3), ball_pos=(cx,cy),
                                         ball_speed_px=speed, direction=direction)
            else:
                self._state = _State.OUTSIDE
                self._dwell = 0

        return None


class GoalDetectionEngine:
    """Full pipeline: BallDetector/RoboflowDetector + KalmanTracker + CameraCut + GoalGeometry + GoalFSM."""

    def __init__(self, model_path: str = "yolov8n.pt", roboflow_cfg: Optional[dict] = None):
        self._detector = None
        
        # Prefer Roboflow if config is provided and valid
        if roboflow_cfg and roboflow_cfg.get("api_key"):
            self._detector = RoboflowDetector(
                api_key=roboflow_cfg["api_key"],
                workspace=roboflow_cfg.get("workspace", "matcha-ai"),
                project=roboflow_cfg.get("project", "soccer-ball-detection"),
                version=roboflow_cfg.get("version", 1)
            )
        
        # Fallback to local YOLO
        if self._detector is None or not self._detector.available:
            self._detector = BallDetector(model_path)
            
        self._tracker    = KalmanBallTracker()
        self._cut        = CameraCutDetector()
        self._geometry:  Optional[GoalGeometry] = None
        self._geometry_right: Optional[GoalGeometry] = None
        self._fsm:       Optional[GoalFSM]      = None
        self._fsm_right: Optional[GoalFSM]      = None
        self._frame_id   = 0
        self._fps        = 30.0
        self._goals: List[GoalEvent] = []
        self.frame_w     = 0
        self.frame_h     = 0
        logger.info("GoalDetectionEngine ready")

    def init(self, frame_w: int, frame_h: int, fps: float):
        self.frame_w  = frame_w
        self.frame_h  = frame_h
        self._fps     = max(fps, 1.0)
        # Left goal
        self._geometry = GoalGeometry(frame_w, frame_h)
        self._geometry.auto_calibrate()
        self._fsm = GoalFSM(self._geometry, self._fps)
        # Right goal
        self._geometry_right = GoalGeometry(frame_w, frame_h)
        self._geometry_right.set_goal_polygon(Cfg.DEFAULT_GOAL_POLY_NORM_RIGHT, build_homography=True)
        self._fsm_right = GoalFSM(self._geometry_right, self._fps)

    def set_goal_polygon(self, pts_norm: np.ndarray):
        if self._geometry is None:
            raise RuntimeError("Call init() first")
        self._geometry.set_goal_polygon(pts_norm, build_homography=True)

    def process_frame(self, frame: np.ndarray) -> Optional[GoalEvent]:
        if self._geometry is None:
            h, w = frame.shape[:2]
            self.init(w, h, self._fps)

        self._frame_id += 1
        timestamp = self._frame_id / self._fps

        # Camera cut -> reset tracker
        if self._cut.is_cut(frame):
            self._tracker.reset()

        # Downscale for speed
        h, w = frame.shape[:2]
        if w > 960:
            scale = 960.0 / w
            proc = cv2.resize(frame, (960, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            scale, proc = 1.0, frame

        # Detect ball
        obs = self._detector.detect(proc, self._frame_id)
        if obs is not None and scale < 1.0:
            inv = 1.0 / scale
            obs = BallObs(cx=obs.cx*inv, cy=obs.cy*inv,
                          w=obs.w*inv, h=obs.h*inv,
                          conf=obs.conf, frame_id=obs.frame_id)

        # Kalman update
        pos = self._tracker.update(obs)
        if pos is None:
            return None

        cx, cy = pos
        track  = self._tracker.track
        vx, vy = track.velocity() if track else (0.0, 0.0)
        speed  = track.speed()    if track else 0.0

        # Check both goal FSMs (left and right)
        event = self._fsm.step(
            cx=cx, cy=cy, vx=vx, vy=vy, speed=speed,
            frame_id=self._frame_id, timestamp=timestamp,
            ball_conf=obs.conf if obs else 0.5,
        )
        if event is None and self._fsm_right is not None:
            event = self._fsm_right.step(
                cx=cx, cy=cy, vx=vx, vy=vy, speed=speed,
                frame_id=self._frame_id, timestamp=timestamp,
                ball_conf=obs.conf if obs else 0.5,
            )
        if event:
            self._goals.append(event)
        return event

    def process_video(self, video_path: str, frame_step: int = 2,
                      callback=None) -> List[GoalEvent]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            return []
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
        fh   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        self.init(fw, fh, fps)
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % frame_step != 0:
                    continue
                event = self.process_frame(frame)
                if event and callback:
                    callback(idx, event)
        finally:
            cap.release()
        logger.info(f"process_video: {idx} frames, {len(self._goals)} goals")
        return list(self._goals)

    def reset(self):
        self._tracker.reset()
        self._cut.reset()
        if self._fsm:
            self._fsm.reset()
        if self._fsm_right:
            self._fsm_right.reset()
        self._frame_id = 0
        self._goals.clear()

    @property
    def goals(self) -> List[GoalEvent]:
        return list(self._goals)


def goal_events_to_raw(events: List[GoalEvent]) -> List[dict]:
    """Convert GoalEvents to analysis.py raw_event dicts."""
    return [
        {
            "timestamp":   round(e.timestamp, 2),
            "type":        "GOAL",
            "confidence":  e.confidence,
            "description": f"Ball entered goal ({e.direction}) speed={e.ball_speed_px:.1f}px/f",
            "source":      "goal_detection",
        }
        for e in events
    ]
