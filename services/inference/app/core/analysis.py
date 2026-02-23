import cv2
import logging
import requests
import os

# Ensure ~/bin is on PATH for ffmpeg and other local binaries
_home_bin = os.path.join(os.path.expanduser("~"), "bin")
if _home_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _home_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Base paths (work for both Docker and native Windows) ─────────────────────
# When running in Docker: /app is the workdir
# When running natively: services/inference is the workdir
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # services/inference/
UPLOADS_DIR = BASE_DIR.parent.parent / "uploads"  # workspace/uploads/
MUSIC_DIR = BASE_DIR / "app" / "music"  # services/inference/app/music/

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MUSIC_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "VIDEO_SAMPLE_FPS": 1.0,
    "VIDEO_PROCESS_FPS": 1.0,
    "MOTION_WINDOW_SECS": 5.0,
    "TARGET_FRAME_HEIGHT": 480,
    "MAX_FRAME_WIDTH": 800,
    "YOLO_DOWNSCALE_HEIGHT": 480,
    "MOTION_PEAK_THRESHOLD": 0.45,
    "MOTION_FALLBACK_THRESHOLD": 0.55,
    "MOTION_MIN_GAP_SECS": 20.0,
    "MOTION_FALLBACK_MIN_GAP": 20.0,
    "CANDIDATE_MIN_MOTION": 0.65,
    "MAX_MOTION_BASED_EVENTS": 8,
    "HIGHLIGHT_CLIP_DURATION": 30.0,
    "HIGHLIGHT_CLIP_PRE_PCT": 0.35,
    "HIGHLIGHT_CLIP_POST_PCT": 0.65,
    "HIGHLIGHT_COUNT": 5,
    "HIGHLIGHT_MIN_SPREAD_PCT": 0.15,
    "AUDIO_PEAK_PERCENTILE": 90,
    "AUDIO_MOTION_DIVISOR": 40.0,
    "VISION_FAILURE_THRESHOLD": 5,
    "YOLO_SKIP_MOTION_THRESHOLD": 0.15,
    "YOLO_SKIP_MOTION_INTERVAL": 3,
    "LATE_GAME_PCT": 0.85,
    "EARLY_GAME_PCT": 0.08,
    "FRAME_STEP_FOR_1FPS": 30,
    "COMPRESS_SIZE_THRESHOLD_MB": 100,
    "COMPRESS_OUTPUT_HEIGHT": 480,
    "COMPRESS_OUTPUT_FPS": 1,
    "FFMPEG_CLIP_TIMEOUT": 120,
    "FFMPEG_CONCAT_TIMEOUT": 60,
    "FFMPEG_COMPRESS_TIMEOUT": 300,
    "MAX_SUMMARY_CHARS": 5000,
    "MAX_COMMENTARY_CHARS": 1000,
    "MAX_HIGHLIGHT_COMMENTARY_CHARS": 500,
    "MAX_EVENTTYPE_CHARS": 50,
    # ── Goal Detection Parameters ─────────────────────────────────────────
    "GOAL_DETECTION_ENABLED": True,
    "GOAL_DETECTION_MIN_FRAMES": 3,  # Frames ball must be in goal area
    "GOAL_DETECTION_MIN_SIZE": 10,  # Min ball bbox size in pixels
    "GOAL_DETECTION_MAX_SIZE": 200,  # Max ball bbox size in pixels
    "GOAL_DETECTION_CONFIDENCE_THRESHOLD": 0.5,  # Goal confidence threshold
    # ── Roboflow API Settings ─────────────────────────────────────────────
    "ROBOFLOW_API_KEY": os.getenv("ROBOFLOW_API_KEY"),
    "ROBOFLOW_WORKSPACE": os.getenv("ROBOFLOW_WORKSPACE", "matcha-ai"),
    "ROBOFLOW_PROJECT": os.getenv("ROBOFLOW_PROJECT", "soccer-ball-detection"),
    "ROBOFLOW_VERSION": int(os.getenv("ROBOFLOW_VERSION", "1")),
    # ── Live Stream Parameters ────────────────────────────────────────────
    "STREAM_BUFFER_SIZE": 15,  # Seconds to buffer before checking for events
    "STREAM_MAX_IDLE_SECS": 300,  # Auto-stop after 5 mins of no frames
    "LIVE_EMIT_INTERVAL_SECS": 1.0, # How often to update progress/status
    # ── Performance Optimization (NEW) ────────────────────────────────────
    "ENABLE_GPU_ACCELERATION": True,  # Use CUDA/GPU for YOLO if available
    "PARALLEL_WORKERS": 4,  # Number of parallel frame processing threads
    "BATCH_YOLO_SIZE": 8,  # Batch frames for YOLO inference
    "SMART_FRAME_SKIP": True,  # Skip low-motion frames intelligently
    "MOTION_CACHE_ENABLED": True,  # Cache motion scores to avoid recalculation
    "ENABLE_INFERENCE_CACHING": True,  # Cache YOLO results for similar frames
    "CACHE_SIMILARITY_THRESHOLD": 0.95,  # Frame similarity for cache hits (0.0-1.0)
    # ── Phase 2 Optimizations ────────────────────────────────────────────
    "ENHANCED_BALL_TRACKING": True,  # Kalman filter + trajectory prediction
    "BALL_SMOOTHING_WINDOW": 3,  # Frames for ball trajectory smoothing
    "CONTEXT_AWARE_COMMENTARY": True,  # Gemini analysis of team formation & tactics
    "DYNAMIC_AUDIO_MIXING": True,  # Auto-adjust volumes based on match intensity
    "SMART_HIGHLIGHT_SELECTION": True,  # Narrative flow + deduplication
    "HIGHLIGHT_NARRATIVE_CONTEXT": True,  # Group related events together
    "MIN_EVENT_GAP_FOR_GROUPING": 15.0,  # Seconds to group events (build-up + goal)
}

# ── Heatmap & Speed analytics ───────────────────────────────────────────────
try:
    from app.core.heatmap import generate_heatmap, estimate_ball_speed
    HEATMAP_AVAILABLE = True
    logger.info("Heatmap module loaded ✓")
except ImportError as e:
    logger.warning(f"Heatmap module not available: {e}")
    HEATMAP_AVAILABLE = False
    generate_heatmap = None
    estimate_ball_speed = None

# ── Goal Detection (vision-based goal line crossing) ──────────────────────────
try:
    from app.core.goal_detection import GoalDetectionEngine
    GOAL_DETECTION_AVAILABLE = True
    logger.info("Goal Detection engine loaded ✓")
except ImportError as e:
    logger.warning(f"Goal Detection not available: {e}")
    GOAL_DETECTION_AVAILABLE = False
    GoalDetectionEngine = None

# ── Goalpost Detection (for spatial awareness) ────────────────────────────────
try:
    from app.core.goalpost_detection import GoalpostDetector, GoalpostTracker
    GOALPOST_DETECTION_AVAILABLE = True
    logger.info("Goalpost Detection module loaded ✓")
except ImportError as e:
    logger.warning(f"Goalpost Detection not available: {e}")
    GOALPOST_DETECTION_AVAILABLE = False
    GoalpostDetector = None
    GoalpostTracker = None

# ── SoccerNet (football-specific event detection) ────────────────────────────
try:
    from app.core.soccernet_detector import detect_football_events
    SOCCERNET_AVAILABLE = True
    logger.info("SoccerNet detector loaded ✓")
except ImportError as e:
    logger.warning(f"SoccerNet detector not available: {e}")
    SOCCERNET_AVAILABLE = False
    detect_football_events = None

# ── CV Physics (YOLO-based heuristics) ───────────────────────────────────────
try:
    from app.core.cv_detector import detect_all as detect_cv_physics
    CV_PHYSICS_AVAILABLE = True
    logger.info("CV Physics detector loaded ✓")
except ImportError as e:
    logger.warning(f"CV Physics detector not available: {e}")
    CV_PHYSICS_AVAILABLE = False
    detect_cv_physics = None

# ── Soccer Analysis (broadcast-quality overlays) ─────────────────────────────
try:
    from app.core.soccer_analysis import is_available as _sa_is_available
    SOCCER_ANALYSIS_AVAILABLE = _sa_is_available()
    if SOCCER_ANALYSIS_AVAILABLE:
        logger.info("Soccer Analysis overlay engine loaded ✓")
    else:
        logger.warning("Soccer Analysis overlay: missing dependencies (supervision / scikit-learn)")
except ImportError as e:
    logger.warning(f"Soccer Analysis overlay not available: {e}")
    SOCCER_ANALYSIS_AVAILABLE = False

# ── YOLO ─────────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO  # type: ignore
    from ultralytics.nn.tasks import DetectionModel  # type: ignore
    # PyTorch 2.6+ requires adding safe globals for model loading
    if hasattr(torch.serialization, "add_safe_globals"):  # type: ignore
        import torch.nn.modules.container
        import torch.nn.modules.conv
        import torch.nn.modules.batchnorm
        import torch.nn.modules.activation
        import torch.nn.modules.pooling
        import torch.nn.modules.upsampling
        torch.serialization.add_safe_globals([  # type: ignore
            DetectionModel,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.upsampling.Upsample,
        ])
except Exception as e:
    logger.warning(f"Could not add safe globals: {e}")
    from ultralytics import YOLO  # type: ignore

from app.core.llm import (
    analyze_frame_with_vision, 
    generate_commentary, 
    generate_match_summary, 
    _get_gemini
)
from app.core.tts import tts_generate, get_tts_available as _get_tts
from app.core.video_utils import (
    generate_silent_audio as _generate_silent_audio, 
    create_highlight_reel, 
    precompress_video as _precompress_video
)

# ── Performance Optimization Utilities ────────────────────────────────────────
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
import hashlib

class FrameCache:
    """LRU cache for YOLO inference results to avoid redundant processing."""
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.95):
        self.cache = {}
        self.order = deque(maxlen=max_size)
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0
    
    def _get_hash(self, frame: np.ndarray) -> str:
        """Compute frame hash using the first few bytes."""
        return hashlib.md5(frame[::4, ::4].tobytes()).hexdigest()  # Sample every 4th pixel for speed
    
    def get(self, frame: np.ndarray):
        """Try to retrieve cached result for similar frame."""
        frame_hash = self._get_hash(frame)
        if frame_hash in self.cache:
            self.hits += 1
            return self.cache[frame_hash]
        self.misses += 1
        return None
    
    def put(self, frame: np.ndarray, result):
        """Cache the inference result."""
        frame_hash = self._get_hash(frame)
        if len(self.order) >= self.max_size and frame_hash not in self.cache:
            oldest = self.order[0]
            del self.cache[oldest]
        self.cache[frame_hash] = result
        self.order.append(frame_hash)
    
    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": f"{hit_rate:.1f}%", "size": len(self.cache)}

# Initialize frame cache
_frame_cache = FrameCache(max_size=100, similarity_threshold=CONFIG["CACHE_SIMILARITY_THRESHOLD"])

def _detect_gpu_availability():
    """Check if GPU is available for acceleration."""
    if not CONFIG["ENABLE_GPU_ACCELERATION"]:
        return False
    try:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            logger.info(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        return gpu_available
    except:
        return False

GPU_AVAILABLE = _detect_gpu_availability()

# ── Logic moved to app.core.llm ───────────────────────────────────────────────
def _frame_to_pil(frame):
    from app.core.llm import _frame_to_pil as f2p
    return f2p(frame)


def validate_candidate_moment(cap, timestamp: float, fps: float, duration: float) -> Optional[Dict]:
    """
    Validate a candidate moment by analyzing multiple frames around it.
    Uses majority voting across frames for reliable event classification.
    
    Returns: {"event_type": str, "confidence": float, "timestamp": float, "description": str}
             or None if no significant event detected.
    """
    # Sample 3 frames: 0.5s before, at timestamp, and 0.5s after
    offsets = [-0.5, 0.0, 0.5]
    results = []
    
    for offset in offsets:
        t = timestamp + offset
        if t < 0 or t > duration:
            continue
            
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Downscale for faster API calls
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        
        result = analyze_frame_with_vision(frame, t)
        if result["event_type"] != "NONE" and result["confidence"] >= 0.5:
            results.append(result)
    
    if not results:
        return None
    
    # Majority voting: pick the most common event type
    from collections import Counter
    event_counts = Counter(r["event_type"] for r in results)
    most_common_event, count = event_counts.most_common(1)[0]
    
    # Require at least 2 frames to agree for non-GOAL events
    # Goals are dramatic enough that 1 confident detection is enough
    if most_common_event != "GOAL" and count < 2:
        return None
    
    # Average confidence of matching results
    matching = [r for r in results if r["event_type"] == most_common_event]
    avg_confidence = sum(r["confidence"] for r in matching) / len(matching)
    
    # Use the best description
    best_result = max(matching, key=lambda r: r["confidence"])
    
    return {
        "event_type": most_common_event,
        "confidence": round(avg_confidence, 3),
        "timestamp": round(timestamp, 2),
        "description": best_result["description"],
        "frame_votes": count,
    }


# Track consecutive Vision AI failures for fallback
_vision_failures = 0
_MAX_VISION_FAILURES = 5  # After this many failures, use fallback


def fallback_heuristic_event(motion_score: float, timestamp: float, duration: float) -> Optional[Dict]:
    """
    Fallback event detection when Vision AI is unavailable.
    Uses motion score + temporal position to generate generic highlights.
    Much less accurate but ensures some highlights are generated.
    """
    # Very high motion in key periods suggests something important
    late_game = duration > 0 and (timestamp / duration) > 0.75
    
    # Conservative thresholds - only flag extremely high motion moments
    if motion_score >= 0.7:
        event_type = "HIGHLIGHT"  # Generic highlight
        confidence = min(0.6, motion_score * 0.8)
        desc = "High action moment detected (Vision AI fallback)"
    elif motion_score >= 0.55 and late_game:
        event_type = "HIGHLIGHT"
        confidence = 0.5
        desc = "Late-game action moment (Vision AI fallback)"
    else:
        return None
    
    return {
        "event_type": event_type,
        "confidence": round(confidence, 3),
        "timestamp": round(timestamp, 2),
        "description": desc,
    }


def validate_candidate_with_fallback(cap, timestamp: float, fps: float, duration: float, motion_score: float) -> Optional[Dict]:
    """
    Validate a candidate moment, with fallback to heuristics if Vision AI fails.
    """
    global _vision_failures
    
    # If too many Vision AI failures, use fallback immediately
    if _vision_failures >= _MAX_VISION_FAILURES:
        logger.warning("Vision AI unavailable - using motion-based fallback")
        return fallback_heuristic_event(motion_score, timestamp, duration)
    
    result = validate_candidate_moment(cap, timestamp, fps, duration)
    
    if result is None:
        # Check if this was due to Vision AI failure (no frames analyzed)
        # vs. genuinely no event detected
        # We can't easily distinguish, so just try the fallback for high-motion moments
        if motion_score >= 0.65:
            _vision_failures += 1
            if _vision_failures >= _MAX_VISION_FAILURES:
                logger.warning(f"Vision AI failed {_vision_failures} times - switching to fallback mode")
            return fallback_heuristic_event(motion_score, timestamp, duration)
        return None
    
    # Vision AI worked - reset failure counter
    _vision_failures = 0
    return result


def find_motion_peaks(motion_windows: list, threshold: Optional[float] = None, min_gap: Optional[float] = None) -> list:
    if threshold is None:
        threshold = CONFIG["MOTION_PEAK_THRESHOLD"]
    if min_gap is None:
        min_gap = CONFIG["MOTION_MIN_GAP_SECS"]
    
    # Type narrowing - ensure not None
    assert isinstance(threshold, float) and isinstance(min_gap, float)
        
    if not isinstance(motion_windows, list) or not motion_windows:
        return []
    if threshold < 0 or threshold > 1:
        logger.warning(f"Invalid threshold {threshold}, using default")
        threshold = CONFIG["MOTION_PEAK_THRESHOLD"]
    
    candidates = []
    last_peak = -999
    
    for w in motion_windows:
        if not isinstance(w, dict) or "motionScore" not in w or "timestamp" not in w:
            continue
        if w["motionScore"] >= threshold:
            t = w["timestamp"]
            if t - last_peak >= min_gap:
                candidates.append(t)
                last_peak = t
    
    return candidates


# ── FFmpeg helpers ───────────────────────────────────────────────────────────
# TTS logic moved to app.core.tts

import subprocess
import tempfile

# Highlight logic moved to app.core.video_utils


ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:4000/api/v1")

# Use YOLOv8s-pose for person keypoints, and tiny model for ball
# Downloads automatically on first run (~22MB)
model = YOLO("yolov8s-pose.pt")
ball_model = YOLO("yolov8n.pt")

# ── COCO classes to track for visualization ──────────────────────────────────
# NOTE: YOLO is now ONLY used for ball/player tracking visualization on canvas
# Events are detected via Gemini Vision analysis of candidate moments
YOLO_TRACK_CLASSES = {"sports ball", "person"}

# Legacy mapping - kept for compatibility but NO LONGER used for event creation
YOLO_TYPE_MAP = {"sports ball": "GOAL", "person": "TACKLE"}

# Per-class minimum YOLO confidence thresholds (for tracking visualization)
MIN_CONF: Dict[str, float] = {
    "sports ball": 0.30,   # small + blurry → lower threshold
    "person":      0.50,   # for player tracking
}

# ── Minimum gap between events of the same type (seconds) ────────────────────
# Used by Vision AI event detection phase
MODEL_MIN_GAP: Dict[str, float] = {
    "GOAL":        5.0,
    "TACKLE":     45.0,
    "SAVE":       20.0,
    "FOUL":       15.0,
    "Celebrate":  30.0,
    "PENALTY":    10.0,
    "RED_CARD":   10.0,
    "YELLOW_CARD": 10.0,
    "CORNER":     10.0,
    "OFFSIDE":    10.0,
}
DEFAULT_MIN_GAP = 20.0

# ── Event weight table (out of 10) ───────────────────────────────────────────
EVENT_WEIGHTS = {
    "GOAL":         10.0,
    "PENALTY":       9.5,
    "RED_CARD":      9.0,
    "SAVE":          8.0,
    "YELLOW_CARD":   7.0,
    "CELEBRATION":   6.5,   # Vision AI detected celebration
    "FOUL":          6.0,
    "HIGHLIGHT":     5.5,   # Fallback generic highlight from motion
    "TACKLE":        5.0,
    "CORNER":        4.0,
    "OFFSIDE":       2.5,
}
W1, W2, W3, W4 = 0.40, 0.20, 0.25, 0.15


# ── Scoring helpers ───────────────────────────────────────────────────────────
def time_context_weight(timestamp, duration):
    """Late-game moments carry more weight."""
    if duration <= 0:
        return 0.70
    pct = timestamp / duration
    if pct > 0.92:   return 1.00   # injury time / dying minutes
    if pct > 0.85:   return 0.95   # final 10 min
    if pct > 0.70:   return 0.85   # last quarter
    if pct > 0.50:   return 0.75   # second half
    if pct > 0.45:   return 0.60   # around half-time
    return 0.65                    # first half


def compute_context_score(event_type, motion_score, timestamp, duration, confidence):
    ew    = EVENT_WEIGHTS.get(event_type, 4.0) / 10.0
    audio = min(motion_score * 1.3, 1.0)
    tw    = time_context_weight(timestamp, duration)
    base  = (ew * W1) + (audio * W2) + (motion_score * W3) + (tw * W4)
    score = base * (0.5 + 0.5 * confidence)
    if duration > 0 and (timestamp / duration) > 0.85 and event_type == "GOAL":
        score *= 2.0   # late goals doubled
    if duration > 0 and (timestamp / duration) < 0.08 and event_type in ("SAVE", "TACKLE"):
        score *= 1.3   # frantic early-game bump
    return round(min(score * 10.0, 10.0), 2)


# ── Fallback commentary ───────────────────────────────────────────────────────
_FALLBACK = {
    "GOAL":        {"high": "GOOOAL! Sensational — the crowd erupts!", "mid": "Goal! Crucial finish puts them ahead!", "low": "Goal scored."},
    "TACKLE":      {"high": "FEROCIOUS TACKLE! Incredible commitment!", "mid": "Strong challenge wins the ball back.", "low": "Tackle wins possession."},
    "FOUL":        {"high": "DEFINITE FOUL! Referee steps in immediately!", "mid": "Free kick awarded — bodies flying here.", "low": "Foul given."},
    "SAVE":        {"high": "UNBELIEVABLE SAVE! Superhuman goalkeeping!", "mid": "Good stop from the keeper — keeping them in it.", "low": "Save made."},
    "CELEBRATION": {"high": "INCREDIBLE SCENES! The players are losing their minds!", "mid": "Celebrations break out on the pitch!", "low": "The players celebrate."},
    "HIGHLIGHT":   {"high": "WHAT A MOMENT! Crucial action in this match!", "mid": "Important moment of play here.", "low": "Key moment of play."},
}


def _fallback_commentary(event_type, final_score, timestamp, duration):
    minute = max(1, int(timestamp / 60))
    late   = duration > 0 and (timestamp / duration) > 0.85
    energy = "high" if final_score >= 7.5 else ("mid" if final_score >= 5 else "low")
    text   = _FALLBACK.get(event_type, {}).get(energy, f"{event_type} at minute {minute}.")
    if "minute" not in text.lower():
        text = text.rstrip("!.") + f" at minute {minute}."
    if late and final_score >= 7:
        text = "LATE DRAMA! " + text
    return text


# ── Gemini commentary ─────────────────────────────────────────────────────────
# Commentary logic moved to app.core.llm


# ── Gemini match summary ──────────────────────────────────────────────────────
# Summary logic moved to app.core.llm


# ── Highlight selection ───────────────────────────────────────────────────────
def select_highlights(scored_events, duration, top_n=5, clip_secs=30.0):
    """
    Top-N non-overlapping highlights, spread across the full video.
    Two rules:
      1. No time-window overlap between clips.
      2. Clip centres must be at least 15% of duration apart (prevents same-scene
         from different YOLO frames appearing twice).
    """
    if not scored_events:
        return []
    sorted_evs = sorted(scored_events, key=lambda x: x["finalScore"], reverse=True)
    min_spread = max(30.0, duration * 0.15)   # at least 15% of the video length
    used, highlights = [], []

    for ev in sorted_evs:
        if len(highlights) >= top_n:
            break
        ts    = ev["timestamp"]
        start = max(0.0, ts - clip_secs * 0.35)
        end   = min(duration if duration > 0 else ts + 60.0, ts + clip_secs * 0.65)

        # Rule 1 – no overlap
        if any(not (end <= w[0] or start >= w[1]) for w in used):
            continue

        # Rule 2 – clips must be spread out
        if any(abs(ts - (h["startTime"] + (h["endTime"] - h["startTime"]) / 2)) < min_spread
               for h in highlights):
            continue

        used.append((start, end))
        highlights.append({
            "startTime":  round(start, 1),
            "endTime":    round(end, 1),
            "score":      ev["finalScore"],
            "eventType":  ev["type"],
            "commentary": ev.get("commentary", ""),
        })

    return sorted(highlights, key=lambda x: x["startTime"])


# ── Team colour clustering ────────────────────────────────────────────────────
def _crop_jersey(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Return the torso crop (middle 40% height, inner 60% width) of a person box."""
    h, w = frame.shape[:2]
    bx1, by1 = int(x1 * w), int(y1 * h)
    bx2, by2 = int(x2 * w), int(y2 * h)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 4 or bh < 10:
        return np.array([])
    # Torso: rows 30-70%, cols 20-80%
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


def _cluster_teams(colours: list[list[int]], n: int = 2) -> tuple[list[list[int]], list[int]]:
    """
    K-means cluster colours into n teams.
    Returns (centroids, labels) where centroids = [[R,G,B], ...]
    Uses numpy-only mini K-means (no sklearn needed).
    """
    if len(colours) < n:
        defaults = [[220, 50, 50], [50, 100, 220]]
        return defaults[:n], [i % n for i in range(len(colours))]
    data = np.array(colours, dtype=np.float32)
    # Initialise centres from extremes
    centres = data[np.random.choice(len(data), n, replace=False)]
    for _ in range(20):
        dists  = np.stack([np.linalg.norm(data - c, axis=1) for c in centres], axis=1)
        labels = np.argmin(dists, axis=1)
        new_c  = np.stack([
            data[labels == k].mean(axis=0) if np.any(labels == k) else centres[k]
            for k in range(n)
        ])
        if np.allclose(centres, new_c, atol=1.0):
            break
        centres = new_c
    final_labels = np.argmin(
        np.stack([np.linalg.norm(data - c, axis=1) for c in centres], axis=1), axis=1
    )
    return [[int(v) for v in c.tolist()] for c in centres], final_labels.tolist()


def _get_motion_at(windows, timestamp):
    if not windows:
        return 0.3
    return min(windows, key=lambda w: abs(w["timestamp"] - timestamp))["motionScore"]


# ── PHASE 2 OPTIMIZATION 1: ENHANCED BALL TRACKING ──────────────────────────
def smooth_ball_trajectory(track_frames: list, window_size: int = 3) -> list:
    """
    Smooth ball positions using a sliding window average to reduce jitter.
    Improves visual quality and tracking stability.
    """
    if not track_frames or window_size < 1:
        return track_frames
    
    smoothed = []
    for i, frame in enumerate(track_frames):
        if not frame.get("b") or len(frame["b"]) == 0:
            smoothed.append(frame)
            continue
        
        # Get surrounding frames for smoothing
        start = max(0, i - window_size // 2)
        end = min(len(track_frames), i + window_size // 2 + 1)
        window_frames = track_frames[start:end]
        
        # Average ball positions across window
        all_balls = []
        for wf in window_frames:
            all_balls.extend(wf.get("b", []))
        
        if all_balls:
            # Average each coordinate
            smoothed_ball = [
                round(np.mean([b[j] for b in all_balls]), 4)
                for j in range(len(all_balls[0]))
            ]
            frame_copy = frame.copy()
            frame_copy["b"] = [smoothed_ball[:4]]  # Keep top ball
            smoothed.append(frame_copy)
        else:
            smoothed.append(frame)
    
    return smoothed


def predict_ball_trajectory(balls: list, fps: float = 30.0) -> dict:
    """
    Predict ball movement trajectory based on historical positions.
    Useful for detecting ball speed and direction changes.
    """
    if len(balls) < 2:
        return {"direction": "unknown", "speed": 0.0, "confidence": 0.0}
    
    try:
        # Get last 3 positions
        recent = balls[-3:] if len(balls) >= 3 else balls
        if len(recent) < 2:
            return {"direction": "unknown", "speed": 0.0, "confidence": 0.0}
        
        # Calculate velocity
        x_vel = recent[-1][0] - recent[-2][0]
        y_vel = recent[-1][1] - recent[-2][1]
        
        # Speed in normalized units per frame
        speed = np.sqrt(x_vel**2 + y_vel**2)
        
        # Direction in degrees (0=right, 90=down)
        direction = np.degrees(np.arctan2(y_vel, x_vel))
        
        # Confidence based on consistency
        confidence = min(speed * 2, 1.0)  # Higher speed = more confident
        
        return {
            "direction": f"{direction:.1f}°",
            "speed": round(speed, 4),
            "confidence": round(confidence, 3),
            "velocity": [round(x_vel, 4), round(y_vel, 4)]
        }
    except Exception as e:
        logger.debug(f"Trajectory prediction failed: {e}")
        return {"direction": "unknown", "speed": 0.0, "confidence": 0.0}


# ── PHASE 2 OPTIMIZATION 2: CONTEXT-AWARE COMMENTARY ─────────────────────────
def analyze_team_formation(track_frames: list, team_colors: list) -> dict:
    """
    Analyze team formation and positioning from tracking data.
    Returns formation metrics for context-aware commentary.
    """
    if not track_frames:
        return {"formation": "unknown", "spacing": 0.0, "cohesion": 0.0}
    
    try:
        # Get positions from last few frames
        recent_frames = track_frames[-5:]
        all_positions = []
        
        for frame in recent_frames:
            for person in frame.get("p", [])[:11]:  # 11 players max
                if len(person) >= 3:
                    all_positions.append((person[0], person[1]))  # x, y
        
        if len(all_positions) < 4:
            return {"formation": "unknown", "spacing": 0.0, "cohesion": 0.0}
        
        positions = np.array(all_positions)
        
        # Calculate pairwise distances
        distances = np.sqrt(((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2))
        
        # Average spacing (excluding self-distances of 0)
        spacing = np.mean(distances[distances > 0.01])
        
        # Cohesion = inverse of spacing (closer = higher cohesion)
        cohesion = 1.0 / (1.0 + spacing)
        
        # Formation classification based on spacing
        if spacing < 0.15:
            formation = "compact"  # Defensive
        elif spacing < 0.25:
            formation = "balanced"
        else:
            formation = "spread"  # Attacking
        
        return {
            "formation": formation,
            "spacing": round(spacing, 3),
            "cohesion": round(cohesion, 3),
            "player_count": len(all_positions)
        }
    except Exception as e:
        logger.debug(f"Formation analysis failed: {e}")
        return {"formation": "unknown", "spacing": 0.0, "cohesion": 0.0}


# ── PHASE 2 OPTIMIZATION 3: DYNAMIC AUDIO MIXING ──────────────────────────────
def calculate_dynamic_audio_volumes(motion_score: float, emotion_score: float) -> dict:
    """
    Calculate audio volumes dynamically based on match intensity.
    Higher intensity = louder crowd, reduced music, emphasized commentary.
    """
    # Normalize inputs to 0-1 range
    motion = max(0.0, min(1.0, motion_score))
    emotion = max(0.0, min(1.0, emotion_score / 10.0))
    
    # Average intensity
    intensity = (motion + emotion) / 2.0
    
    # Dynamic volume adjustments
    volumes = {
        "music": max(0.02, 0.15 * (1.0 - intensity)),  # Fade out in intense moments
        "crowd": 0.25 + (0.35 * intensity),  # Ramp up with intensity
        "roar": 0.1 + (0.4 * intensity),  # Roar on big moments
        "commentary": 1.2 + (0.3 * intensity)  # Always prominent, boost on action
    }
    
    # Normalize to prevent clipping
    max_vol = max(volumes.values())
    if max_vol > 1.5:
        scale = 1.5 / max_vol
        volumes = {k: round(v * scale, 3) for k, v in volumes.items()}
    else:
        volumes = {k: round(v, 3) for k, v in volumes.items()}
    
    return volumes


# ── PHASE 2 OPTIMIZATION 4: SMART HIGHLIGHT SELECTION ────────────────────────
def group_related_events(scored_events: list, min_gap_secs: float = 15.0) -> list:
    """
    Group related events (e.g., build-up + goal) for better narrative flow.
    Returns events with group_id for clustering.
    """
    if not scored_events:
        return []
    
    grouped = []
    current_group = 0
    last_group_time = scored_events[0]["timestamp"]
    
    for event in scored_events:
        time_since_group = event["timestamp"] - last_group_time
        
        # Start new group if gap is large
        if time_since_group > min_gap_secs:
            current_group += 1
            last_group_time = event["timestamp"]
        
        event_copy = event.copy()
        event_copy["group_id"] = current_group
        event_copy["time_in_group"] = round(time_since_group, 2)
        grouped.append(event_copy)
    
    return grouped


def select_highlights_with_narrative(scored_events: list, duration: float, top_n: int = 5, 
                                      clip_secs: float = 30.0, use_groups: bool = True) -> list:
    """
    Enhanced highlight selection that considers narrative flow and event grouping.
    Groups build-up sequences with their payoff (e.g., goal sequences).
    """
    if not scored_events:
        return []
    
    # Group related events if enabled
    if use_groups:
        grouped_events = group_related_events(scored_events, min_gap_secs=15.0)
    else:
        grouped_events = scored_events
    
    # Sort by final score
    sorted_evs = sorted(grouped_events, key=lambda x: x["finalScore"], reverse=True)
    
    min_spread = max(30.0, duration * 0.15)
    used = []
    highlights = []
    
    for ev in sorted_evs:
        if len(highlights) >= top_n:
            break
        
        ts = ev["timestamp"]
        
        # Extend clip backwards if there's a group with lead-up events
        start_buffer = clip_secs * 0.50 if "group_id" in ev and ev.get("time_in_group", 0) > 10 else clip_secs * 0.35
        
        start = max(0.0, ts - start_buffer)
        end = min(duration if duration > 0 else ts + 60.0, ts + clip_secs * 0.65)
        
        # Rule 1: No overlap
        if any(not (end <= w[0] or start >= w[1]) for w in used):
            continue
        
        # Rule 2: Clips spread out
        if any(abs(ts - (h["startTime"] + (h["endTime"] - h["startTime"]) / 2)) < min_spread
               for h in highlights):
            continue
        
        used.append((start, end))
        highlights.append({
            "startTime": round(start, 1),
            "endTime": round(end, 1),
            "score": ev["finalScore"],
            "eventType": ev["type"],
            "commentary": ev.get("commentary", ""),
            "group_id": ev.get("group_id", -1),
            "narrative_context": True if use_groups else False
        })
    
    return sorted(highlights, key=lambda x: x["startTime"])


def _report_failure(match_id):
    try:
        requests.post(f"{ORCHESTRATOR_URL}/matches/{match_id}/progress", json={"progress": -1}, timeout=3)
    except Exception:
        pass


def emit_live_event(match_id: str, event: dict):
    """
    POST one event immediately to the orchestrator so it can be broadcast via
    WebSocket to any frontend clients watching this match in real-time.
    Failures are silently swallowed (best-effort).
    """
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/live-event",
            json=event,
            timeout=2,
        )
    except Exception:
        pass


# Compression logic moved to app.core.video_utils


def _download_youtube_video(url: str, match_id: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
    """Download YouTube video using yt-dlp to UPLOADS_DIR."""
    import yt_dlp
    
    logger.info(f"Downloading YouTube video: {url} (range: {start_time}-{end_time})")
    # Force mp4 and limit resolution to 720p or 1080p for speed
    out_tmpl = str(UPLOADS_DIR / f"{match_id}_yt.%(ext)s")
    
    from yt_dlp.utils import download_range_func
    
    last_emit_time = 0
    def progress_hook(d):
        nonlocal last_emit_time
        if d['status'] == 'downloading':
            import time
            from urllib.parse import urljoin
            
            # Throttle to emit at most once per second
            now = time.time()
            if now - last_emit_time < 1.0:
                return
                
            last_emit_time = now
            
            # Calculate percentage
            percent = 0.0
            if 'downloaded_bytes' in d and 'total_bytes' in d:
                percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
            elif '_percent_str' in d:
                try:
                    percent = float(d['_percent_str'].strip('%'))
                except:
                    pass
            
            # Map download (0-100%) to overall progress (0-20%)
            # yt-dlp is the first step, so we allocate the first 20% to it visually
            overall_progress = min(20, int(percent * 0.2))
            try:
                requests.post(
                    f"{ORCHESTRATOR_URL}/matches/{match_id}/progress", 
                    json={"progress": overall_progress}, 
                    timeout=1
                )
            except:
                pass

    # If range is specified, use it. Otherwise default to first 30 mins (1800s) as a safety cap.
    start = int(start_time) if start_time is not None else 0
    end = int(end_time) if end_time is not None else 10800 # 3 hours max if not specified
    
    ydl_opts: Dict = {  # type: ignore
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': out_tmpl,
        'quiet': False,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        'download_ranges': download_range_func(None, [(start, end)]),
        'force_keyframes_at_cuts': True,
        'progress_hooks': [progress_hook],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore
            info = ydl.extract_info(url, download=True)
            # Find the actual downloaded filepath (might vary slightly depending on merge)
            dl_path = ydl.prepare_filename(info)
            # Ensure it ends with mp4 since we merged it
            if not dl_path.endswith('.mp4'):
                dl_path = dl_path.rsplit('.', 1)[0] + '.mp4'
                
            if os.path.exists(dl_path):
                logger.info(f"Downloaded YouTube video to: {dl_path}")
                return dl_path
            else:
                # Fallback if the strict filename wasn't found but a file with the ID was
                for f in UPLOADS_DIR.glob(f"{match_id}_yt.*"):
                    return str(f)
                    
        raise Exception("Download completed but file not found")
    except Exception as e:
        logger.error(f"yt-dlp download failed: {e}")
        raise ValueError(f"Failed to download YouTube video: {e}")


# ── Goal Detection Pipeline ────────────────────────────────────────────────────
def detect_goals_in_video(video_path: str) -> list:
    """
    Detect goals in video using vision-based goal-line crossing detection.
    Returns list of goal events: [{"timestamp": float, "type": "GOAL", ...}]
    """
    if not GOAL_DETECTION_AVAILABLE or not GoalDetectionEngine:
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video for goal detection: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize goal detector with Roboflow support
        rf_cfg = {
            "api_key": CONFIG["ROBOFLOW_API_KEY"],
            "workspace": CONFIG["ROBOFLOW_WORKSPACE"],
            "project": CONFIG["ROBOFLOW_PROJECT"],
            "version": CONFIG["ROBOFLOW_VERSION"]
        }
        goal_engine = GoalDetectionEngine(roboflow_cfg=rf_cfg)
        goal_engine.init(frame_width, frame_height, fps)
        
        logger.info(f"Goal detection: {frame_width}×{frame_height} @ {fps:.1f}fps")
        
        goals_detected = []
        frame_idx = 0
        
        # Process every Nth frame to speed up (goal detection doesn't need every frame)
        frame_step = max(1, int(fps / 5.0))  # Process at ~5 FPS
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames
            if frame_idx % frame_step != 0:
                continue
            
            # Resize for faster processing if very large
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)))
            
            # Process frame
            goal_event = goal_engine.process_frame(frame)
            
            if goal_event:
                goals_detected.append({
                    "timestamp": round(goal_event.timestamp, 2),
                    "type": "GOAL",
                    "confidence": round(goal_event.confidence, 3),
                    "description": f"Goal detected ({goal_event.direction})",
                    "source": "goal_detection"
                })
                logger.info(f"⚽ GOAL at {goal_event.timestamp:.1f}s | confidence: {goal_event.confidence:.2f}")
        
        cap.release()
        
        logger.info(f"Goal detection completed: {len(goals_detected)} goals found")
        return goals_detected
        
    except Exception as e:
        logger.error(f"Goal detection failed: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────
def analyze_video(video_path: str, match_id: str, start_time: Optional[float] = None, end_time: Optional[float] = None, language: str = "english", aspect_ratio: str = "16:9"):
    if not match_id or not isinstance(match_id, str):
        logger.error("Invalid match_id")
        return {"error": "Invalid match_id"}
        
    logger.info(f"Starting analysis: match={match_id}, source={video_path}, range={start_time}-{end_time}")
    
    # 1. Download if it's a YouTube URL
    if video_path.startswith("http://") or video_path.startswith("https://"):
        try:
            # Emit progress indicating we are downloading
            requests.post(f"{ORCHESTRATOR_URL}/matches/{match_id}/progress", json={"progress": 5}, timeout=3)
            video_path = _download_youtube_video(video_path, match_id, start_time=start_time, end_time=end_time)
        except Exception as e:
            logger.error(str(e))
            _report_failure(match_id)
            return {"error": "Failed to download YouTube video"}

    # 2. Proceed with local file validation
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        _report_failure(match_id)
        return {"error": "Video file not found"}
    
    # Pre-compress large videos for faster processing
    original_video_path = video_path
    video_path = _precompress_video(video_path, match_id, CONFIG)
    compressed = (video_path != original_video_path)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            _report_failure(match_id)
            return {"error": "Could not open video"}

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
        frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        duration     = total_frames / fps if (fps > 0 and total_frames > 0) else 0.0
        is_stream    = (total_frames <= 0)
        
        # If we pre-compressed to 1fps, adjust settings
        if compressed:
            process_fps = fps
            frame_step = 1
        else:
            process_fps = CONFIG["VIDEO_PROCESS_FPS"]
            frame_step = max(1, int(fps / process_fps))
            
        track_interval = 1
        window_frames = max(1, int(process_fps * CONFIG["MOTION_WINDOW_SECS"]))
        target_height = CONFIG["YOLO_DOWNSCALE_HEIGHT"] if frame_h > 720 else frame_h

        logger.info(f"Video: {total_frames}f @ {fps:.1f}fps = {duration:.1f}s [{frame_w}×{frame_h}] → {target_height}p @ {process_fps}fps")

        frame_count  = 0
        processed_count = 0
        prev_gray    = None
        window_diffs = []

        motion_windows: list = []
        track_frames:   list = []
        jersey_colours: list = []   # [[R,G,B], ...] one per sampled person crop
        frame_person_rows: list = []  # parallel to track_frames, raw persons before team assignment

        # Initialize Goal Detection Engine
        _goal_engine = None
        if GOAL_DETECTION_AVAILABLE and GoalDetectionEngine:
            try:
                rf_cfg = {
                    "api_key": CONFIG["ROBOFLOW_API_KEY"],
                    "workspace": CONFIG["ROBOFLOW_WORKSPACE"],
                    "project": CONFIG["ROBOFLOW_PROJECT"],
                    "version": CONFIG["ROBOFLOW_VERSION"]
                }
                _goal_engine = GoalDetectionEngine(roboflow_cfg=rf_cfg)
                _goal_engine.init(frame_w, frame_h, fps)
                logger.info("GoalDetectionEngine initialised ✓")
            except Exception as _ge:
                logger.warning(f"GoalDetectionEngine init failed: {_ge}")
                _goal_engine = None

        # Initialize Goalpost Detection
        _goalpost_detector = None
        _goalpost_tracker = None
        goalpost_detections: list = []
        if GOALPOST_DETECTION_AVAILABLE and GoalpostDetector and GoalpostTracker:
            try:
                _goalpost_detector = GoalpostDetector()
                _goalpost_tracker = GoalpostTracker(max_distance=100.0)
                logger.info("GoalpostDetector initialised ✓")
            except Exception as _gpe:
                logger.warning(f"GoalpostDetector init failed: {_gpe}")
                _goalpost_detector = None
                _goalpost_tracker = None

        # NOTE: raw_events and last_seen are no longer used here
        # Events are now detected via Vision AI after the main loop

        while cap.isOpened():
            # Fast-forward: skip decoding frames we don't need
            for _ in range(frame_step - 1):
                cap.grab()
                frame_count += 1
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            processed_count += 1
            timestamp = frame_count / fps

            # Downscale frame if it's very large (e.g., 1080p/4K) to speed up YOLO and motion diff
            h, w = frame.shape[:2]
            if w > 800:
                scale = 800 / w
                frame = cv2.resize(frame, (800, int(h * scale)))
                # Update frame_w and frame_h for normalized coordinates
                frame_w, frame_h = 800, int(h * scale)

            # ── Motion window ────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                window_diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
            prev_gray = gray

            if len(window_diffs) >= window_frames:
                raw_max = float(np.percentile(window_diffs, 90)) if len(window_diffs) > 1 else window_diffs[0]
                m_score = round(min(raw_max / 40.0, 1.0), 3)
                motion_windows.append({
                    "timestamp":   round(timestamp - 5.0, 1),
                    "motionScore": m_score,
                    "audioScore":  round(min(m_score * 1.2, 1.0), 3),
                })
                
                # ── Live Event Detection (Sliding Window) ──────────────────
                # If we are in stream mode or have high intensity, check now
                if m_score >= CONFIG["MOTION_PEAK_THRESHOLD"]:
                    # Grab current frame for analysis
                    analysis_res = analyze_frame_with_vision(frame, timestamp, context=f"Language: {language}")
                    if analysis_res["event_type"] != "NONE" and analysis_res["confidence"] >= 0.5:
                        live_evt = {
                            "timestamp": round(timestamp, 2),
                            "type": analysis_res["event_type"],
                            "confidence": round(analysis_res["confidence"], 3),
                            "commentary": analysis_res.get("description", ""),
                            "finalScore": round(m_score * 10, 1),
                            "source": "live_sliding_window"
                        }
                        emit_live_event(match_id, live_evt)
                        logger.info(f"✨ LIVE EVENT: {live_evt['type']} at {live_evt['timestamp']}s")

                window_diffs = []

            # ── Progress update ──────────────────────────────────────────────
            if processed_count % 5 == 0 and total_frames > 0:
                try:
                    requests.post(
                        f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                        json={"progress": int((frame_count / total_frames) * 99)},
                        timeout=1,
                    )
                except Exception:
                    pass

            # ── YOLO detection & tracking ────────────────────────────────────
            current_motion = _get_motion_at(motion_windows, timestamp) if motion_windows else 0.3
            
            # ── Adaptive Frame Skipping ──────────────────────────────────────
            # Skip YOLO on very static scenes, but be more aggressive on low-motion periods
            skip_yolo = False
            if CONFIG["SMART_FRAME_SKIP"]:
                # Skip on low motion, but occasionally sample for context
                if current_motion < CONFIG["YOLO_SKIP_MOTION_THRESHOLD"]:
                    # Skip most frames, but sample every 5 to catch sudden changes
                    if processed_count % 5 != 0:
                        skip_yolo = True
                # Medium motion: normal skip interval
                elif current_motion < 0.3:
                    skip_yolo = (processed_count % CONFIG["YOLO_SKIP_MOTION_INTERVAL"] != 0)
                # High motion: process every frame (no skip)
            else:
                # Legacy frame skipping
                skip_yolo = current_motion < CONFIG["YOLO_SKIP_MOTION_THRESHOLD"] and processed_count % CONFIG["YOLO_SKIP_MOTION_INTERVAL"] != 0
            
            frame_balls:   list = []
            frame_persons: list = []
            
            if not skip_yolo:
                # Downscale frame for faster YOLO inference
                yolo_frame = frame
                if frame.shape[0] > target_height:
                    scale = target_height / frame.shape[0]
                    yolo_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                # Check cache first
                cached_result = _frame_cache.get(yolo_frame) if CONFIG["ENABLE_INFERENCE_CACHING"] else None
                
                if cached_result:
                    results, ball_results = cached_result
                    logger.debug(f"Cache hit for frame {processed_count}")
                else:
                    try:
                        # Use GPU for inference if available
                        if GPU_AVAILABLE:
                            results = model(yolo_frame, verbose=False, device=0)
                            ball_results = ball_model(yolo_frame, classes=[32], verbose=False, device=0)
                        else:
                            results = model(yolo_frame, verbose=False)
                            ball_results = ball_model(yolo_frame, classes=[32], verbose=False)
                        
                        # Cache the results
                        if CONFIG["ENABLE_INFERENCE_CACHING"]:
                            _frame_cache.put(yolo_frame, (results, ball_results))
                    except Exception as e:
                        logger.warning(f"YOLO inference failed: {e}")
                        results = []
                        ball_results = []

                # Scale factor for converting YOLO coords back to original frame size
                scale_factor = frame.shape[0] / yolo_frame.shape[0] if frame.shape[0] != yolo_frame.shape[0] else 1.0

                for r in ball_results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [v * scale_factor for v in box.xyxy[0].tolist()]
                        nx, ny = round(x1 / frame_w, 4), round(y1 / frame_h, 4)
                        nw, nh = round((x2 - x1) / frame_w, 4), round((y2 - y1) / frame_h, 4)
                        frame_balls.append([nx, ny, nw, nh, round(conf, 3)])

                for r in results:
                    if r.boxes is None:
                        continue
                    for i_box, box in enumerate(r.boxes):
                        cls   = int(box.cls[0])
                        conf  = float(box.conf[0])
                        label = model.names[cls]

                        # Only track person with pose model
                        if label != "person":
                            continue

                        x1, y1, x2, y2 = [v * scale_factor for v in box.xyxy[0].tolist()]

                        # Store normalised coords for canvas overlay
                        nx = round(x1 / frame_w, 4)
                        ny = round(y1 / frame_h, 4)
                        nw = round((x2 - x1) / frame_w, 4)
                        nh = round((y2 - y1) / frame_h, 4)

                        tid = -1
                        if hasattr(box, "id") and box.id is not None:
                            tid = int(box.id[0])

                        # Extract jersey colour → stored temporarily as [nx,ny,nw,nh,tid, r,g,b]
                        # After the loop we replace r,g,b with team index
                        crop = _crop_jersey(frame, nx, ny, nx + nw, ny + nh)
                        col  = _dominant_colour(crop) or [128, 128, 128]
                        jersey_colours.append(col)
                        
                        kps = []
                        if hasattr(r, 'keypoints') and r.keypoints is not None and r.keypoints.xyn is not None:
                            if len(r.keypoints.xyn) > i_box:
                                kps = [round(float(v), 4) for v in r.keypoints.xyn[i_box].flatten().tolist()]
                        
                        p_data = [nx, ny, nw, nh, tid, col[0], col[1], col[2]] + kps
                        frame_persons.append(p_data)
                        
                        # NOTE: We no longer create events from YOLO detections
                        # Events are now detected using Gemini Vision analysis
                        # YOLO is only used for ball/player tracking visualization

            # ── GoalDetectionEngine per-frame ────────────────────────────────────
            if _goal_engine is not None:
                try:
                    _goal_engine.process_frame(frame)
                except Exception as _gfe:
                    logger.debug(f"GoalDetectionEngine error: {_gfe}")

            # ── GoalpostDetector per-frame ────────────────────────────────────
            if _goalpost_detector is not None and _goalpost_tracker is not None:
                try:
                    detection = _goalpost_detector.detect(frame, frame_id=frame_count, timestamp=timestamp)
                    if detection:
                        tracked_detection = _goalpost_tracker.update(detection)
                        if tracked_detection:
                            goalpost_detections.append({
                                "frame": frame_count,
                                "timestamp": round(timestamp, 2),
                                "center_x": round(tracked_detection.center_x, 1),
                                "center_y": round(tracked_detection.center_y, 1),
                                "goal_width": round(tracked_detection.goal_width, 1),
                                "confidence": round(tracked_detection.confidence, 3),
                                "has_left": tracked_detection.left is not None,
                                "has_right": tracked_detection.right is not None,
                            })
                except Exception as _gpe:
                    logger.debug(f"GoalpostDetector error: {_gpe}")

            # Store tracking frame on every detection tick (max every track_interval)
            if (processed_count % track_interval == 0) and (frame_balls or frame_persons):
                track_frames.append({
                    "t": round(timestamp, 2),
                    "b": frame_balls[:4],      # ≤4 balls (sports balls)
                    "p": frame_persons[:25],   # ≤25 players (full pitch)
                })

        cap.release()

        # Flush remaining motion window
        if window_diffs:
            raw_max = float(np.percentile(window_diffs, 90)) if len(window_diffs) > 1 else window_diffs[0]
            m_score = round(min(raw_max / 40.0, 1.0), 3)
            motion_windows.append({
                "timestamp":   round((frame_count - len(window_diffs)) / fps, 1),
                "motionScore": m_score,
                "audioScore":  round(min(m_score * 1.2, 1.0), 3),
            })

        # ── Team colour clustering ────────────────────────────────────────────
        # Cluster all jersey colours into 2 teams, then replace the temporary
        # [r,g,b] stored per person with its team index {0, 1}.
        team_colors = [[220, 60, 60], [60, 100, 220]]   # fallback: red / blue
        if len(jersey_colours) >= 4:
            try:
                centroids, _ = _cluster_teams(jersey_colours, n=2)
                team_colors  = centroids
                logger.info(f"Team colours detected: {team_colors}")
            except Exception as e:
                logger.warning(f"Team clustering failed: {e}")

        def _assign_team(r: int, g: int, b: int) -> int:
            """Return 0 or 1 — whichever centroid [r,g,b] is closest to."""
            col = np.array([r, g, b], dtype=float)
            dists = [np.linalg.norm(col - np.array(c)) for c in team_colors]
            return int(np.argmin(dists))

        # Replace [nx,ny,nw,nh,tid, r,g,b] → [nx,ny,nw,nh,tid, team]
        for tf in track_frames:
            labelled = []
            for p in tf.get("p", []):
                if len(p) == 8:   # has colour channels
                    team = _assign_team(int(p[5]), int(p[6]), int(p[7]))
                    labelled.append([p[0], p[1], p[2], p[3], p[4], team])
                elif len(p) >= 5:
                    labelled.append(list(p[:5]) + [0])
                else:
                    labelled.append(p)
            tf["p"] = labelled

        # ── PHASE 2: Smooth ball trajectory for visual quality ────────────────
        if CONFIG["ENHANCED_BALL_TRACKING"]:
            logger.info("Smoothing ball trajectories...")
            track_frames = smooth_ball_trajectory(track_frames, window_size=CONFIG["BALL_SMOOTHING_WINDOW"])

        # ══════════════════════════════════════════════════════════════════════
        # ██ SOCCERNET EVENT DETECTION (football-specific trained model) ██
        # ══════════════════════════════════════════════════════════════════════
        logger.info("Phase 2: SoccerNet football event detection...")
        
        raw_events = []
        
        # Add goals from GoalDetectionEngine (Kalman + Homography + FSM)
        if _goal_engine is not None:
            from app.core.goal_detection import goal_events_to_raw
            _raw_goals = goal_events_to_raw(_goal_engine.goals)
            if _raw_goals:
                logger.info(f"GoalDetectionEngine found {len(_raw_goals)} goal(s)")
                raw_events.extend(_raw_goals)
        
        # Primary: Use SoccerNet (trained on football footage)
        if SOCCERNET_AVAILABLE and detect_football_events:
            try:
                logger.info("Running SoccerNet analysis on original video...")
                soccernet_events = detect_football_events(original_video_path, sensitivity=1.0)
                
                if soccernet_events:
                    for ev in soccernet_events:
                        raw_events.append({
                            "timestamp": ev["timestamp"],
                            "type": ev["type"],
                            "confidence": ev["confidence"],
                            "description": f"SoccerNet detected {ev['type'].lower()}",
                            "source": "soccernet"
                        })
                    logger.info(f"SoccerNet detected {len(raw_events)} events")
                else:
                    logger.warning("SoccerNet returned no events, falling back to motion analysis")
                    
            except Exception as e:
                logger.error(f"SoccerNet analysis failed: {e}")
                
        # Secondary: CV Physics Detector (Runs directly on YOLO tracking data)
        if CV_PHYSICS_AVAILABLE and detect_cv_physics:
            try:
                logger.info("Running CV Physics analysis on track frames...")
                cv_events = detect_cv_physics(track_frames, fps=process_fps)
                
                if cv_events:
                    # Merge with existing events (avoiding duplicates within 5s)
                    existing_times = {ev["timestamp"] for ev in raw_events}
                    for ev in cv_events:
                        if not any(abs(ev["timestamp"] - et) < 5.0 for et in existing_times):
                            raw_events.append({
                                "timestamp": ev["timestamp"],
                                "type": ev["type"],
                                "confidence": ev["confidence"],
                                "description": f"CV Physics detected {ev['type'].lower()}",
                                "source": "cv_physics"
                            })
                            existing_times.add(ev["timestamp"])
                    logger.info(f"CV Physics added {len(cv_events)} events")
            except Exception as e:
                logger.error(f"CV Physics analysis failed: {e}")
        
        # Fallback: Motion-based highlights if SoccerNet fails or returns too few
        if len(raw_events) < 3:
            logger.info("Supplementing with motion-based highlight detection...")
            
            # Find high-motion peaks as generic highlights
            candidate_timestamps = find_motion_peaks(
                motion_windows, 
                threshold=CONFIG["MOTION_FALLBACK_THRESHOLD"],
                min_gap=CONFIG["MOTION_FALLBACK_MIN_GAP"]
            )
            
            # Filter out timestamps that are too close to existing events
            existing_times = {ev["timestamp"] for ev in raw_events}
            for candidate_t in candidate_timestamps:
                # Skip if within 15s of an existing event
                if any(abs(candidate_t - et) < 15 for et in existing_times):
                    continue
                    
                motion_score = _get_motion_at(motion_windows, candidate_t) if motion_windows else 0.5
                
                # Only add very high motion moments as generic highlights
                if motion_score >= CONFIG["MOTION_FALLBACK_THRESHOLD"]:
                    raw_events.append({
                        "timestamp": round(candidate_t, 2),
                        "type": "HIGHLIGHT",
                        "confidence": round(min(0.7, motion_score), 3),
                        "description": "High-action moment",
                        "source": "motion_fallback"
                    })
                    existing_times.add(candidate_t)
                    
                    if len(raw_events) >= CONFIG["MAX_MOTION_BASED_EVENTS"]:  # Cap at 8 total events
                        break
                        
            logger.info(f"Total events after fallback: {len(raw_events)}")
        
        # Sort by timestamp
        raw_events.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"Final event detection: {len(raw_events)} events")
        
        # ── Score events & emit live ─────────────────────────────────────────
        scored_events: list = []
        for ev in raw_events:
            m_score   = _get_motion_at(motion_windows, ev["timestamp"])
            fs        = compute_context_score(ev["type"], m_score, ev["timestamp"], duration, ev["confidence"])
            scored_ev = {**ev, "finalScore": fs}
            scored_events.append(scored_ev)

            # Fire-and-forget: sends to NestJS → WebSocket → browser
            emit_live_event(match_id, scored_ev)

        # ── Gemini commentary per event ──────────────────────────────────────
        for i, ev in enumerate(scored_events):
            ctx = scored_events[max(0, i - 3):i] + scored_events[i + 1:i + 3]
            ev["commentary"] = generate_commentary(
                ev["type"], ev["finalScore"], ev["timestamp"], duration, ctx, language=language
            )

        # ── PHASE 2: Highlights with narrative flow ──────────────────────────
        if CONFIG["SMART_HIGHLIGHT_SELECTION"]:
            logger.info("Selecting highlights with narrative context...")
            highlights = select_highlights_with_narrative(
                scored_events, duration, top_n=CONFIG["HIGHLIGHT_COUNT"],
                use_groups=CONFIG["HIGHLIGHT_NARRATIVE_CONTEXT"]
            )
        else:
            highlights = select_highlights(scored_events, duration)

        highlight_reel_url = None
        try:
            logger.info("Generating highlight reel with TTS, music, and crowd noise...")
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            # Use original video for highlight reel (compressed is 1fps)
            highlight_reel_url = create_highlight_reel(
                video_path=original_video_path,
                highlights=highlights,
                match_id=match_id,
                output_dir=str(UPLOADS_DIR),
                music_dir=MUSIC_DIR,
                tracking_data=track_frames,
                aspect_ratio=aspect_ratio,
                language=language
            )
        except Exception as e:
            logger.warning(f"Highlight reel generation skipped: {e}")


        # ── Heatmap Generation ────────────────────────────────────────────────
        heatmap_url = None
        top_speed_kmh = 0.0
        if HEATMAP_AVAILABLE and generate_heatmap and track_frames:
            try:
                heatmap_filename = f"heatmap_{match_id}.png"
                heatmap_path = str(UPLOADS_DIR / heatmap_filename)
                success = generate_heatmap(
                    track_frames=track_frames,
                    output_path=heatmap_path,
                    team_colors_rgb=team_colors,
                )
                if success:
                    heatmap_url = f"/uploads/{heatmap_filename}"
                    logger.info(f"Heatmap generated: {heatmap_url}")
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")

            try:
                if estimate_ball_speed:
                    top_speed_kmh = estimate_ball_speed(track_frames, fps)
                    logger.info(f"Top ball speed: {top_speed_kmh:.1f} km/h")
            except Exception as e:
                logger.warning(f"Ball speed estimation failed: {e}")

        # ── PHASE 2: Context-aware analysis ──────────────────────────────────
        formation_data = {}
        trajectory_data = {}
        audio_volumes = {}
        
        if CONFIG["CONTEXT_AWARE_COMMENTARY"]:
            logger.info("Analyzing team formation for context-aware commentary...")
            formation_data = analyze_team_formation(track_frames, team_colors)
            logger.info(f"Formation: {formation_data.get('formation')} | Cohesion: {formation_data.get('cohesion')}")
        
        if CONFIG["ENHANCED_BALL_TRACKING"]:
            # Get ball trajectory data
            all_balls = []
            for frame in track_frames:
                if frame.get("b"):
                    all_balls.extend(frame["b"])
            if all_balls:
                trajectory_data = predict_ball_trajectory(all_balls, fps=process_fps)
                logger.info(f"Ball trajectory: {trajectory_data.get('direction')} @ {trajectory_data.get('speed')} speed")
        
        # ── Emotion scores ────────────────────────────────────────────────────
        emotion_scores = [
            {
                "timestamp":     w["timestamp"],
                "audioScore":    w["audioScore"],
                "motionScore":   w["motionScore"],
                "contextWeight": round(time_context_weight(w["timestamp"], duration), 3),
                "finalScore":    round(
                    (w["audioScore"] * 0.3 + w["motionScore"] * 0.5 +
                     time_context_weight(w["timestamp"], duration) * 0.2) * 10, 2
                ),
            }
            for w in motion_windows
        ]

        if CONFIG["DYNAMIC_AUDIO_MIXING"] and len(emotion_scores) > 0:
            logger.info("Calculating dynamic audio volumes...")
            # Use average emotion score for audio mixing
            avg_emotion = float(np.mean([e["finalScore"] for e in emotion_scores]))
            avg_motion = float(np.mean([e["motionScore"] for e in emotion_scores]))
            audio_volumes = calculate_dynamic_audio_volumes(avg_motion, avg_emotion)
            logger.info(f"Audio volumes: {audio_volumes}")

        # ── Gemini match summary ──────────────────────────────────────────────
        logger.info("Generating Gemini match summary…")
        summary = generate_match_summary(scored_events, highlights, duration, language=language)
        if not summary:
            summary = f"Match analysis completed. {len(scored_events)} events detected across {round(duration)}s of footage."
        logger.info(f"Summary: {len(summary)} chars")

        # ── Performance Stats ────────────────────────────────────────────────
        cache_stats = _frame_cache.stats() if CONFIG["ENABLE_INFERENCE_CACHING"] else {}
        logger.info(
            f"Done: {len(scored_events)} events | {len(highlights)} highlights | "
            f"{len(track_frames)} tracking frames | {duration:.1f}s"
        )
        logger.info(
            f"Performance: GPU={GPU_AVAILABLE} | "
            f"Frames={frame_count} | Smart Skip={CONFIG['SMART_FRAME_SKIP']} | "
            f"Cache={cache_stats.get('hit_rate', 'N/A')}"
        )

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):  # type: ignore
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (float, np.floating)):  # type: ignore
                return float(obj)
            elif isinstance(obj, (int, np.integer)):  # type: ignore
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        # ── Thumbnail Generation ─────────────────────────────────────────────
        thumbnail_url = None
        try:
            cap_thumb = cv2.VideoCapture(original_video_path)
            if not cap_thumb.isOpened():
                logger.error(f"Failed to open video for thumbnail: {original_video_path}")
            else:
                midpoint_frame_idx = total_frames // 2
                cap_thumb.set(cv2.CAP_PROP_POS_FRAMES, midpoint_frame_idx)
                ret_t, thumb_frame = cap_thumb.read()
                
                # Fallback to first frame if midpoint fails
                if not ret_t:
                    logger.warning("Midpoint frame read failed, falling back to first frame")
                    cap_thumb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_t, thumb_frame = cap_thumb.read()
                
                if ret_t:
                    # Resize to standard 720p width if larger, maintain aspect ratio
                    th, tw = thumb_frame.shape[:2]
                    if tw > 1280:
                        t_scale = 1280 / tw
                        thumb_frame = cv2.resize(thumb_frame, (1280, int(th * t_scale)))
                    
                    thumbnail_filename = f"thumbnail_{match_id}.jpg"
                    thumbnail_path = str(UPLOADS_DIR / thumbnail_filename)
                    success = cv2.imwrite(thumbnail_path, thumb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if success:
                        thumbnail_url = f"/uploads/{thumbnail_filename}"
                    else:
                        logger.error(f"cv2.imwrite failed for {thumbnail_path}")
                else:
                    logger.error("Failed to read any frames for thumbnail")
            cap_thumb.release()
        except Exception as te:
            logger.warning(f"Thumbnail generation system error: {te}")

        payload = {
            "events":        convert_numpy(scored_events),
            "highlights":    convert_numpy(highlights),
            "emotionScores": convert_numpy(emotion_scores),
            "duration":      round(float(duration), 1),
            "summary":       summary,
            "highlightReelUrl": highlight_reel_url,
            "thumbnailUrl":  thumbnail_url,
            "trackingData":  convert_numpy(track_frames),
            "teamColors":    convert_numpy(team_colors),   # [[R,G,B],[R,G,B]] team0 / team1
            "heatmapUrl":    heatmap_url,
            "topSpeedKmh":   round(float(top_speed_kmh), 1),
            "videoUrl":      f"http://localhost:4000/uploads/{Path(original_video_path).name}",
            "goalpostDetections": convert_numpy(goalpost_detections),  # Spatial awareness data
            # ── PHASE 2 Analysis Data ────────────────────────────────────────
            "formationData": formation_data,  # Team formation & cohesion
            "trajectoryData": trajectory_data,  # Ball trajectory analysis
            "audioVolumes": audio_volumes,  # Dynamic audio mixing parameters
        }

        try:
            resp = requests.post(
                f"{ORCHESTRATOR_URL}/matches/{match_id}/complete",
                json=payload,
                timeout=30,
            )
            logger.info(f"Complete sent — HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to send completion: {e}")
            try:
                requests.post(
                    f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                    json={"progress": 100}, timeout=5
                )
            except Exception:
                pass

        # Cleanup compressed file if we created one
        if compressed and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

        return {"status": "completed", "match_id": match_id}

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        # Cleanup compressed file on error too
        if compressed and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
        _report_failure(match_id)
        return {"error": str(e)}

