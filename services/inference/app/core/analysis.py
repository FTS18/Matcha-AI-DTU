import cv2
import logging
import sys
import requests
import os
import numpy as np
import torch
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local modular imports
from app.core.frame_cache import FrameCache
from app.core.vision_validator import (
    validate_candidate_with_fallback,
    find_motion_peaks,
)
from app.core.spatial_analysis import (
    smooth_ball_trajectory,
    predict_ball_trajectory,
    analyze_team_formation,
)
from app.core.team_detector import _crop_jersey, _dominant_colour, _cluster_teams
from app.core.downloader import _download_youtube_video as _download_youtube_video_base
from app.core.llm import (
    analyze_frame_with_vision,
    analyze_frames_batch,
    generate_commentary,
    generate_commentary_parallel,
    generate_match_summary,
    _get_gemini,
)
from app.core.tts import tts_generate, get_tts_available as _get_tts
from app.core.video_utils import (
    generate_silent_audio as _generate_silent_audio,
    create_highlight_reel,
    precompress_video as _precompress_video,
)
from app.core.scoring.engine import (
    compute_context_score,
    score_goal,
    score_save,
    score_foul,
)
from app.core.audio_engine import calculate_dynamic_audio_volumes
from app.core.highlight_manager import (
    select_highlights,
    select_highlights_with_narrative,
    group_related_events,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure ~/bin is on PATH
_home_bin = os.path.join(os.path.expanduser("~"), "bin")
if _home_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _home_bin + os.pathsep + os.environ.get("PATH", "")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOADS_DIR = BASE_DIR.parent.parent / "uploads"
MUSIC_DIR = BASE_DIR / "app" / "music"
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
    "GOAL_DETECTION_ENABLED": True,
    "GOAL_DETECTION_MIN_FRAMES": 3,
    "GOAL_DETECTION_MIN_SIZE": 10,
    "GOAL_DETECTION_MAX_SIZE": 200,
    "GOAL_DETECTION_CONFIDENCE_THRESHOLD": 0.5,
    "ROBOFLOW_API_KEY": os.getenv("ROBOFLOW_API_KEY"),
    "ROBOFLOW_WORKSPACE": os.getenv("ROBOFLOW_WORKSPACE", "matcha-ai"),
    "ROBOFLOW_PROJECT": os.getenv("ROBOFLOW_PROJECT", "soccer-ball-detection"),
    "ROBOFLOW_VERSION": int(os.getenv("ROBOFLOW_VERSION", "1")),
    "STREAM_BUFFER_SIZE": 15,
    "STREAM_MAX_IDLE_SECS": 300,
    "LIVE_EMIT_INTERVAL_SECS": 1.0,
    "ENABLE_GPU_ACCELERATION": True,
    "PARALLEL_WORKERS": 4,
    "BATCH_YOLO_SIZE": 8,
    "SMART_FRAME_SKIP": True,
    "MOTION_CACHE_ENABLED": True,
    "ENABLE_INFERENCE_CACHING": True,
    "CACHE_SIMILARITY_THRESHOLD": 0.95,
    "ENHANCED_BALL_TRACKING": True,
    "BALL_SMOOTHING_WINDOW": 3,
    "CONTEXT_AWARE_COMMENTARY": True,
    "DYNAMIC_AUDIO_MIXING": True,
    "SMART_HIGHLIGHT_SELECTION": True,
    "HIGHLIGHT_NARRATIVE_CONTEXT": True,
    "MIN_EVENT_GAP_FOR_GROUPING": 15.0,
}

# Initialize frame cache
_frame_cache = FrameCache(
    max_size=100, similarity_threshold=CONFIG["CACHE_SIMILARITY_THRESHOLD"]
)


def _detect_gpu_availability():
    """Check if GPU is available for acceleration."""
    if not CONFIG["ENABLE_GPU_ACCELERATION"]:
        return False
    try:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            logger.info(f" GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
        return gpu_available
    except Exception:
        pass
    return False


GPU_AVAILABLE = _detect_gpu_availability()


# ── Logic moved to app.core.llm ───────────────────────────────────────────────
def _frame_to_pil(frame):
    from app.core.llm import _frame_to_pil as f2p

    return f2p(frame)


# ── Vision Validation Logic ──────────────────────────────────────────────────
# Moved to app.core.vision_validator


# ── FFmpeg helpers ───────────────────────────────────────────────────────────
# TTS logic moved to app.core.tts

import subprocess
import tempfile

# Highlight logic moved to app.core.video_utils


ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:4000/api/v1")

# ── Modular Imports (Refactored) ─────────────────────────────────────────────
from app.core.scoring.engine import (
    compute_context_score,
    score_goal,
    score_save,
    score_foul,
)
from app.core.audio_engine import calculate_dynamic_audio_volumes
from app.core.highlight_manager import (
    select_highlights,
    select_highlights_with_narrative,
    group_related_events,
)

# Use YOLOv8n-pose for much faster CPU inference (Nano model)
model = YOLO("yolov8n-pose.pt")
ball_model = YOLO("yolov8n.pt")

# ── COCO classes to track for visualization ──────────────────────────────────
# NOTE: YOLO is now ONLY used for ball/player tracking visualization on canvas
# Events are detected via Gemini Vision analysis of candidate moments
YOLO_TRACK_CLASSES = {"sports ball", "person"}

# Legacy mapping - kept for compatibility but NO LONGER used for event creation
YOLO_TYPE_MAP = {"sports ball": "GOAL", "person": "TACKLE"}

# Per-class minimum YOLO confidence thresholds (for tracking visualization)
MIN_CONF: Dict[str, float] = {
    "sports ball": 0.30,  # small + blurry → lower threshold
    "person": 0.50,  # for player tracking
}

# ── Minimum gap between events of the same type (seconds) ────────────────────
# Used by Vision AI event detection phase
MODEL_MIN_GAP: Dict[str, float] = {
    "GOAL": 5.0,
    "TACKLE": 45.0,
    "SAVE": 20.0,
    "FOUL": 15.0,
    "Celebrate": 30.0,
    "PENALTY": 10.0,
    "RED_CARD": 10.0,
    "YELLOW_CARD": 10.0,
    "CORNER": 10.0,
    "OFFSIDE": 10.0,
}
DEFAULT_MIN_GAP = 20.0

# ── Event weight table (out of 10) ───────────────────────────────────────────
# Scoring weights and logic moved to app.core.scoring.engine


# ── Fallback commentary ───────────────────────────────────────────────────────
_FALLBACK = {
    "GOAL": {
        "high": "GOOOAL! Sensational — the crowd erupts!",
        "mid": "Goal! Crucial finish puts them ahead!",
        "low": "Goal scored.",
    },
    "TACKLE": {
        "high": "FEROCIOUS TACKLE! Incredible commitment!",
        "mid": "Strong challenge wins the ball back.",
        "low": "Tackle wins possession.",
    },
    "FOUL": {
        "high": "DEFINITE FOUL! Referee steps in immediately!",
        "mid": "Free kick awarded — bodies flying here.",
        "low": "Foul given.",
    },
    "SAVE": {
        "high": "UNBELIEVABLE SAVE! Superhuman goalkeeping!",
        "mid": "Good stop from the keeper — keeping them in it.",
        "low": "Save made.",
    },
    "CELEBRATION": {
        "high": "INCREDIBLE SCENES! The players are losing their minds!",
        "mid": "Celebrations break out on the pitch!",
        "low": "The players celebrate.",
    },
    "HIGHLIGHT": {
        "high": "WHAT A MOMENT! Crucial action in this match!",
        "mid": "Important moment of play here.",
        "low": "Key moment of play.",
    },
}


def _fallback_commentary(event_type, final_score, timestamp, duration):
    minute = max(1, int(timestamp / 60))
    late = duration > 0 and (timestamp / duration) > 0.85
    energy = "high" if final_score >= 7.5 else ("mid" if final_score >= 5 else "low")
    text = _FALLBACK.get(event_type, {}).get(
        energy, f"{event_type} at minute {minute}."
    )
    if "minute" not in text.lower():
        text = text.rstrip("!.") + f" at minute {minute}."
    if late and final_score >= 7:
        text = "LATE DRAMA! " + text
    return text

    # ── Gemini commentary ─────────────────────────────────────────────────────────
    # Commentary logic moved to app.core.llm

    # ── Gemini match summary ──────────────────────────────────────────────────────
    # Summary logic moved to app.core.llm

    # Highlight selection logic moved to app.core.highlight_manager

    return sorted(highlights, key=lambda x: x["startTime"])


# ── Team colour clustering ────────────────────────────────────────────────────
# ── Team Detector Logic ──────────────────────────────────────────────────
# Moved to app.core.team_detector


def _get_motion_at(windows, timestamp):
    if not windows:
        return 0.3
    return min(windows, key=lambda w: abs(w["timestamp"] - timestamp))["motionScore"]


# ── PHASE 2 OPTIMIZATION 1: ENHANCED BALL TRACKING ──────────────────────────
# ── Spatial Analysis Logic ────────────────────────────────────────────────
# Moved to app.core.spatial_analysis


# Audio mixing logic moved to app.core.audio_engine


# Narrative grouping and smart highlights moved to app.core.highlight_manager


def _report_failure(match_id):
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
            json={"progress": -1, "stage": "failed"},
            timeout=3,
        )
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


_tracking_buffer: list = []
_tracking_buffer_lock = None  # Lazy-init to avoid import-time threading


def emit_tracking_frames(match_id: str, frames: list):
    """
    POST a batch of new tracking frames to the orchestrator for real-time
    overlay updates in the browser. Silently skips on failure.
    """
    if not frames:
        return
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/tracking-update",
            json={"frames": frames},
            timeout=3,
        )
    except Exception:
        pass


# Compression logic moved to app.core.video_utils


def _download_youtube_video(
    url: str,
    match_id: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> str:
    """Download YouTube video using yt-dlp to UPLOADS_DIR."""

    def progress_callback(percent):
        overall_progress = min(20, int(percent * 0.2))
        try:
            requests.post(
                f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                json={"progress": overall_progress, "stage": "downloading"},
                timeout=1,
            )
        except Exception:
            pass

    return _download_youtube_video_base(
        url=url,
        output_dir=str(UPLOADS_DIR),
        filename_prefix=match_id,
        start_time=start_time,
        end_time=end_time,
        progress_callback=progress_callback,
    )


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
            "version": CONFIG["ROBOFLOW_VERSION"],
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
                goals_detected.append(
                    {
                        "timestamp": round(goal_event.timestamp, 2),
                        "type": "GOAL",
                        "confidence": round(goal_event.confidence, 3),
                        "description": f"Goal detected ({goal_event.direction})",
                        "source": "goal_detection",
                    }
                )
                logger.info(
                    f" GOAL at {goal_event.timestamp:.1f}s | confidence: {goal_event.confidence:.2f}"
                )

        cap.release()

        logger.info(f"Goal detection completed: {len(goals_detected)} goals found")
        return goals_detected

    except Exception as e:
        logger.error(f"Goal detection failed: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────
def analyze_video(
    video_path: str,
    match_id: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    language: str = "english",
    aspect_ratio: str = "16:9",
):
    if not match_id or not isinstance(match_id, str):
        logger.error("Invalid match_id")
        return {"error": "Invalid match_id"}

    logger.info(
        f"Starting analysis: match={match_id}, source={video_path}, range={start_time}-{end_time}"
    )

    # 1. Download if it's a YouTube URL
    if video_path.startswith("http://") or video_path.startswith("https://"):
        try:
            # Emit progress indicating we are downloading
            requests.post(
                f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                json={"progress": 5, "stage": "downloading"},
                timeout=3,
            )
            video_path = _download_youtube_video(
                video_path, match_id, start_time=start_time, end_time=end_time
            )
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
    try:
        requests.post(
            f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
            json={"progress": 22, "stage": "compressing"},
            timeout=1,
        )
    except Exception:
        pass
    video_path = _precompress_video(video_path, match_id, CONFIG)
    compressed = video_path != original_video_path

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            _report_failure(match_id)
            return {"error": "Could not open video"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        duration = total_frames / fps if (fps > 0 and total_frames > 0) else 0.0
        is_stream = total_frames <= 0

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

        # Emit initial scanning stage
        try:
            requests.post(
                f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                json={"progress": 25, "stage": "scanning"},
                timeout=1,
            )
        except Exception:
            pass

        logger.info(
            f"Video: {total_frames}f @ {fps:.1f}fps = {duration:.1f}s [{frame_w}×{frame_h}] → {target_height}p @ {process_fps}fps"
        )

        frame_count = 0
        processed_count = 0
        prev_gray = None
        window_diffs = []

        motion_windows: list = []
        track_frames: list = []
        jersey_colours: list = []  # [[R,G,B], ...] one per sampled person crop
        frame_person_rows: list = (
            []
        )  # parallel to track_frames, raw persons before team assignment

        # Initialize Goal Detection Engine
        _goal_engine = None
        if GOAL_DETECTION_AVAILABLE and GoalDetectionEngine:
            try:
                rf_cfg = {
                    "api_key": CONFIG["ROBOFLOW_API_KEY"],
                    "workspace": CONFIG["ROBOFLOW_WORKSPACE"],
                    "project": CONFIG["ROBOFLOW_PROJECT"],
                    "version": CONFIG["ROBOFLOW_VERSION"],
                }
                # type: ignore
                _goal_engine = GoalDetectionEngine(roboflow_cfg=rf_cfg)
                _goal_engine.init(frame_w, frame_h, fps)
                logger.info("GoalDetectionEngine initialised ")
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
                logger.info("GoalpostDetector initialised ")
            except Exception as _gpe:
                _goalpost_detector = None
                _goalpost_tracker = None

        # Initialize Vision Transformer Action Spotter
        _transformer_spotter = None
        _dynamic_calibrator = None
        vit_frame_buffer = deque(maxlen=16)  # VideoMAE best works with 16-frame clips

        # Initialize Scoreboard Detector (score tracking & goal verification)
        _scoreboard_detector = None
        if SCOREBOARD_DETECTION_AVAILABLE and ScoreboardDetector:
            try:
                _scoreboard_detector = ScoreboardDetector(
                    sample_interval=max(1, int(fps)),  # Check ~1 per second
                    min_confidence=0.45,
                    score_change_cooldown=10.0,
                )
                logger.info("ScoreboardDetector initialised ")
            except Exception as _sbe:
                logger.warning(f"ScoreboardDetector init failed: {_sbe}")
                _scoreboard_detector = None

        if TRANSFORMER_AVAILABLE and TransformerActionSpotter:
            try:
                _transformer_spotter = TransformerActionSpotter()
                _dynamic_calibrator = DynamicPitchCalibrator()  # type: ignore[misc]
                logger.info("Vision Transformer & Auto-Calibrator initialized ")
            except Exception as _te:
                logger.warning(f"Transformer init failed: {_te}")
                _transformer_spotter = None

        # ── Vision API cooldown: avoid burning quota on rapid-fire frames ────
        _last_vision_call_ts = -999.0  # timestamp of last Gemini vision call
        _VISION_COOLDOWN_SECS = 10.0  # minimum seconds between vision API calls

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

            # ── Dynamic Pitch Calibration (Auto-Homography) ────────────────
            current_H = None
            if _dynamic_calibrator is not None and processed_count % 30 == 0:
                current_H = _dynamic_calibrator.calibrate_frame(frame)
                if current_H is not None:
                    logger.debug(" Auto-Calibration Updated (Camera Moved)")

            # ── Vision Transformer (ViT) Buffering & Analysis ────────────────
            if _transformer_spotter is not None:
                # Store frame for temporal analysis (processes a "video cube")
                # Downscale further for speed as ViT typically uses 224x224
                vit_input_frame = cv2.resize(frame, (224, 224))
                vit_frame_buffer.append(vit_input_frame)

                # Analyze every 16-frame window for complex actions
                if len(vit_frame_buffer) == 16:
                    vit_res = _transformer_spotter.spot_actions(
                        list(vit_frame_buffer), fps
                    )
                    if vit_res and vit_res["confidence"] >= 0.65:
                        # Map common kinetics labels to our events if possible
                        vit_label = vit_res["label"].lower()
                        our_type = "HIGHLIGHT"
                        if "goal" in vit_label or "kick" in vit_label:
                            our_type = "GOAL"
                        elif "tackle" in vit_label:
                            our_type = "TACKLE"

                        vit_evt = {
                            "timestamp": round(timestamp, 2),
                            "type": our_type,
                            "confidence": vit_res["confidence"],
                            "commentary": f"Transformer analysis: {vit_res['label']}",
                            "finalScore": round(vit_res["confidence"] * 10, 1),
                            "source": "vision_transformer",
                        }
                        emit_live_event(match_id, vit_evt)
                        logger.info(
                            f" ViT DETECTED: {vit_res['label']} ({our_type}) at {timestamp:.2f}s"
                        )

            # ── Motion window ────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                window_diffs.append(float(np.mean(cv2.absdiff(prev_gray, gray))))
            prev_gray = gray

            if len(window_diffs) >= window_frames:
                raw_max = (
                    float(np.percentile(window_diffs, 90))
                    if len(window_diffs) > 1
                    else window_diffs[0]
                )
                m_score = round(min(raw_max / 40.0, 1.0), 3)
                motion_windows.append(
                    {
                        "timestamp": round(timestamp - 5.0, 1),
                        "motionScore": m_score,
                        "audioScore": round(min(m_score * 1.2, 1.0), 3),
                    }
                )

                # ── Live Event Detection (Sliding Window) ──────────────────
                # If we are in stream mode or have high intensity, check now
                if m_score >= CONFIG["MOTION_PEAK_THRESHOLD"]:
                    # Cooldown: skip if we called vision too recently (saves quota)
                    if (timestamp - _last_vision_call_ts) < _VISION_COOLDOWN_SECS:
                        pass  # skip this window, too close to last call
                    else:
                        # Grab current frame for analysis
                        analysis_res = analyze_frame_with_vision(
                            frame, timestamp, context=f"Language: {language}"
                        )
                        _last_vision_call_ts = timestamp
                        if (
                            analysis_res["event_type"] != "NONE"
                            and analysis_res["confidence"] >= 0.65
                        ):
                            live_evt = {
                                "timestamp": round(timestamp, 2),
                                "type": analysis_res["event_type"],
                                "confidence": round(analysis_res["confidence"], 3),
                                "commentary": analysis_res.get("description", ""),
                                "finalScore": round(m_score * 10, 1),
                                "source": "live_sliding_window",
                            }
                            emit_live_event(match_id, live_evt)
                            logger.info(
                                f" LIVE EVENT: {live_evt['type']} at {live_evt['timestamp']}s"
                            )

                window_diffs = []

            # ── Progress update ──────────────────────────────────────────────
            if processed_count % 5 == 0 and total_frames > 0:
                try:
                    # Frame processing = 0-60% of total progress
                    frame_pct = int((frame_count / total_frames) * 60)
                    logger.info(
                        f"Processing frame {frame_count}/{total_frames} ({frame_pct}%) "
                    )
                    requests.post(
                        f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                        json={"progress": min(frame_pct, 60), "stage": "scanning"},
                        timeout=1,
                    )
                except Exception:
                    pass

            # ── YOLO detection & tracking ────────────────────────────────────
            current_motion = (
                _get_motion_at(motion_windows, timestamp) if motion_windows else 0.3
            )

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
                    skip_yolo = (
                        processed_count % CONFIG["YOLO_SKIP_MOTION_INTERVAL"] != 0
                    )
                # High motion: process every frame (no skip)
                else:
                    pass
            else:
                # Legacy frame skipping
                skip_yolo = (
                    current_motion < CONFIG["YOLO_SKIP_MOTION_THRESHOLD"]
                    and processed_count % CONFIG["YOLO_SKIP_MOTION_INTERVAL"] != 0
                )

            frame_balls: list = []
            frame_persons: list = []

            if not skip_yolo:
                # Downscale frame for faster YOLO inference
                yolo_frame = frame
                if frame.shape[0] > target_height:
                    scale = target_height / frame.shape[0]
                    yolo_frame = cv2.resize(
                        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
                    )

                # Check cache first
                cached_result = (
                    _frame_cache.get(yolo_frame)
                    if CONFIG["ENABLE_INFERENCE_CACHING"]
                    else None
                )

                if (
                    cached_result
                    and isinstance(cached_result, tuple)
                    and len(cached_result) == 2
                ):
                    results, ball_results = cached_result
                    logger.debug(f"Cache hit for frame {processed_count}")
                else:
                    try:
                        # Use GPU for inference if available
                        if GPU_AVAILABLE:
                            results = model(yolo_frame, verbose=False, device=0)
                            ball_results = ball_model(
                                yolo_frame, classes=[32], verbose=False, device=0
                            )
                        else:
                            results = model(yolo_frame, verbose=False)
                            ball_results = ball_model(
                                yolo_frame, classes=[32], verbose=False
                            )

                        # Cache the results
                        if CONFIG["ENABLE_INFERENCE_CACHING"]:
                            _frame_cache.put(yolo_frame, (results, ball_results))
                    except Exception as e:
                        logger.warning(f"YOLO inference failed: {e}")
                        results = []
                        ball_results = []

                # Scale factor for converting YOLO coords back to original frame size
                scale_factor = (
                    frame.shape[0] / yolo_frame.shape[0]
                    if frame.shape[0] != yolo_frame.shape[0]
                    else 1.0
                )

                for r in ball_results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = [
                            v * scale_factor for v in box.xyxy[0].tolist()
                        ]
                        nx, ny = round(x1 / frame_w, 4), round(y1 / frame_h, 4)
                        nw, nh = round((x2 - x1) / frame_w, 4), round(
                            (y2 - y1) / frame_h, 4
                        )
                        frame_balls.append([nx, ny, nw, nh, round(conf, 3)])

                for r in results:
                    if r.boxes is None:
                        continue
                    for i_box, box in enumerate(r.boxes):
                        cls = int(box.cls[0])
                        track_conf = float(box.conf[0])
                        label = model.names[cls]

                        # Only track person with pose model
                        if label != "person":
                            continue

                        x1, y1, x2, y2 = [
                            v * scale_factor for v in box.xyxy[0].tolist()
                        ]

                        # Store normalised coords for canvas overlay
                        nx = round(x1 / frame_w, 4)
                        ny = round(y1 / frame_h, 4)
                        nw = round((x2 - x1) / frame_w, 4)
                        nh = round((y2 - y1) / frame_h, 4)

                        tid = -1
                        if hasattr(box, "id") and box.id is not None:
                            tid = int(box.id[0])

                        # Extract jersey colour
                        crop = _crop_jersey(frame, nx, ny, nx + nw, ny + nh)
                        col = _dominant_colour(crop) or [128, 128, 128]
                        jersey_colours.append(col)

                        kps = []
                        if (
                            hasattr(r, "keypoints")
                            and r.keypoints is not None
                            and r.keypoints.xyn is not None
                        ):
                            if len(r.keypoints.xyn) > i_box:
                                kps = [
                                    round(float(v), 4)
                                    for v in r.keypoints.xyn[i_box].flatten().tolist()
                                ]

                        p_data = [nx, ny, nw, nh, tid, col[0], col[1], col[2]] + kps
                        frame_persons.append(p_data)

            # ── GoalDetectionEngine per-frame ────────────────────────────────────
            if _goal_engine is not None:
                try:
                    _goal_engine.process_frame(frame)
                except Exception as _gfe:
                    logger.debug(f"GoalDetectionEngine error: {_gfe}")

            # ── GoalpostDetector per-frame ────────────────────────────────────
            if _goalpost_detector is not None and _goalpost_tracker is not None:
                try:
                    detection = _goalpost_detector.detect(
                        frame, frame_id=frame_count, timestamp=timestamp
                    )
                    if detection:
                        tracked_detection = _goalpost_tracker.update(detection)
                        if tracked_detection:
                            goalpost_detections.append(
                                {
                                    "frame": frame_count,
                                    "timestamp": round(timestamp, 2),
                                    "center_x": round(tracked_detection.center_x, 1),
                                    "center_y": round(tracked_detection.center_y, 1),
                                    "goal_width": round(
                                        tracked_detection.goal_width, 1
                                    ),
                                    "confidence": round(
                                        tracked_detection.confidence, 3
                                    ),
                                    "has_left": tracked_detection.left is not None,
                                    "has_right": tracked_detection.right is not None,
                                }
                            )
                except Exception as _gpe:
                    logger.debug(f"GoalpostDetector error: {_gpe}")

            # ── ScoreboardDetector per-frame ─────────────────────────────────
            if _scoreboard_detector is not None:
                try:
                    _scoreboard_detector.process_frame(frame, frame_count, timestamp)
                except Exception as _sbe:
                    logger.debug(f"ScoreboardDetector error: {_sbe}")

            # Store tracking frame on every detection tick (max every track_interval)
            if (processed_count % track_interval == 0) and (
                frame_balls or frame_persons
            ):
                new_frame = {
                    "t": round(timestamp, 2),
                    "b": frame_balls[:4],  # ≤4 balls (sports balls)
                    "p": frame_persons[:25],  # ≤25 players (full pitch)
                }
                track_frames.append(new_frame)
                # Emit in batches of 30 frames so browser overlay stays live
                if len(track_frames) % 30 == 0:
                    emit_tracking_frames(match_id, track_frames[-30:])

        cap.release()

        # Flush remaining motion window
        if window_diffs:
            raw_max = (
                float(np.percentile(window_diffs, 90))
                if len(window_diffs) > 1
                else window_diffs[0]
            )
            m_score = round(min(raw_max / 40.0, 1.0), 3)
            motion_windows.append(
                {
                    "timestamp": round((frame_count - len(window_diffs)) / fps, 1),
                    "motionScore": m_score,
                    "audioScore": round(min(m_score * 1.2, 1.0), 3),
                }
            )

        # ── Team colour clustering ────────────────────────────────────────────
        try:
            requests.post(
                f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                json={"progress": 61, "stage": "tracking"},
                timeout=1,
            )
        except Exception:
            pass

        team_colors = [[220, 60, 60], [60, 100, 220]]  # fallback: red / blue
        if len(jersey_colours) >= 4:
            try:
                centroids, _ = _cluster_teams(jersey_colours, n=2)
                team_colors = centroids
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
                if len(p) == 8:  # has colour channels
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
            track_frames = smooth_ball_trajectory(
                track_frames, window_size=CONFIG["BALL_SMOOTHING_WINDOW"]
            )

        # ── Helper for granular progress updates ─────────────────────────────
        def emit_progress(pct: int, stage: str = ""):
            try:
                requests.post(
                    f"{ORCHESTRATOR_URL}/matches/{match_id}/progress",
                    json={"progress": min(pct, 99), "stage": stage},
                    timeout=1,
                )
                if stage:
                    logger.info(f"Progress {pct}%: {stage}")
            except Exception:
                pass

        # ══════════════════════════════════════════════════════════════════════
        # ██ SOCCERNET EVENT DETECTION (football-specific trained model) ██
        # ══════════════════════════════════════════════════════════════════════
        emit_progress(62, "events")
        logger.info("Phase 2: SoccerNet football event detection...")

        raw_events = []

        # Add goals from GoalDetectionEngine
        if _goal_engine is not None:
            from app.core.goal_detection import goal_events_to_raw

            _raw_goals = goal_events_to_raw(_goal_engine.goals)
            if _raw_goals:
                logger.info(f"GoalDetectionEngine found {len(_raw_goals)} goal(s)")
                raw_events.extend(_raw_goals)

        # Primary: Use SoccerNet
        if SOCCERNET_AVAILABLE and detect_football_events:
            try:
                logger.info("Running SoccerNet analysis on original video...")
                soccernet_events = detect_football_events(
                    original_video_path, sensitivity=1.0
                )

                if soccernet_events:
                    for ev in soccernet_events:
                        raw_events.append(
                            {
                                "timestamp": ev["timestamp"],
                                "type": ev["type"],
                                "confidence": ev["confidence"],
                                "description": f"SoccerNet detected {ev['type'].lower()}",
                                "source": "soccernet",
                            }
                        )
                    logger.info(f"SoccerNet detected {len(raw_events)} events")
                else:
                    logger.warning(
                        "SoccerNet returned no events, falling back to motion analysis"
                    )

            except Exception as e:
                logger.error(f"SoccerNet analysis failed: {e}")

        # Secondary: CV Physics Detector
        emit_progress(67, "cv_physics")
        if CV_PHYSICS_AVAILABLE and detect_cv_physics:
            try:
                logger.info("Running CV Physics analysis on track frames...")
                cv_events = detect_cv_physics(track_frames, fps=process_fps)

                if cv_events:
                    # Merge with existing events (avoiding duplicates within 5s)
                    existing_times = {ev["timestamp"] for ev in raw_events}
                    for ev in cv_events:
                        if not any(
                            abs(ev["timestamp"] - et) < 5.0 for et in existing_times
                        ):
                            raw_events.append(
                                {
                                    "timestamp": ev["timestamp"],
                                    "type": ev["type"],
                                    "confidence": ev["confidence"],
                                    "description": f"CV Physics detected {ev['type'].lower()}",
                                    "source": "cv_physics",
                                }
                            )
                            existing_times.add(ev["timestamp"])
                    logger.info(f"CV Physics added {len(cv_events)} events")
            except Exception as e:
                logger.error(f"CV Physics analysis failed: {e}")

        # Fallback: Motion-based highlights
        if len(raw_events) < 3:
            logger.info("Supplementing with motion-based highlight detection...")
            candidate_timestamps = find_motion_peaks(
                motion_windows,
                threshold=CONFIG["MOTION_FALLBACK_THRESHOLD"],
                min_gap=CONFIG["MOTION_FALLBACK_MIN_GAP"],
            )

            existing_times = {ev["timestamp"] for ev in raw_events}
            for candidate_t in candidate_timestamps:
                if any(abs(candidate_t - et) < 15 for et in existing_times):
                    continue

                motion_score = (
                    _get_motion_at(motion_windows, candidate_t)
                    if motion_windows
                    else 0.5
                )

                if motion_score >= CONFIG["MOTION_FALLBACK_THRESHOLD"]:
                    raw_events.append(
                        {
                            "timestamp": round(candidate_t, 2),
                            "type": "HIGHLIGHT",
                            "confidence": round(min(0.7, motion_score), 3),
                            "description": "High-action moment",
                            "source": "motion_fallback",
                        }
                    )
                    existing_times.add(candidate_t)

                if len(raw_events) >= CONFIG["MAX_MOTION_BASED_EVENTS"]:
                    break

            logger.info(f"Total events after fallback: {len(raw_events)}")

        raw_events.sort(key=lambda x: x["timestamp"])

        # ── Scoreboard-based goal verification ───────────────────────────────
        if _scoreboard_detector is not None and _scoreboard_detector.has_scoreboard:
            logger.info(
                f" Scoreboard detected ({_scoreboard_detector.readings_count} readings) — verifying goals..."
            )
            raw_events = _scoreboard_detector.verify_goal_events(
                raw_events, tolerance_sec=15.0
            )
            final_score = _scoreboard_detector.get_current_score()
            if final_score:
                logger.info(
                    f" Final scoreboard reading: {final_score[0]} - {final_score[1]}"
                )
            raw_events.sort(key=lambda x: x["timestamp"])

        # ── Score events & emit live ─────────────────────────────────────────
        emit_progress(72, "scoring")
        scored_events: list = []
        for ev in raw_events:
            m_score = _get_motion_at(motion_windows, ev["timestamp"])
            fs = compute_context_score(
                ev["type"], m_score, ev["timestamp"], duration, ev["confidence"]
            )
            scored_ev = {**ev, "finalScore": fs}
            scored_events.append(scored_ev)
            emit_live_event(match_id, scored_ev)

        # ── Gemini commentary per event (parallel) ───────────────────────────
        from app.core.llm import generate_commentary_parallel

        emit_progress(75, "commentary")
        try:
            scored_events = generate_commentary_parallel(
                scored_events, duration, language=language, max_workers=3
            )
        except Exception as e:
            logger.warning(
                f"Commentary generation timed out or failed: {e} — continuing without commentary"
            )

        # ── PHASE 2: Highlights with narrative flow ──────────────────────────
        emit_progress(80, "highlights")
        if CONFIG["SMART_HIGHLIGHT_SELECTION"]:
            logger.info("Selecting highlights with narrative context...")
            highlights = select_highlights_with_narrative(
                scored_events,
                duration,
                top_n=CONFIG["HIGHLIGHT_COUNT"],
                use_groups=CONFIG["HIGHLIGHT_NARRATIVE_CONTEXT"],
            )
        else:
            highlights = select_highlights(scored_events, duration)

        highlight_reel_url = None
        highlight_reel_portrait_url = None
        try:
            emit_progress(83, "reel")
            logger.info("Generating highlight reel (16:9 landscape)... ")
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            landscape_result = create_highlight_reel(
                video_path=original_video_path,
                highlights=highlights,
                match_id=match_id,
                output_dir=str(UPLOADS_DIR),
                music_dir=MUSIC_DIR,
                tracking_data=track_frames,
                aspect_ratio="16:9",
                language=language,
            )
            if isinstance(landscape_result, dict):
                clip_urls = landscape_result.get("clip_urls", [])
                highlight_reel_url = landscape_result.get("reel_url")
                for i, h in enumerate(highlights):
                    clip_url = clip_urls[i] if i < len(clip_urls) else None
                    if clip_url:
                        h["videoUrl"] = clip_url
                logger.info(
                    f"Per-clip URLs assigned to {sum(1 for h in highlights if h.get('videoUrl'))} highlights"
                )
        except Exception as e:
            logger.warning(f"Highlight reel (landscape) generation skipped: {e}")

        try:
            emit_progress(85, "reel")
            logger.info("Generating highlight reel (9:16 portrait)... ")
            portrait_result = create_highlight_reel(
                video_path=original_video_path,
                highlights=highlights,
                match_id=match_id,
                output_dir=str(UPLOADS_DIR),
                music_dir=MUSIC_DIR,
                tracking_data=track_frames,
                aspect_ratio="9:16",
                language=language,
            )
            if isinstance(portrait_result, dict):
                highlight_reel_portrait_url = portrait_result.get("reel_url")
                if highlight_reel_portrait_url:
                    logger.info(
                        f"Portrait reel generated: {highlight_reel_portrait_url}"
                    )
        except Exception as e:
            logger.warning(f"Highlight reel (portrait) generation skipped: {e}")

        # ── Heatmap Generation ────────────────────────────────────────────────
        emit_progress(88, "heatmap")
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

        # ── Advanced Tactical Analysis ────────────────────────────────────────
        emit_progress(90, "tactics")
        advanced_stats = {}
        if HEATMAP_AVAILABLE and track_frames:
            try:
                logger.info("Running Advanced Tactical Analysis...")
                p_metrics = calculate_player_metrics(track_frames, fps)
                possession = calculate_possession(track_frames)
                dominance = calculate_dominance(track_frames)
                radar_filename = f"radar_{match_id}.png"
                radar_path = str(UPLOADS_DIR / radar_filename)
                generate_tactical_radar(track_frames, radar_path, team_colors)

                advanced_stats = {
                    "playerMetrics": p_metrics,
                    "possession": possession,
                    "dominance": dominance,
                    "radarUrl": f"/uploads/{radar_filename}",
                }
                logger.info("Tactical analysis complete")
            except Exception as e:
                logger.warning(f"Tactical analysis failed: {e}")

        # ── PHASE 2: Context-aware analysis ──────────────────────────────────
        formation_data = {}
        trajectory_data = {}
        audio_volumes = {}

        if CONFIG["CONTEXT_AWARE_COMMENTARY"]:
            logger.info("Analyzing team formation for context-aware commentary...")
            formation_data = analyze_team_formation(track_frames, team_colors)
            logger.info(
                f"Formation: {formation_data.get('formation')} | Cohesion: {formation_data.get('cohesion')}"
            )

        if CONFIG["ENHANCED_BALL_TRACKING"]:
            all_balls = []
            for frame in track_frames:
                if frame.get("b"):
                    all_balls.extend(frame["b"])
            if all_balls:
                trajectory_data = predict_ball_trajectory(all_balls, fps=process_fps)
                logger.info(
                    f"Ball trajectory: {trajectory_data.get('direction')} @ {trajectory_data.get('speed')} speed"
                )

        # ── Emotion scores ────────────────────────────────────────────────────
        emotion_scores = [
            {
                "timestamp": w["timestamp"],
                "audioScore": w["audioScore"],
                "motionScore": w["motionScore"],
                "contextWeight": round(
                    time_context_weight(w["timestamp"], duration), 3
                ),
                "finalScore": round(
                    (
                        w["audioScore"] * 0.3
                        + w["motionScore"] * 0.5
                        + time_context_weight(w["timestamp"], duration) * 0.2
                    )
                    * 10,
                    2,
                ),
            }
            for w in motion_windows
        ]

        if CONFIG["DYNAMIC_AUDIO_MIXING"] and len(emotion_scores) > 0:
            logger.info("Calculating dynamic audio volumes...")
            avg_emotion = float(np.mean([e["finalScore"] for e in emotion_scores]))
            avg_motion = float(np.mean([e["motionScore"] for e in emotion_scores]))
            audio_volumes = calculate_dynamic_audio_volumes(avg_motion, avg_emotion)
            logger.info(f"Audio volumes: {audio_volumes}")

        # ── Gemini match summary ──────────────────────────────────────────────
        emit_progress(93, "summary")
        logger.info("Generating Gemini match summary…")
        try:
            summary = generate_match_summary(
                scored_events, highlights, duration, language=language
            )
        except Exception as e:
            logger.warning(f"Match summary generation failed: {e}")
            summary = None
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
        emit_progress(97, "thumbnail")
        thumbnail_url = None
        try:
            cap_thumb = cv2.VideoCapture(original_video_path)
            if not cap_thumb.isOpened():
                logger.error(
                    f"Failed to open video for thumbnail: {original_video_path}"
                )
            else:
                midpoint_frame_idx = total_frames // 2
                cap_thumb.set(cv2.CAP_PROP_POS_FRAMES, midpoint_frame_idx)
                ret_t, thumb_frame = cap_thumb.read()

                if not ret_t:
                    logger.warning(
                        "Midpoint frame read failed, falling back to first frame"
                    )
                    cap_thumb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_t, thumb_frame = cap_thumb.read()

                if ret_t:
                    th, tw = thumb_frame.shape[:2]
                    if tw > 1280:
                        t_scale = 1280 / tw
                        thumb_frame = cv2.resize(thumb_frame, (1280, int(th * t_scale)))

                    thumbnail_filename = f"thumbnail_{match_id}.jpg"
                    thumbnail_path = str(UPLOADS_DIR / thumbnail_filename)
                    success = cv2.imwrite(
                        thumbnail_path, thumb_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                    )
                    if success:
                        thumbnail_url = f"/uploads/{thumbnail_filename}"
                cap_thumb.release()
        except Exception as te:
            logger.warning(f"Thumbnail generation system error: {te}")

        payload = {
            "events": convert_numpy(scored_events),
            "highlights": convert_numpy(highlights),
            "emotionScores": convert_numpy(emotion_scores),
            "duration": round(float(duration), 1),
            "summary": summary,
            "highlightReelUrl": highlight_reel_url,
            "highlightReelPortraitUrl": highlight_reel_portrait_url,
            "thumbnailUrl": thumbnail_url,
            "trackingData": convert_numpy(track_frames),
            "teamColors": convert_numpy(team_colors),
            "heatmapUrl": heatmap_url,
            "topSpeedKmh": round(float(top_speed_kmh), 1),
            "videoUrl": f"/uploads/{Path(original_video_path).name}",
            "goalpostDetections": convert_numpy(goalpost_detections),
            "advancedStats": convert_numpy(advanced_stats),
            "formationData": formation_data,
            "trajectoryData": trajectory_data,
            "audioVolumes": audio_volumes,
        }

        emit_progress(99, "saving")
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
                json={"progress": 100, "stage": "done"},
                timeout=5,
            )
        except Exception:
            pass

        # Cleanup compressed file
        if compressed and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

        return {"status": "completed", "match_id": match_id}

    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        if compressed and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
        _report_failure(match_id)
        return {"error": str(e)}
