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

from app.core.audio_engine import calculate_dynamic_audio_volumes
from app.core.highlight_manager import (
    select_highlights,
    select_highlights_with_narrative,
    group_related_events,
)
from app.core.events import detect_all_events, detect_goals_in_video
from app.core.tracking import TrackingManager
from app.core.visuals import VisualsManager
from app.core.analysis.narrative import get_fallback_commentary, compile_final_payload
from app.core.frame_cache import FrameCache
from app.core.vision_engine import VisionEngine
from app.core.post_processing.manager import PostProcessor
from app.core.analysis.comms import (
    emit_progress,
    emit_live_event,
    emit_tracking_frames,
    report_failure,
)
from app.core.downloader import _download_youtube_video as download_youtube_video
from app.core.video import precompress_video as _precompress_video
from app.core.llm import analyze_frame_with_vision
from app.core.vision_validator import find_motion_peaks

# Optional engines
try:
    from app.core.goal_detection import GoalDetectionEngine

    GOAL_DETECTION_AVAILABLE = True
except ImportError:
    GoalDetectionEngine = None
    GOAL_DETECTION_AVAILABLE = False

try:
    from app.core.goalpost_detection import GoalpostDetector, GoalpostTracker

    GOALPOST_DETECTION_AVAILABLE = True
except ImportError:
    GoalpostDetector = GoalpostTracker = None
    GOALPOST_DETECTION_AVAILABLE = False

try:
    from app.core.scoreboard_detector import ScoreboardDetector

    SCOREBOARD_DETECTION_AVAILABLE = True
except ImportError:
    ScoreboardDetector = None
    SCOREBOARD_DETECTION_AVAILABLE = False

try:
    from app.core.transformer_detector import TransformerActionSpotter
    from app.core.dynamic_calibration import DynamicPitchCalibrator

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TransformerActionSpotter = DynamicPitchCalibrator = None
    TRANSFORMER_AVAILABLE = False

try:
    from app.core.soccernet_detector import detect_football_events

    SOCCERNET_AVAILABLE = True
except ImportError:
    detect_football_events = None
    SOCCERNET_AVAILABLE = False

CV_PHYSICS_AVAILABLE = False
detect_cv_physics = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure ~/bin is on PATH
_home_bin = os.path.join(os.path.expanduser("~"), "bin")
if _home_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _home_bin + os.pathsep + os.environ.get("PATH", "")

from app.core.analysis.config import CONFIG, UPLOADS_DIR, MUSIC_DIR, BASE_DIR

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


def _frame_to_pil(frame):
    from app.core.llm import _frame_to_pil as f2p

    return f2p(frame)


ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:4000/api/v1")


def _get_motion_at(windows, timestamp):
    if not windows:
        return 0.3
    return min(windows, key=lambda w: abs(w["timestamp"] - timestamp))["motionScore"]


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
            emit_progress(match_id, 5, "downloading")

            def yt_progress(percent):
                prog = min(20, int(percent * 0.2))
                emit_progress(match_id, prog, "downloading")

            video_path = download_youtube_video(
                url=video_path,
                output_dir=str(UPLOADS_DIR),
                filename_prefix=match_id,
                start_time=start_time,
                end_time=end_time,
                progress_callback=yt_progress,
            )
        except Exception as e:
            logger.error(str(e))
            report_failure(match_id)
            return {"error": "Failed to download YouTube video"}

    # 2. Proceed with local file validation
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        report_failure(match_id)
        return {"error": "Video file not found"}

    # Pre-compress large videos for faster processing
    original_video_path = video_path
    emit_progress(match_id, 22, "compressing")
    video_path = _precompress_video(video_path, match_id, CONFIG)
    compressed = video_path != original_video_path

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open: {video_path}")
            report_failure(match_id)
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

        emit_progress(match_id, 25, "scanning")

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

        # Initialize Vision Engine
        vision_engine = VisionEngine()
        vision_engine.init_models()
        model, ball_model = vision_engine.get_models()

        _tracking_manager = TrackingManager(model, ball_model, CONFIG)

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

            # ── Vision Transformer (ViT) Action Spotting ───────────────────
            from app.core.events.transformer import spot_transformer_actions

            if _transformer_spotter is not None:
                vit_input_frame = cv2.resize(frame, (224, 224))
                vit_frame_buffer.append(vit_input_frame)

                vit_evt = spot_transformer_actions(
                    vit_frame_buffer, timestamp, fps, _transformer_spotter
                )
                if vit_evt:
                    emit_live_event(match_id, vit_evt)
                    logger.info(f" ViT DETECTED: {vit_evt['type']} at {timestamp:.2f}s")

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
                if m_score >= CONFIG["MOTION_PEAK_THRESHOLD"]:
                    if (timestamp - _last_vision_call_ts) >= _VISION_COOLDOWN_SECS:
                        from app.core.events.live_analyzer import analyze_live_window

                        live_evt = analyze_live_window(
                            frame,
                            timestamp,
                            m_score,
                            language,
                            analyze_frame_with_vision,
                        )
                        if live_evt:
                            _last_vision_call_ts = timestamp
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
                frame_balls, frame_persons, frame_jersey_colours = (
                    _tracking_manager.process_frame(
                        frame, frame_w, frame_h, cache=_frame_cache
                    )
                )
                jersey_colours.extend(frame_jersey_colours)
            else:
                # Still need to handle frame_balls/frame_persons if skipping YOLO
                # but they will be empty as initialized above
                pass

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

        # ── Tracking & Refinement ─────────────────────────────────────────────
        tracker = TrackingManager(model, ball_model, CONFIG)
        # Note: jersey_colours was collected during frame iteration in previous steps (lines 800-990)
        track_frames, team_colors = tracker.refine_tracks(
            track_frames, jersey_colours, [[220, 60, 60], [60, 100, 220]]
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

        # ── Event Detection (Multi-Engine) ──────────────────────────────────
        raw_events = detect_all_events(
            video_path=original_video_path,
            track_frames=track_frames,
            motion_windows=motion_windows,
            fps=fps,
            process_fps=process_fps,
            config=CONFIG,
            goal_engine=_goal_engine,
            scoreboard_detector=_scoreboard_detector,
            soccernet_detector=(
                detect_football_events
                if (SOCCERNET_AVAILABLE and detect_football_events)
                else None
            ),
            cv_physics_detector=(
                detect_cv_physics
                if (CV_PHYSICS_AVAILABLE and detect_cv_physics)
                else None
            ),
            vision_validator_peaks=find_motion_peaks,
            get_motion_at_fn=_get_motion_at,
            progress_callback=emit_progress,
        )

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
        from app.core.scoring.engine import score_raw_events

        emit_progress(match_id, 72, "scoring")
        scored_events = score_raw_events(
            raw_events, motion_windows, duration, _get_motion_at
        )
        for ev in scored_events:
            emit_live_event(match_id, ev)

        # ── Gemini commentary per event (parallel) ───────────────────────────
        from app.core.llm import generate_commentary_parallel

        emit_progress(match_id, 75, "commentary")
        try:
            scored_events = generate_commentary_parallel(
                scored_events, duration, language=language, max_workers=3
            )
        except Exception as e:
            logger.warning(
                f"Commentary generation timed out or failed: {e} — continuing without commentary"
            )

        # ── Select Highlights for Reel ──────────────────────────────────────
        emit_progress(match_id, 80, "highlights")
        highlights = select_highlights(scored_events)

        # ── Post-Processing & Completion ─────────────────────────────────────
        post_processor = PostProcessor(match_id, CONFIG, UPLOADS_DIR, MUSIC_DIR)
        payload = post_processor.run_post_processing(
            scored_events=scored_events,
            highlights=highlights,
            track_frames=track_frames,
            team_colors=team_colors,
            duration=duration,
            original_video_path=original_video_path,
            total_frames=total_frames,
            fps=fps,
            language=language,
        )

        emit_progress(match_id, 99, "saving")
        try:
            resp = requests.post(
                f"{ORCHESTRATOR_URL}/matches/{match_id}/complete",
                json=payload,
                timeout=30,
            )
            logger.info(f"Complete sent — HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Failed to send completion: {e}")

        emit_progress(match_id, 100, "done")

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
        report_failure(match_id)
        return {"error": str(e)}
