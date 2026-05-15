import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def detect_all_events(
    video_path: str,
    track_frames: List[Dict[str, Any]],
    motion_windows: List[Dict[str, Any]],
    fps: float,
    process_fps: float,
    config: Dict[str, Any],
    goal_engine: Any = None,
    scoreboard_detector: Any = None,
    soccernet_detector: Any = None,
    cv_physics_detector: Any = None,
    vision_validator_peaks: Any = None,
    get_motion_at_fn: Any = None,
    progress_callback: Any = None,
) -> List[Dict[str, Any]]:
    """
    Orchestrate multi-stage event detection:
    1. GoalDetectionEngine (CV-based)
    2. SoccerNet (Deep Learning)
    3. CV Physics (Rule-based)
    4. Motion Peaks (Fallback)
    5. Scoreboard Verification
    """
    raw_events = []

    if progress_callback:
        progress_callback(62, "events")

    # 1. GoalDetectionEngine
    if goal_engine is not None:
        try:
            from app.core.goal_detection import goal_events_to_raw

            _raw_goals = goal_events_to_raw(goal_engine.goals)
            if _raw_goals:
                logger.info(f"GoalDetectionEngine found {len(_raw_goals)} goal(s)")
                raw_events.extend(_raw_goals)
        except Exception as e:
            logger.error(f"GoalDetectionEngine processing failed: {e}")

    # 2. SoccerNet
    if soccernet_detector:
        try:
            logger.info("Running SoccerNet analysis...")
            sn_events = soccernet_detector(video_path, sensitivity=1.0)
            if sn_events:
                for ev in sn_events:
                    raw_events.append(
                        {
                            "timestamp": ev["timestamp"],
                            "type": ev["type"],
                            "confidence": ev["confidence"],
                            "description": f"SoccerNet detected {ev['type'].lower()}",
                            "source": "soccernet",
                        }
                    )
                logger.info(f"SoccerNet detected {len(sn_events)} events")
        except Exception as e:
            logger.error(f"SoccerNet analysis failed: {e}")

    # 3. CV Physics
    if progress_callback:
        progress_callback(67, "cv_physics")

    if cv_physics_detector:
        try:
            logger.info("Running CV Physics analysis...")
            cv_events = cv_physics_detector(track_frames, fps=process_fps)
            if cv_events:
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

    # 4. Fallback: Motion-based highlights
    if len(raw_events) < 3 and vision_validator_peaks:
        logger.info("Supplementing with motion-based highlight detection...")
        candidate_timestamps = vision_validator_peaks(
            motion_windows,
            threshold=config["MOTION_FALLBACK_THRESHOLD"],
            min_gap=config["MOTION_FALLBACK_MIN_GAP"],
        )

        existing_times = {ev["timestamp"] for ev in raw_events}
        for candidate_t in candidate_timestamps:
            if any(abs(candidate_t - et) < 15 for et in existing_times):
                continue

            motion_score = (
                get_motion_at_fn(motion_windows, candidate_t)
                if get_motion_at_fn
                else 0.5
            )
            if motion_score >= config["MOTION_FALLBACK_THRESHOLD"]:
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

            if len(raw_events) >= config["MAX_MOTION_BASED_EVENTS"]:
                break

    raw_events.sort(key=lambda x: x["timestamp"])

    # 5. Scoreboard Verification
    if scoreboard_detector and getattr(scoreboard_detector, "has_scoreboard", False):
        try:
            logger.info("Verifying goals with scoreboard data...")
            raw_events = scoreboard_detector.verify_goal_events(
                raw_events, tolerance_sec=15.0
            )
            raw_events.sort(key=lambda x: x["timestamp"])
        except Exception as e:
            logger.error(f"Scoreboard verification failed: {e}")

    return raw_events
