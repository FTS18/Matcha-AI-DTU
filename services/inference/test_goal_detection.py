#!/usr/bin/env python3
"""
Goal Detection Module Test Script
Quick test to verify goal detection is working correctly.
"""

import sys
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules import correctly."""
    logger.info("Testing imports...")

    try:
        from app.core.goal_detection import (
            GoalDetectionEngine,
            GoalGeometry,
            KalmanBallTracker,
            BallObs,
            GoalEvent,
        )

        logger.info(" Goal detection module imported successfully")
        return True
    except Exception as e:
        logger.error(f" Import failed: {e}")
        return False


def test_goal_detection_engine():
    """Test GoalDetectionEngine initialization."""
    logger.info("\nTesting GoalDetectionEngine...")

    try:
        from app.core.goal_detection import GoalDetectionEngine

        engine = GoalDetectionEngine()
        engine.init(frame_w=1280, frame_h=720, fps=30.0)
        logger.info(f" Engine created: {engine.__class__.__name__}")

        # Test auto-calibration
        if hasattr(engine, "_geometry") and engine._geometry:
            engine._geometry.auto_calibrate()
            logger.info(f" Auto-calibration successful")
        else:
            logger.warning(" Engine geometry not available for auto-calibration")

        return True
    except Exception as e:
        logger.error(f" Engine test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ball_tracker():
    """Test KalmanBallTracker component."""
    logger.info("\nTesting KalmanBallTracker...")

    try:
        from app.core.goal_detection import KalmanBallTracker, BallObs

        tracker = KalmanBallTracker()
        logger.info(f" Tracker created")

        # Test with dummy detections
        detection = BallObs(
            cx=640, cy=360, w=10, h=10, conf=0.95, frame_id=1
        )

        cx, cy = tracker.update(detection)
        logger.info(f" Tracker update successful, tracked center: ({cx}, {cy})")

        return True
    except Exception as e:
        logger.error(f" Tracker test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_analysis_integration():
    """Test goal detection integration in analysis module."""
    logger.info("\nTesting analysis.py integration...")

    try:
        # Import analysis which uses goal_detection
        from app.core import analysis
        
        # Check if GOAL_DETECTION_AVAILABLE is present
        available = getattr(analysis, "GOAL_DETECTION_AVAILABLE", False)
        logger.info(f" GOAL_DETECTION_AVAILABLE: {available}")

        return True
    except Exception as e:
        logger.error(f" Analysis integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("GOAL DETECTION MODULE TEST SUITE")
    logger.info("=" * 60)

    results = {
        "Imports": test_imports(),
        "Engine": test_goal_detection_engine(),
        "Tracker": test_ball_tracker(),
        "Analysis Integration": test_analysis_integration(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    for name, passed in results.items():
        status = " PASS" if passed else " FAIL"
        logger.info(f"{status:8} {name}")

    all_passed = all(results.values())
    logger.info("=" * 60)
    logger.info(
        f"Overall: {' ALL TESTS PASSED' if all_passed else ' SOME TESTS FAILED'}"
    )
    logger.info("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
