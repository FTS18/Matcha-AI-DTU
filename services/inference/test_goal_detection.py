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
            GoalLineCalibrator,
            BallTracker,
            BallDetection,
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
        engine.calibrator.auto_calibrate()
        logger.info(f" Auto-calibration successful")
        logger.info(f" Goal line range: {engine.calibrator.get_goal_line_x_range()}")

        return True
    except Exception as e:
        logger.error(f" Engine test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_ball_tracker():
    """Test BallTracker component."""
    logger.info("\nTesting BallTracker...")

    try:
        from app.core.goal_detection import BallTracker, BallDetection

        tracker = BallTracker(max_age=30, min_hits=3)
        logger.info(f" Tracker created")

        # Test with dummy detections
        detection = BallDetection(
            x=640, y=360, confidence=0.95, bbox=(630, 350, 650, 370), frame_id=1
        )

        tracked = tracker.update([detection])
        logger.info(f" Tracker update successful, tracked: {len(tracked)} objects")

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
        from app.core.analysis import GOAL_DETECTION_AVAILABLE, GoalDetectionEngine

        logger.info(f" Goal detection imports successful")
        logger.info(f" GOAL_DETECTION_AVAILABLE: {GOAL_DETECTION_AVAILABLE}")

        if GOAL_DETECTION_AVAILABLE:
            logger.info(f" GoalDetectionEngine: {GoalDetectionEngine.__name__}")

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
