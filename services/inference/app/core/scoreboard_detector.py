"""
Scoreboard Detection & Goal Verification
=========================================
Detects the on-screen scoreboard in broadcast football footage and extracts
the current score using pure OpenCV (no OCR dependencies).

Architecture:
  1. ROI Estimation  -- Scoreboards are typically in top-left/top-right 30% of frame
  2. Text Region Detection -- MSER + edge-based detection for high-contrast text
  3. Digit Recognition -- Template-based digit matching using 7-segment patterns
  4. Score Tracking -- Track score changes over time, verify goal events

Works without tesseract/easyocr — uses structural digit recognition.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoreReading:
    """A single scoreboard reading at a given timestamp."""
    frame_id: int
    timestamp: float
    score_left: Optional[int]
    score_right: Optional[int]
    confidence: float
    roi_region: str  # "top_left", "top_right", "top_center"


@dataclass
class ScoreChange:
    """Detected score change (potential goal)."""
    timestamp: float
    frame_id: int
    old_left: int
    old_right: int
    new_left: int
    new_right: int
    side: str  # "left" or "right"
    confidence: float


class ScoreboardDetector:
    """
    Detects scoreboards in broadcast football and tracks score changes.
    
    Usage:
        detector = ScoreboardDetector()
        for frame_id, frame, timestamp in frames:
            detector.process_frame(frame, frame_id, timestamp)
        changes = detector.get_score_changes()
    """

    # Typical scoreboard regions (normalised coords)
    _ROI_CANDIDATES = [
        ("top_left",   0.0, 0.0, 0.35, 0.12),
        ("top_center", 0.25, 0.0, 0.50, 0.12),
        ("top_right",  0.65, 0.0, 0.35, 0.12),
    ]

    def __init__(
        self,
        sample_interval: int = 30,   # Process every N-th frame
        min_confidence: float = 0.5,
        score_change_cooldown: float = 10.0,  # Min seconds between score changes
    ):
        self._sample_interval = sample_interval
        self._min_confidence = min_confidence
        self._cooldown = score_change_cooldown
        
        self._readings: List[ScoreReading] = []
        self._score_changes: List[ScoreChange] = []
        self._last_stable_score: Optional[Tuple[int, int]] = None
        self._last_change_time: float = -999.0
        self._scoreboard_roi: Optional[str] = None  # Lock to detected region
        self._frame_count = 0
        
        # Stable score requires N consistent readings
        self._recent_scores: deque = deque(maxlen=5)

    def process_frame(
        self, frame: np.ndarray, frame_id: int, timestamp: float
    ) -> Optional[ScoreReading]:
        """Process a frame and return a score reading if found."""
        self._frame_count += 1
        if self._frame_count % self._sample_interval != 0:
            return None

        h, w = frame.shape[:2]
        best_reading: Optional[ScoreReading] = None
        best_conf = 0.0

        # Try candidate ROIs (or locked region)
        rois = self._ROI_CANDIDATES
        if self._scoreboard_roi:
            rois = [r for r in rois if r[0] == self._scoreboard_roi]

        for name, rx, ry, rw, rh in rois:
            x1 = int(rx * w)
            y1 = int(ry * h)
            x2 = int((rx + rw) * w)
            y2 = int((ry + rh) * h)
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            score_left, score_right, conf = self._detect_score_in_roi(roi)
            if conf > best_conf and conf >= self._min_confidence:
                best_conf = conf
                best_reading = ScoreReading(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    score_left=score_left,
                    score_right=score_right,
                    confidence=conf,
                    roi_region=name,
                )

        if best_reading:
            self._readings.append(best_reading)
            # Lock to the region once we find one consistently
            if len(self._readings) >= 3 and not self._scoreboard_roi:
                region_counts = Counter(r.roi_region for r in self._readings[-5:])
                top_region, count = region_counts.most_common(1)[0]
                if count >= 3:
                    self._scoreboard_roi = top_region
                    logger.info(f"Scoreboard locked to region: {top_region}")

            self._track_score_change(best_reading)

        return best_reading

    def _detect_score_in_roi(
        self, roi: np.ndarray
    ) -> Tuple[Optional[int], Optional[int], float]:
        """
        Detect score digits in a scoreboard ROI using structural analysis.
        Returns (score_left, score_right, confidence).
        """
        if roi.shape[0] < 10 or roi.shape[1] < 20:
            return None, None, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        h, w = gray.shape

        # ── Step 1: Find high-contrast text regions ──────────────────────
        # Scoreboards typically have bright text on dark bg or vice versa
        _, binary_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        _, binary_dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        
        # Use whichever has more structure
        bright_px = np.count_nonzero(binary_bright)
        dark_px = np.count_nonzero(binary_dark)
        
        # Scoreboard text is usually a small fraction of the ROI
        bright_ratio = bright_px / (h * w) if h * w > 0 else 0
        dark_ratio = dark_px / (h * w) if h * w > 0 else 0
        
        # Pick the one with reasonable text density (5-40%)
        if 0.02 < bright_ratio < 0.45:
            binary = binary_bright
        elif 0.02 < dark_ratio < 0.45:
            binary = binary_dark
        else:
            return None, None, 0.0

        # ── Step 2: Find digit-like contours ─────────────────────────────
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        digit_candidates = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bh / bw if bw > 0 else 0
            area = bw * bh
            # Digits are taller than wide (aspect 1.2-5.0), reasonable size
            if (1.0 < aspect < 6.0 
                and bh > h * 0.25  # At least 25% of ROI height
                and bh < h * 0.95  # Not the whole ROI
                and bw > 3         # At least a few pixels wide
                and area > 30):    # Minimum area
                digit_candidates.append((bx, by, bw, bh))

        if len(digit_candidates) < 2:
            return None, None, 0.0

        # Sort by x position
        digit_candidates.sort(key=lambda d: d[0])

        # ── Step 3: Group into left score and right score ────────────────
        # Find the largest gap between digit clusters (that's the separator)
        if len(digit_candidates) >= 2:
            gaps = []
            for i in range(len(digit_candidates) - 1):
                gap_start = digit_candidates[i][0] + digit_candidates[i][2]
                gap_end = digit_candidates[i + 1][0]
                gaps.append((gap_end - gap_start, i))
            
            if gaps:
                gaps.sort(reverse=True)
                split_idx = gaps[0][1]  # Split at biggest gap
                left_digits = digit_candidates[:split_idx + 1]
                right_digits = digit_candidates[split_idx + 1:]
            else:
                return None, None, 0.0
        else:
            return None, None, 0.0

        # ── Step 4: Recognise digits using structural features ───────────
        score_left = self._recognize_digit_group(binary, left_digits)
        score_right = self._recognize_digit_group(binary, right_digits)

        if score_left is None or score_right is None:
            return None, None, 0.0

        # Sanity check: football scores are typically 0-15
        if score_left > 20 or score_right > 20:
            return None, None, 0.0

        # Confidence based on digit clarity
        conf = min(0.9, 0.4 + 0.1 * len(digit_candidates))
        return score_left, score_right, conf

    def _recognize_digit_group(
        self, binary: np.ndarray, digit_bboxes: List[Tuple[int, int, int, int]]
    ) -> Optional[int]:
        """Recognize a group of digit bboxes as a number using segment analysis."""
        if not digit_bboxes:
            return None

        digits = []
        for bx, by, bw, bh in digit_bboxes:
            digit_img = binary[by:by+bh, bx:bx+bw]
            if digit_img.size == 0:
                continue
            d = self._classify_digit(digit_img)
            if d is not None:
                digits.append(d)

        if not digits:
            return None

        # Combine digits: [1, 2] -> 12
        result = 0
        for d in digits:
            result = result * 10 + d
        return result

    def _classify_digit(self, digit_img: np.ndarray) -> Optional[int]:
        """
        Classify a single digit image using 7-segment analysis.
        Divides the digit into regions and checks pixel density.
        """
        h, w = digit_img.shape[:2]
        if h < 5 or w < 3:
            return None

        # Resize to standard size for consistent analysis
        std = cv2.resize(digit_img, (14, 20), interpolation=cv2.INTER_NEAREST)

        # Define 7 segments (like a digital clock display):
        # Top, Top-Left, Top-Right, Middle, Bottom-Left, Bottom-Right, Bottom
        segments = {
            'top':    std[0:4, 2:12],
            'top_l':  std[2:10, 0:5],
            'top_r':  std[2:10, 9:14],
            'mid':    std[8:12, 2:12],
            'bot_l':  std[10:18, 0:5],
            'bot_r':  std[10:18, 9:14],
            'bot':    std[16:20, 2:12],
        }

        # Calculate fill ratio for each segment
        fills = {}
        for name, seg in segments.items():
            if seg.size > 0:
                fills[name] = np.count_nonzero(seg) / seg.size
            else:
                fills[name] = 0.0

        t = 0.35  # Threshold for "segment is on"

        on = {k: v > t for k, v in fills.items()}

        # Match against known 7-segment patterns
        patterns = {
            0: {'top': True,  'top_l': True,  'top_r': True,  'mid': False, 'bot_l': True,  'bot_r': True,  'bot': True},
            1: {'top': False, 'top_l': False, 'top_r': True,  'mid': False, 'bot_l': False, 'bot_r': True,  'bot': False},
            2: {'top': True,  'top_l': False, 'top_r': True,  'mid': True,  'bot_l': True,  'bot_r': False, 'bot': True},
            3: {'top': True,  'top_l': False, 'top_r': True,  'mid': True,  'bot_l': False, 'bot_r': True,  'bot': True},
            4: {'top': False, 'top_l': True,  'top_r': True,  'mid': True,  'bot_l': False, 'bot_r': True,  'bot': False},
            5: {'top': True,  'top_l': True,  'top_r': False, 'mid': True,  'bot_l': False, 'bot_r': True,  'bot': True},
            6: {'top': True,  'top_l': True,  'top_r': False, 'mid': True,  'bot_l': True,  'bot_r': True,  'bot': True},
            7: {'top': True,  'top_l': False, 'top_r': True,  'mid': False, 'bot_l': False, 'bot_r': True,  'bot': False},
            8: {'top': True,  'top_l': True,  'top_r': True,  'mid': True,  'bot_l': True,  'bot_r': True,  'bot': True},
            9: {'top': True,  'top_l': True,  'top_r': True,  'mid': True,  'bot_l': False, 'bot_r': True,  'bot': True},
        }

        best_digit = None
        best_match = -1
        for digit, pattern in patterns.items():
            matches = sum(1 for k in pattern if on.get(k) == pattern[k])
            if matches > best_match:
                best_match = matches
                best_digit = digit

        # Require at least 5/7 segments to match
        if best_match >= 5:
            return best_digit
        return None

    def _track_score_change(self, reading: ScoreReading) -> None:
        """Track score changes and detect goals."""
        if reading.score_left is None or reading.score_right is None:
            return

        current = (reading.score_left, reading.score_right)
        self._recent_scores.append(current)

        # Need consistent readings before establishing stable score
        if len(self._recent_scores) < 3:
            return

        # Check if recent readings agree (at least 3/5)
        score_counts = Counter(self._recent_scores)
        most_common, count = score_counts.most_common(1)[0]

        if count < 3:
            return  # Not stable enough

        # First stable score detected
        if self._last_stable_score is None:
            self._last_stable_score = most_common
            logger.info(f"📊 Scoreboard detected: {most_common[0]} - {most_common[1]}")
            return

        # Check for score change
        old = self._last_stable_score
        new = most_common

        if old == new:
            return  # No change

        # Verify it's a valid score increment (exactly +1 on one side)
        left_diff = new[0] - old[0]
        right_diff = new[1] - old[1]

        # Cooldown check
        if reading.timestamp - self._last_change_time < self._cooldown:
            return

        if left_diff == 1 and right_diff == 0:
            side = "left"
        elif right_diff == 1 and left_diff == 0:
            side = "right"
        elif left_diff == 0 and right_diff == 0:
            return
        else:
            # Score changed by more than 1 or both sides — might be a camera
            # cut to a different graphic. Log but don't verify.
            logger.warning(
                f"📊 Unusual score change: {old[0]}-{old[1]} → {new[0]}-{new[1]}, skipping"
            )
            self._last_stable_score = new
            return

        change = ScoreChange(
            timestamp=reading.timestamp,
            frame_id=reading.frame_id,
            old_left=old[0],
            old_right=old[1],
            new_left=new[0],
            new_right=new[1],
            side=side,
            confidence=reading.confidence,
        )
        self._score_changes.append(change)
        self._last_stable_score = new
        self._last_change_time = reading.timestamp
        logger.info(
            f"📊 SCORE CHANGE at {reading.timestamp:.1f}s: "
            f"{old[0]}-{old[1]} → {new[0]}-{new[1]} ({side} team scored)"
        )

    def get_score_changes(self) -> List[ScoreChange]:
        """Return all detected score changes."""
        return list(self._score_changes)

    def get_current_score(self) -> Optional[Tuple[int, int]]:
        """Return the last stable score or None."""
        return self._last_stable_score

    def verify_goal_events(
        self, goal_events: List[dict], tolerance_sec: float = 15.0
    ) -> List[dict]:
        """
        Verify goal events against scoreboard changes.
        
        For each goal event, check if the scoreboard shows a +1 increment
        within ±tolerance_sec. Updates the event with verification info.
        
        Returns: Updated event list with 'scoreboard_verified' field.
        """
        changes = self.get_score_changes()
        
        for ev in goal_events:
            if ev.get("type", "").upper() != "GOAL":
                ev["scoreboard_verified"] = None  # N/A for non-goals
                continue

            ev_time = ev.get("timestamp", 0)
            verified = False
            matched_change = None

            for sc in changes:
                if abs(sc.timestamp - ev_time) <= tolerance_sec:
                    verified = True
                    matched_change = sc
                    break

            ev["scoreboard_verified"] = verified
            if verified and matched_change:
                ev["scoreboard_score"] = f"{matched_change.new_left}-{matched_change.new_right}"
                ev["confidence"] = min(1.0, ev.get("confidence", 0.5) + 0.2)  # Boost confidence
                logger.info(
                    f"✅ Goal at {ev_time:.1f}s VERIFIED by scoreboard: "
                    f"{matched_change.old_left}-{matched_change.old_right} → "
                    f"{matched_change.new_left}-{matched_change.new_right}"
                )
            elif ev.get("type", "").upper() == "GOAL":
                logger.info(
                    f"⚠️ Goal at {ev_time:.1f}s NOT verified by scoreboard "
                    f"(no matching score change within ±{tolerance_sec}s)"
                )

        # Also check for score changes that don't match any detected goal
        for sc in changes:
            has_match = any(
                abs(sc.timestamp - ev.get("timestamp", 0)) <= tolerance_sec
                for ev in goal_events
                if ev.get("type", "").upper() == "GOAL"
            )
            if not has_match:
                logger.info(
                    f"📊 Scoreboard goal at {sc.timestamp:.1f}s ({sc.side}) "
                    f"has no matching detection — adding as new GOAL event"
                )
                goal_events.append({
                    "timestamp": sc.timestamp,
                    "type": "GOAL",
                    "confidence": sc.confidence * 0.8,  # Slightly lower conf for scoreboard-only
                    "description": f"Scoreboard-detected goal ({sc.side} team, "
                                   f"{sc.new_left}-{sc.new_right})",
                    "source": "scoreboard",
                    "scoreboard_verified": True,
                    "scoreboard_score": f"{sc.new_left}-{sc.new_right}",
                })

        return goal_events

    @property
    def readings_count(self) -> int:
        return len(self._readings)

    @property
    def has_scoreboard(self) -> bool:
        """Whether a scoreboard was detected."""
        return self._scoreboard_roi is not None
