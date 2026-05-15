"""
Goalpost Detection Module
=========================
Detects goalposts in football/soccer video for better spatial awareness.
Supports both 2D (bounding box) and 3D (wireframe) detection approaches.
"""

import logging
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Goalpost:
 """Represents a detected goalpost."""
 x: float
 y: float
 width: float
 height: float
 confidence: float
 is_left: bool # True for left post, False for right post
 frame_id: int
 timestamp: float


@dataclass
class GoalpostPair:
 """Represents both posts of a goal."""
 left: Optional[Goalpost]
 right: Optional[Goalpost]
 center_x: float
 center_y: float
 goal_width: float
 confidence: float
 frame_id: int
 timestamp: float


class GoalpostDetector:
 """
 Detects goalposts using color and edge detection.
 Goalposts are typically white or bright colored vertical rectangles.
 """

 def __init__(self, 
 white_lower_hsv: Tuple = (0, 0, 200),
 white_upper_hsv: Tuple = (180, 30, 255),
 min_post_height: int = 50,
 max_aspect_ratio: float = 3.0):
 """
 Initialize goalpost detector.
 
 Args:
 white_lower_hsv: Lower HSV threshold for white color
 white_upper_hsv: Upper HSV threshold for white color
 min_post_height: Minimum height in pixels for a post
 max_aspect_ratio: Maximum height/width ratio for a post
 """
 self.white_lower_hsv = white_lower_hsv
 self.white_upper_hsv = white_upper_hsv
 self.min_post_height = min_post_height
 self.max_aspect_ratio = max_aspect_ratio
 self._frame_history: List[np.ndarray] = []
 self._max_history = 5

 def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> Optional[GoalpostPair]:
 """
 Detect goalposts in frame.
 
 Args:
 frame: Input video frame (BGR)
 frame_id: Frame identifier
 timestamp: Timestamp in seconds
 
 Returns:
 GoalpostPair with detected posts, or None if not detected
 """
 try:
 # Convert to HSV for better color detection
 hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 
 # Create mask for white pixels (goalposts are white)
 mask = cv2.inRange(hsv, self.white_lower_hsv, self.white_upper_hsv)
 
 # Apply morphological operations to clean up mask
 kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
 mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
 mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
 
 # Find contours
 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
 if not contours:
 return None
 
 # Filter and score contours as potential goalposts
 posts = []
 for contour in contours:
 x, y, w, h = cv2.boundingRect(contour)
 
 # Filter by size
 if h < self.min_post_height:
 continue
 
 # Filter by aspect ratio (should be tall and narrow)
 aspect_ratio = h / max(w, 1)
 if aspect_ratio > self.max_aspect_ratio:
 continue
 
 # Calculate confidence based on fill ratio and vertical alignment
 area = cv2.contourArea(contour)
 bbox_area = w * h
 fill_ratio = area / max(bbox_area, 1)
 
 # Posts should have high fill ratio (solid color)
 if fill_ratio < 0.6:
 continue
 
 confidence = fill_ratio * (min(aspect_ratio / self.max_aspect_ratio, 1.0))
 
 posts.append({
 'x': x + w / 2,
 'y': y + h / 2,
 'w': w,
 'h': h,
 'confidence': confidence,
 'area': area
 })
 
 if len(posts) < 2:
 return None
 
 # Sort by x coordinate and pair them
 posts.sort(key=lambda p: p['x'])
 
 # Find best pair (should be roughly the same height, separated horizontally)
 best_pair = None
 best_score = 0
 
 for i in range(len(posts) - 1):
 left = posts[i]
 right = posts[i + 1]
 
 # Verify they're separated horizontally
 if right['x'] - left['x'] < max(left['w'], right['w']):
 continue
 
 # Height similarity check
 height_diff = abs(left['h'] - right['h'])
 height_avg = (left['h'] + right['h']) / 2
 height_penalty = min(height_diff / height_avg, 1.0)
 
 # Y-position similarity (should be at similar heights)
 y_diff = abs(left['y'] - right['y'])
 y_penalty = min(y_diff / height_avg, 0.5)
 
 # Separation score (should be well-separated)
 separation = right['x'] - left['x']
 separation_score = min(separation / (left['w'] + right['w']), 2.0)
 
 pair_score = (left['confidence'] + right['confidence']) / 2 * (1.0 - height_penalty - y_penalty) * separation_score
 
 if pair_score > best_score:
 best_score = pair_score
 best_pair = (i, i + 1, pair_score)
 
 if best_pair is None or best_score < 0.3:
 return None
 
 left_idx, right_idx, _ = best_pair
 left_post = posts[left_idx]
 right_post = posts[right_idx]
 
 # Create Goalpost objects
 left = Goalpost(
 x=left_post['x'],
 y=left_post['y'],
 width=left_post['w'],
 height=left_post['h'],
 confidence=left_post['confidence'],
 is_left=True,
 frame_id=frame_id,
 timestamp=timestamp
 )
 
 right = Goalpost(
 x=right_post['x'],
 y=right_post['y'],
 width=right_post['w'],
 height=right_post['h'],
 confidence=right_post['confidence'],
 is_left=False,
 frame_id=frame_id,
 timestamp=timestamp
 )
 
 # Calculate goal center and width
 center_x = (left.x + right.x) / 2
 center_y = (left.y + right.y) / 2
 goal_width = abs(right.x - left.x)
 pair_confidence = (left.confidence + right.confidence) / 2 * best_score
 
 return GoalpostPair(
 left=left,
 right=right,
 center_x=center_x,
 center_y=center_y,
 goal_width=goal_width,
 confidence=pair_confidence,
 frame_id=frame_id,
 timestamp=timestamp
 )
 
 except Exception as e:
 logger.debug(f"Goalpost detection error: {e}")
 return None

 def draw_on_frame(self, frame: np.ndarray, goalpost_pair: GoalpostPair) -> np.ndarray:
 """
 Draw detected goalposts on frame.
 
 Args:
 frame: Input frame
 goalpost_pair: Detected goalpost pair
 
 Returns:
 Frame with drawn goalposts
 """
 frame = frame.copy()
 
 if goalpost_pair.left:
 left = goalpost_pair.left
 x1 = int(left.x - left.width / 2)
 y1 = int(left.y - left.height / 2)
 x2 = int(left.x + left.width / 2)
 y2 = int(left.y + left.height / 2)
 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
 cv2.putText(frame, f"L:{left.confidence:.2f}", (x1, y1 - 5),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
 
 if goalpost_pair.right:
 right = goalpost_pair.right
 x1 = int(right.x - right.width / 2)
 y1 = int(right.y - right.height / 2)
 x2 = int(right.x + right.width / 2)
 y2 = int(right.y + right.height / 2)
 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
 cv2.putText(frame, f"R:{right.confidence:.2f}", (x1, y1 - 5),
 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
 
 # Draw goal center and cross
 cx, cy = int(goalpost_pair.center_x), int(goalpost_pair.center_y)
 cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
 cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 1)
 cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 1)
 
 # Draw goal width indicator
 if goalpost_pair.left and goalpost_pair.right:
 left_x = int(goalpost_pair.left.x)
 right_x = int(goalpost_pair.right.x)
 cv2.line(frame, (left_x, cy), (right_x, cy), (0, 0, 255), 2)
 
 return frame


class GoalpostTracker:
 """Tracks goalpost positions across frames for consistency."""

 def __init__(self, max_distance: float = 50.0):
 """
 Initialize tracker.
 
 Args:
 max_distance: Maximum allowed distance between detections in consecutive frames
 """
 self.max_distance = max_distance
 self.last_detection: Optional[GoalpostPair] = None
 self.detections: List[GoalpostPair] = []

 def update(self, detection: Optional[GoalpostPair]) -> Optional[GoalpostPair]:
 """
 Update tracker with new detection.
 
 Args:
 detection: Current frame's detection
 
 Returns:
 Filtered/smoothed detection
 """
 if detection is None:
 # Loss of detection
 self.last_detection = None
 return None
 
 # If first detection, accept it
 if self.last_detection is None:
 self.last_detection = detection
 self.detections.append(detection)
 return detection
 
 # Check distance from last detection
 dist = np.hypot(
 detection.center_x - self.last_detection.center_x,
 detection.center_y - self.last_detection.center_y
 )
 
 if dist > self.max_distance:
 # Too far, consider it lost
 return None
 
 # Smooth the detection using exponential moving average
 alpha = 0.3 # Smoothing factor
 smoothed = GoalpostPair(
 left=detection.left,
 right=detection.right,
 center_x=alpha * detection.center_x + (1 - alpha) * self.last_detection.center_x,
 center_y=alpha * detection.center_y + (1 - alpha) * self.last_detection.center_y,
 goal_width=alpha * detection.goal_width + (1 - alpha) * self.last_detection.goal_width,
 confidence=min(detection.confidence, self.last_detection.confidence + 0.1),
 frame_id=detection.frame_id,
 timestamp=detection.timestamp
 )
 
 self.last_detection = smoothed
 self.detections.append(smoothed)
 
 # Keep only recent history
 if len(self.detections) > 60:
 self.detections = self.detections[-60:]
 
 return smoothed
