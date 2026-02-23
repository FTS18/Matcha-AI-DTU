"""
Physics-based Computer Vision Event Detector.
Detects football events directly from YOLO bounding box tracking data (track_frames)
using velocity, proximity, spatial clustering, and colour heuristics.

Operates entirely on CPU without requiring additional deep learning models.
"""

import math
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Goal zone heuristic (assumes camera is roughly midfield, goals are at the far edges)
# x < 0.10 = left goal zone, x > 0.90 = right goal zone (tightened from 0.15/0.85)
GOAL_ZONE_LEFT = 0.10
GOAL_ZONE_RIGHT = 0.90


def _distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _iou(box1, box2):
    """Calculate Intersection over Union of two [x, y, w, h] boxes."""
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1
    
    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2
    
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0
        
    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    return inter_area / float(box1_area + box2_area - inter_area)


def _get_player_centroid(p):
    """p is [x, y, w, h, tid, ...]"""
    return (p[0] + p[2] / 2, p[1] + p[3] / 2)


def _get_ball_centroid(b):
    """b is [x, y, w, h, conf]"""
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


class PhysicsDetector:
    def __init__(self, fps: float):
        self.fps = max(1.0, fps)
        self.history = []  # rolling buffer of frames
        self.events = []
        
        # State
        self.ball_velocities = []
        self.last_ball_pos = None
        self.goal_zone_dwell_frames = 0
        
    def process_frame(self, frame_data: dict):
        """frame_data is a single dict from track_frames: {'t': 1.2, 'b': [...], 'p': [...]}"""
        t = frame_data["t"]
        balls = frame_data.get("b", [])
        players = frame_data.get("p", [])
        
        self.history.append(frame_data)
        if len(self.history) > int(self.fps * 4):  # keep 4 seconds
            self.history.pop(0)
            
        # 1. Ball Trajectory (Saves & Goals)
        if balls:
            best_ball = max(balls, key=lambda b: b[4]) if len(balls) > 1 else balls[0]
            curr_pos = _get_ball_centroid(best_ball)
            
            if self.last_ball_pos:
                # Velocity as % of screen width per second
                dist = _distance(curr_pos, self.last_ball_pos)
                velocity = dist * self.fps
                self.ball_velocities.append(velocity)
                if len(self.ball_velocities) > int(self.fps):
                    self.ball_velocities.pop(0)
                
                # Sharp deceleration (SAVE/TACKLE context)
                if len(self.ball_velocities) >= 3:
                    avg_v = sum(self.ball_velocities[:-1]) / len(self.ball_velocities[:-1])
                    if avg_v > 0.5 and velocity < 0.1:  # Ball was fast, suddenly stopped
                        # Check proximity to a player (goalkeeper or defender)
                        for p in players:
                            if _distance(curr_pos, _get_player_centroid(p)) < 0.05:
                                is_goal_area = curr_pos[0] < GOAL_ZONE_LEFT or curr_pos[0] > GOAL_ZONE_RIGHT
                                ev_type = "SAVE" if is_goal_area else "TACKLE"
                                self.events.append({
                                    "timestamp": t, "type": ev_type, "confidence": 0.75, 
                                    "source": "cv_physics", "desc": f"Ball deceleration block ({ev_type})"
                                })
                                self.ball_velocities.clear()
                                break
                                
            # Goal Zone Dwell (Ball stays inside goal area = GOAL)
            # Ball must be in the goal zone: x < GOAL_ZONE_LEFT or x > GOAL_ZONE_RIGHT
            if curr_pos[0] < GOAL_ZONE_LEFT or curr_pos[0] > GOAL_ZONE_RIGHT:
                self.goal_zone_dwell_frames += 1
                if self.goal_zone_dwell_frames == int(self.fps * 1.0): # 1s dwell in goal zone
                    # Extra guard: require ball was moving towards goal before dwell
                    was_moving_toward_goal = False
                    if self.last_ball_pos is not None:
                        if curr_pos[0] < GOAL_ZONE_LEFT and self.last_ball_pos[0] > curr_pos[0]:
                            was_moving_toward_goal = True
                        elif curr_pos[0] > GOAL_ZONE_RIGHT and self.last_ball_pos[0] < curr_pos[0]:
                            was_moving_toward_goal = True
                    if was_moving_toward_goal or self.goal_zone_dwell_frames >= int(self.fps * 2.0):
                        self.events.append({
                            "timestamp": t, "type": "GOAL", "confidence": 0.85,
                            "source": "cv_physics", "desc": "Ball dwelled in goal zone"
                        })
            else:
                self.goal_zone_dwell_frames = 0
                
            # Shot Detection (Acceleration from foot towards goal)
            if self.last_ball_pos and len(self.ball_velocities) >= 3:
                avg_v = sum(self.ball_velocities[:-1]) / len(self.ball_velocities[:-1])
                dist = _distance(curr_pos, self.last_ball_pos)
                velocity = dist * self.fps
                
                if velocity > 0.4 and avg_v < 0.2:  # Sudden high acceleration
                    moving_left = curr_pos[0] < self.last_ball_pos[0]
                    towards_goal = (moving_left and curr_pos[0] < 0.5) or (not moving_left and curr_pos[0] > 0.5)
                    
                    if towards_goal:
                        for p in players:
                            if len(p) >= 42:  # has pose keypoints (8 base + 34 kps)
                                lax, lay = p[38], p[39]
                                rax, ray = p[40], p[41]
                                # Check distance from ball to ankles
                                if (lax > 0 and lay > 0 and _distance(self.last_ball_pos, (lax, lay)) < 0.08) or \
                                   (rax > 0 and ray > 0 and _distance(self.last_ball_pos, (rax, ray)) < 0.08):
                                    self.events.append({
                                        "timestamp": t, "type": "SHOT", "confidence": 0.85, 
                                        "source": "cv_physics", "desc": "Shot from player foot detected"
                                    })
                                    break
                
            self.last_ball_pos = curr_pos
        else:
            self.last_ball_pos = None
            
        # 2. Player Proximity (Tackles / Fouls)
        # Look for bounding box overlaps indicating collisions
        for i, p1 in enumerate(players):
            for j, p2 in enumerate(players[i+1:]):
                overlap = _iou(p1[:4], p2[:4])
                if overlap > 0.15:  # Significant overlap
                    self.events.append({
                        "timestamp": t - 0.5, "type": "TACKLE", "confidence": min(0.4 + overlap, 0.8),
                        "source": "cv_physics", "desc": "Player collision/tackle"
                    })

        # 3. Spatial Clustering (Corners / Free Kicks / Celebrations)
        if len(players) >= 5:
            centroids = [_get_player_centroid(p) for p in players]
            # Simple O(N^2) clustering: count players within 0.15 screen width of each other
            max_cluster_size = 0
            cluster_center = None
            
            for c1 in centroids:
                size = sum(1 for c2 in centroids if _distance(c1, c2) < 0.15)
                if size > max_cluster_size:
                    max_cluster_size = size
                    cluster_center = c1
                    
            if max_cluster_size >= 6 and cluster_center is not None:  # 6+ players in tight cluster
                # Check location
                if cluster_center[0] < 0.1 or cluster_center[0] > 0.9:
                    if cluster_center[1] < 0.2 or cluster_center[1] > 0.8:
                        self.events.append({
                            "timestamp": t, "type": "CORNER", "confidence": 0.8,
                            "source": "cv_physics", "desc": "Player cluster at corner flag"
                        })
                elif (cluster_center[0] < 0.25 or cluster_center[0] > 0.75) and 0.3 < cluster_center[1] < 0.7:
                    # In box/penalty arc — could be free kick or celebration
                    # Check recent events for a goal
                    recent_goal = any(e["type"] == "GOAL" for e in self.events if t - e["timestamp"] < 15.0)
                    if recent_goal:
                        self.events.append({
                            "timestamp": t, "type": "CELEBRATION", "confidence": 0.8,
                            "source": "cv_physics", "desc": "Team cluster post-goal"
                        })
                    else:
                        self.events.append({
                            "timestamp": t, "type": "FOUL", "confidence": 0.6,
                            "source": "cv_physics", "desc": "Player cluster (free kick setup)"
                        })

        # 4. Kickoff (SoG) Detection
        if balls and players:
            # Check if ball is strictly central
            best_ball = max(balls, key=lambda b: b[4]) if len(balls) > 1 else balls[0]
            curr_pos = _get_ball_centroid(best_ball)
            if 0.45 < curr_pos[0] < 0.55 and 0.45 < curr_pos[1] < 0.55:
                # Count players near the ball
                near_players = sum(1 for p in players if _distance(curr_pos, _get_player_centroid(p)) < 0.15)
                
                # Check team separation
                team_0_xs = [ _get_player_centroid(p)[0] for p in players if len(p) > 5 and p[5] == 0 ]
                team_1_xs = [ _get_player_centroid(p)[0] for p in players if len(p) > 5 and p[5] == 1 ]
                
                if near_players >= 2 and len(team_0_xs) > 3 and len(team_1_xs) > 3:
                    avg_x0 = sum(team_0_xs) / len(team_0_xs)
                    avg_x1 = sum(team_1_xs) / len(team_1_xs)
                    
                    if abs(avg_x0 - avg_x1) > 0.15: # Clear separation
                        self.events.append({
                            "timestamp": t, "type": "KICKOFF", "confidence": 0.9,
                            "source": "cv_physics", "desc": "Start of Game / Kickoff detected"
                        })

    def get_deduplicated_events(self) -> List[Dict]:
        """Merge events of same type within 5 seconds, keeping highest confidence."""
        if not self.events:
            return []
            
        # Sort by timestamp
        sorted_evs = sorted(self.events, key=lambda x: x["timestamp"])
        merged = []
        
        for ev in sorted_evs:
            if not merged:
                merged.append(ev)
                continue
                
            last_ev = merged[-1]
            if ev["type"] == last_ev["type"] and (ev["timestamp"] - last_ev["timestamp"]) < 5.0:
                # Merge: keep highest confidence, update timestamp to the max conf one
                if ev["confidence"] > last_ev["confidence"]:
                    merged[-1] = ev
            else:
                merged.append(ev)
                
        return merged


def detect_all(track_frames: List[Dict], fps: float) -> List[Dict]:
    """
    Run all CV physics heuristics over the full match track_frames.
    Returns list of {"timestamp": float, "type": str, "confidence": float, "source": "cv_physics"}
    """
    if not track_frames:
        return []
        
    logger.info("Starting CV Physics event detection pass...")
    detector = PhysicsDetector(fps)
    
    # Process every 4th frame (don't need high framerate for box heuristics)
    for i, frame in enumerate(track_frames):
        if i % 4 == 0:
            detector.process_frame(frame)
            
    events = detector.get_deduplicated_events()
    
    # Filter out low confidence generic tackles to prevent spam
    filtered = []
    for ev in events:
        if ev["type"] == "TACKLE" and ev["confidence"] < 0.65:
            continue
        filtered.append(ev)
        
    logger.info(f"CV Physics detected {len(filtered)} events")
    for e in filtered:
        logger.debug(f"  [cv] {e['type']} at {e['timestamp']}s ({e['confidence']:.2f})")
        
    return filtered
