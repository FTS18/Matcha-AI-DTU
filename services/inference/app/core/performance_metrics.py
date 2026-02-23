import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def calculate_player_metrics(track_frames: List[Dict], fps: float) -> Dict:
    """
    Calculate high-level performance metrics for all players.
    - Total distance covered (meters)
    - Top speed (km/h)
    - High Intensity Sprints (>20 km/h)
    """
    player_data = {} # {track_id: {"positions": [], "distances": [], "speeds": []}}
    
    # We assume a 105m x 68m pitch
    PITCH_W = 105.0
    PITCH_H = 68.0
    
    for frame in track_frames:
        timestamp = frame.get("t", 0)
        players = frame.get("p", [])
        
        for p in players:
            if len(p) < 5: continue
            nx, ny, nw, nh, tid = p[0], p[1], p[2], p[3], p[4]
            if tid == -1: continue
            
            # Use feet position (bottom center)
            px = (nx + nw/2) * PITCH_W
            py = (ny + nh) * PITCH_H
            
            if tid not in player_data:
                player_data[tid] = {"positions": [], "total_dist": 0.0, "top_speed": 0.0, "sprints": 0}
            
            player_data[tid]["positions"].append((timestamp, px, py))

    # Calculate metrics
    results = {}
    for tid, data in player_data.items():
        positions = data["positions"]
        if len(positions) < 2: continue
        
        total_dist = 0.0
        speeds = []
        
        for i in range(1, len(positions)):
            t1, x1, y1 = positions[i-1]
            t2, x2, y2 = positions[i]
            dt = t2 - t1
            
            if 0 < dt < 1.0: # Ignore teleportations or large gaps
                dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_dist += dist
                speed_kmh = (dist / dt) * 3.6
                if speed_kmh < 40: # Ignore noise
                    speeds.append(speed_kmh)
        
        if not speeds: continue
        
        top_speed = float(np.percentile(speeds, 95))
        sprint_count = sum(1 for s in speeds if s > 20.0) # Sprints over 20km/h
        
        results[str(tid)] = {
            "total_distance_m": round(total_dist, 1),
            "top_speed_kmh": round(top_speed, 1),
            "sprints": sprint_count,
            "average_speed_kmh": round(np.mean(speeds), 1),
            "activity_profile": _analyze_activity_profile(positions, speeds)
        }
        
    return results

def _analyze_activity_profile(positions: List, speeds: List) -> Dict:
    """
    Detect behavioral patterns like intensity drops or bursts.
    """
    if not speeds: return {"intensity": "low"}
    
    avg_speed = np.mean(speeds)
    high_intensity_pct = sum(1 for s in speeds if s > 15.0) / len(speeds)
    
    profile = "Moderate"
    if high_intensity_pct > 0.4: profile = "High Intensity"
    elif avg_speed < 5.0: profile = "Walking/Static"
    
    return {
        "intensity": profile,
        "high_intensity_effort_pct": round(high_intensity_pct * 100, 1)
    }
