import cv2
import numpy as np
from typing import List, Dict

def generate_tactical_radar(track_frames: List[Dict], output_path: str, team_colors: List):
    """
    Generate a 2D Tactical Radar Video overlay or final summary image.
    Shows the Voronoi / Territory control of both teams.
    """
    PITCH_W, PITCH_H = 800, 520
    canvas = np.zeros((PITCH_H, PITCH_W, 3), dtype=np.uint8)
    
    # Draw simple pitch
    cv2.rectangle(canvas, (0,0), (PITCH_W, PITCH_H), (30, 100, 30), -1)
    
    # Gather last positions to show a "Current State" radar
    last_frame = track_frames[-1] if track_frames else {}
    players = last_frame.get("p", [])
    
    if not players: return
    
    # Create points for Voronoi
    points = []
    colors = []
    for p in players:
        if len(p) < 6: continue
        nx, ny, nw, nh, tid, team_idx = p[0], p[1], p[2], p[3], p[4], int(p[5])
        points.append([int((nx + nw/2) * PITCH_W), int((ny + nh) * PITCH_H)])
        colors.append(team_colors[team_idx % 2])

    # Draw Points
    for pt, col in zip(points, colors):
        cv2.circle(canvas, tuple(pt), 6, col, -1)
        cv2.circle(canvas, tuple(pt), 7, (255,255,255), 1)

    cv2.imwrite(output_path, canvas)

def calculate_possession(track_frames: List[Dict]) -> Dict:
    """
    Calculate team possession percentage based on ball proximity to players.
    """
    possession = {0: 0, 1: 0}
    total_samples = 0
    
    for frame in track_frames:
        balls = frame.get("b", [])
        players = frame.get("p", [])
        if not balls or not players: continue
        
        bx, by = balls[0][0] + balls[0][2]/2, balls[0][1] + balls[0][3]/2
        
        # Find closest player
        min_dist = 999
        closest_team = -1
        
        for p in players:
            if len(p) < 6: continue
            px, py = p[0] + p[2]/2, p[1] + p[3]
            dist = np.sqrt((bx-px)**2 + (by-py)**2)
            if dist < min_dist:
                min_dist = dist
                closest_team = int(p[5])
        
        if closest_team != -1 and min_dist < 0.15: # Ball must be within 15% of screen
            possession[closest_team] += 1
            total_samples += 1
            
    if total_samples == 0: return {"team_a": 50, "team_b": 50}
    
    p_a = round((possession[0] / total_samples) * 100)
    return {"team_a": p_a, "team_b": 100 - p_a}

def calculate_dominance(track_frames: List[Dict]) -> Dict:
    """
    Calculate which team 'dominates' the pitch territory.
    """
    territory = {0: 0.0, 1: 0.0}
    
    for frame in track_frames:
        players = frame.get("p", [])
        if not players: continue
        
        # Calculate average X position of players (weighted toward the opposition goal)
        # Assuming team 0 attacks right (>0.5) and team 1 attacks left (<0.5)
        pos_a = [p[0] for p in players if len(p) >= 6 and p[5] == 0]
        pos_b = [p[0] for p in players if len(p) >= 6 and p[5] == 1]
        
        if pos_a: territory[0] += np.mean(pos_a)
        if pos_b: territory[1] += (1.0 - np.mean(pos_b))
        
    total = sum(territory.values()) or 1.0
    return {
        "team_a": round((territory[0] / total) * 100, 1),
        "team_b": round((territory[1] / total) * 100, 1)
    }
