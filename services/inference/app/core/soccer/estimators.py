from typing import Dict
from .utils import measure_distance


class SpeedAndDistanceEstimator:
    def __init__(self, frame_rate: float = 24.0):
        self.frame_window = 5
        self.frame_rate = max(1.0, frame_rate)

    def add_speed_and_distance_to_tracks(self, tracks: dict):
        total_distance: Dict[str, Dict[int, float]] = {}
        for obj_name, obj_tracks in tracks.items():
            if obj_name in ("ball", "referees"):
                continue
            n_frames = len(obj_tracks)
            for frame_num in range(0, n_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, n_frames - 1)
                for track_id in obj_tracks[frame_num]:
                    if track_id not in obj_tracks[last_frame]:
                        continue
                    start_pos = obj_tracks[frame_num][track_id].get(
                        "position_transformed"
                    )
                    end_pos = obj_tracks[last_frame][track_id].get(
                        "position_transformed"
                    )
                    if start_pos is None or end_pos is None:
                        continue
                    dist = measure_distance(start_pos, end_pos)
                    elapsed = (last_frame - frame_num) / self.frame_rate
                    if elapsed <= 0:
                        continue
                    speed_kmh = (dist / elapsed) * 3.6

                    if obj_name not in total_distance:
                        total_distance[obj_name] = {}
                    if track_id not in total_distance[obj_name]:
                        total_distance[obj_name][track_id] = 0.0
                    total_distance[obj_name][track_id] += dist

                    for fn in range(frame_num, last_frame):
                        if track_id not in tracks[obj_name][fn]:
                            continue
                        tracks[obj_name][fn][track_id]["speed"] = speed_kmh
                        tracks[obj_name][fn][track_id]["distance"] = total_distance[
                            obj_name
                        ][track_id]
