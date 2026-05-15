import numpy as np
from typing import Dict, Any, Optional


class TeamAssigner:
    """K-Means based team colour assignment from jersey crops."""

    def __init__(self):
        self.team_colors: Dict[int, Any] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans = None

    def _get_clustering_model(self, image: np.ndarray):
        from sklearn.cluster import KMeans

        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        if image.size == 0:
            return np.array([128, 128, 128])
        top_half = image[0 : int(image.shape[0] / 2), :]
        if top_half.size == 0:
            return np.array([128, 128, 128])
        kmeans = self._get_clustering_model(top_half)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame: np.ndarray, player_detections: dict):
        from sklearn.cluster import KMeans

        player_colors = []
        for _, det in player_detections.items():
            bbox = det["bbox"]
            try:
                c = self.get_player_color(frame, bbox)
                player_colors.append(c)
            except Exception:
                pass
        if len(player_colors) < 2:
            self.team_colors = {1: np.array([220, 60, 60]), 2: np.array([60, 100, 220])}
            return
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(
        self, frame: np.ndarray, player_bbox: list, player_id: int
    ) -> int:
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        if self.kmeans is None:
            return 1
        try:
            player_color = self.get_player_color(frame, player_bbox)
            team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        except Exception:
            team_id = 1
        self.player_team_dict[player_id] = team_id
        return team_id
