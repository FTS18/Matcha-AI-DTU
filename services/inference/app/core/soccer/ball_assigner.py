from .utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_distance: float = 70.0):
        self.max_player_ball_distance = max_distance

    def assign_ball_to_player(self, players: dict, ball_bbox: list) -> int:
        ball_position = get_center_of_bbox(ball_bbox)
        minimum_distance = 99999.0
        assigned_player = -1
        for player_id, player in players.items():
            player_bbox = player["bbox"]
            dist_left = measure_distance(
                (player_bbox[0], player_bbox[-1]), ball_position
            )
            dist_right = measure_distance(
                (player_bbox[2], player_bbox[-1]), ball_position
            )
            distance = min(dist_left, dist_right)
            if distance < self.max_player_ball_distance and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id
        return assigned_player
