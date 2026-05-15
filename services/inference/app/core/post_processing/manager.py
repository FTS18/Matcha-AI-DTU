import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PostProcessor:
    def __init__(
        self, match_id: str, config: Dict[str, Any], uploads_dir: Path, music_dir: Path
    ):
        self.match_id = match_id
        self.config = config
        self.uploads_dir = uploads_dir
        self.music_dir = music_dir

    def run_post_processing(
        self,
        scored_events: List[Dict],
        highlights: List[Dict],
        track_frames: List[Dict],
        team_colors: List[Any],
        duration: float,
        original_video_path: str,
        total_frames: int,
        fps: float,
        language: str = "english",
    ) -> Dict[str, Any]:
        """Orchestrate all post-analysis generation tasks."""
        from app.core.visuals import VisualsManager
        from app.core.video import create_highlight_reel
        from app.core.analysis.narrative import (
            calculate_emotion_scores,
            compile_final_payload,
        )
        from app.core.audio_engine import calculate_dynamic_audio_volumes

        # 1. Highlight Reels
        urls = self._generate_reels(
            original_video_path, highlights, track_frames, language
        )

        # 2. Visuals (Thumbnail, Heatmap, Tactical Stats)
        visuals = VisualsManager(self.match_id, self.uploads_dir)
        urls["thumbnail"] = visuals.generate_thumbnail(
            original_video_path, total_frames
        )

        # Heatmap
        from app.core.soccer import generate_heatmap

        urls["heatmap"] = visuals.generate_heatmap(
            track_frames, team_colors, generate_heatmap
        )

        # Tactical Stats
        from app.core.soccer import (
            calculate_player_metrics,
            calculate_possession,
            calculate_dominance,
            generate_tactical_radar,
            analyze_team_formation,
            predict_ball_trajectory,
            estimate_ball_speed,
        )

        advanced_stats = visuals.generate_tactical_stats(
            track_frames,
            fps,
            team_colors,
            metrics_fn=calculate_player_metrics,
            possession_fn=calculate_possession,
            dominance_fn=calculate_dominance,
            radar_fn=generate_tactical_radar,
        )

        # Ball Speed
        top_speed_kmh = 0.0
        try:
            top_speed_kmh = estimate_ball_speed(track_frames, fps)
        except Exception:
            pass

        # Tactical Context
        formation_data, trajectory_data = visuals.generate_context_data(
            track_frames,
            team_colors,
            fps,
            formation_fn=(
                analyze_team_formation
                if self.config["CONTEXT_AWARE_COMMENTARY"]
                else None
            ),
            trajectory_fn=(
                predict_ball_trajectory
                if self.config["ENHANCED_BALL_TRACKING"]
                else None
            ),
        )

        # 3. Emotion & Audio
        emotion_scores = calculate_emotion_scores(
            [], duration
        )  # motion_windows passed separately in real use
        audio_volumes = {}
        if self.config["DYNAMIC_AUDIO_MIXING"] and emotion_scores:
            import numpy as np

            avg_emotion = float(np.mean([e["finalScore"] for e in emotion_scores]))
            avg_motion = 0.5  # Default
            audio_volumes = calculate_dynamic_audio_volumes(avg_motion, avg_emotion)

        # 4. Compile Final Payload
        extra_stats = {
            "topSpeed": top_speed_kmh,
            "advanced": advanced_stats,
            "formation": formation_data,
            "trajectory": trajectory_data,
            "audioVolumes": audio_volumes,
        }

        urls["video"] = f"/uploads/{Path(original_video_path).name}"

        payload = compile_final_payload(
            scored_events,
            highlights,
            emotion_scores,
            duration,
            "",
            urls,
            track_frames,
            team_colors,
            extra_stats,
        )

        return payload

    def _generate_reels(
        self,
        video_path: str,
        highlights: List[Dict],
        track_frames: List[Dict],
        language: str,
    ) -> Dict[str, Any]:
        from app.core.video import create_highlight_reel

        urls = {}

        # Landscape
        try:
            logger.info("Generating landscape reel...")
            res = create_highlight_reel(
                video_path=video_path,
                highlights=highlights,
                match_id=self.match_id,
                output_dir=str(self.uploads_dir),
                music_dir=self.music_dir,
                tracking_data=track_frames,
                aspect_ratio="16:9",
                language=language,
            )
            if isinstance(res, dict):
                urls["landscape"] = res.get("reel_url")
                # Assign clip URLs to highlights
                clip_urls = res.get("clip_urls", [])
                for i, h in enumerate(highlights):
                    if i < len(clip_urls):
                        h["videoUrl"] = clip_urls[i]
        except Exception as e:
            logger.warning(f"Landscape reel failed: {e}")

        # Portrait
        try:
            logger.info("Generating portrait reel...")
            res = create_highlight_reel(
                video_path=video_path,
                highlights=highlights,
                match_id=self.match_id,
                output_dir=str(self.uploads_dir),
                music_dir=self.music_dir,
                tracking_data=track_frames,
                aspect_ratio="9:16",
                language=language,
            )
            if isinstance(res, dict):
                urls["portrait"] = res.get("reel_url")
        except Exception as e:
            logger.warning(f"Portrait reel failed: {e}")

        return urls
