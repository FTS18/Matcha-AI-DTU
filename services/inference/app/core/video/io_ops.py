import os
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_ffmpeg(cmd: list, timeout: int = 120) -> bool:
    """Run ffmpeg command, return True on success."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")[-800:]
            logger.warning(f"ffmpeg non-zero exit ({result.returncode}): {stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"ffmpeg error: {e}")
        return False


def generate_silent_audio(output_path: str, duration: float = 10.0) -> bool:
    """Generate a silent audio file of the given duration."""
    try:
        return _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                str(duration),
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                output_path,
            ]
        )
    except Exception as e:
        logger.error(f"Failed to generate silent audio: {e}")
        return False


def precompress_video(video_path: str, match_id: str, config: dict) -> str:
    """
    Pre-compresses large videos before heavy analysis to save GPU memory and time.
    Downscales to target height and fixes FPS.
    """
    try:
        # Check size (MB)
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if size_mb < config.get("COMPRESS_SIZE_THRESHOLD_MB", 100):
            return video_path

        out_path = os.path.join(
            os.path.dirname(video_path), f"compressed_{match_id}.mp4"
        )
        if os.path.exists(out_path):
            return out_path

        logger.info(f"Pre-compressing {size_mb:.1f}MB video for faster analysis...")

        target_h = config.get("COMPRESS_OUTPUT_HEIGHT", 480)
        target_fps = config.get("COMPRESS_OUTPUT_FPS", 1)

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"scale=-2:{target_h},fps={target_fps}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-an",  # Remove audio for analysis speed
            out_path,
        ]

        if _run_ffmpeg(cmd, timeout=config.get("FFMPEG_COMPRESS_TIMEOUT", 300)):
            return out_path

    except Exception:
        logger.warning(f"Pre-compression failed, using original video.")

    return video_path
