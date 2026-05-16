import os
import logging
from typing import Optional, Dict, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def _download_youtube_video(
    url: str,
    output_dir: str,
    filename_prefix: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """Download YouTube video using yt-dlp to output_dir."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    logger.info(f"Downloading YouTube video: {url} (range: {start_time}-{end_time})")

    out_tmpl = str(Path(output_dir) / f"{filename_prefix}_yt.%(ext)s")

    def progress_hook(d):
        if d["status"] == "downloading" and progress_callback:
            percent = 0.0
            if "downloaded_bytes" in d and "total_bytes" in d:
                percent = (d["downloaded_bytes"] / d["total_bytes"]) * 100
            elif "_percent_str" in d:
                try:
                    percent = float(d["_percent_str"].strip("%"))
                except Exception:
                    pass
            progress_callback(percent)

    # If range is specified, use it. Otherwise default to first 3 hours.
    start = int(start_time) if start_time is not None else 0
    end = int(end_time) if end_time is not None else 10800

    ydl_opts: Dict = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_tmpl,
        "quiet": False,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "progress_hooks": [progress_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            dl_path = ydl.prepare_filename(info)
            if not dl_path.endswith(".mp4"):
                dl_path = dl_path.rsplit(".", 1)[0] + ".mp4"

            if os.path.exists(dl_path):
                logger.info(f"Downloaded YouTube video to: {dl_path}")
                return dl_path
            else:
                for f in Path(output_dir).glob(f"{filename_prefix}_yt.*"):
                    return str(f)

            raise Exception("Download completed but file not found")
    except Exception as e:
        logger.error("yt-dlp download failed (internal error suppressed for security)")
        raise ValueError("Failed to download YouTube video")
