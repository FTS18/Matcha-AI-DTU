import os
import logging
import cv2
import subprocess
from .io_ops import _run_ffmpeg

logger = logging.getLogger(__name__)

try:
    from app.core.soccer import process_clip_frames as _sa_process
except ImportError:
    _sa_process = None


def _annotate_clip_with_soccer_analysis(
    clip_path: str, output_dir: str, match_id: str, clip_idx: int
) -> str:
    """
    Read a raw video clip, run the soccer-analysis overlay pipeline on its
    frames (player ellipses, speed/distance, ball control %), then write the
    annotated frames back to a new file. Returns the path to the annotated
    clip (or the original clip path on failure).
    """
    if _sa_process is None:
        return clip_path

    try:
        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return clip_path

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return clip_path

        logger.info(
            f"Soccer analysis: annotating clip {clip_idx} "
            f"({len(frames)} frames, {w}x{h} @ {fps:.1f} fps)"
        )
        annotated = _sa_process(frames, fps=fps)

        if annotated is None or len(annotated) == 0:
            return clip_path

        annotated_path = os.path.join(output_dir, f"sa_v_{match_id}_{clip_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))
        for f in annotated:
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h))
            writer.write(f)
        writer.release()

        # Re-encode to H.264 so ffmpeg can concat/xfade it reliably
        h264_path = os.path.join(output_dir, f"sa_h264_{match_id}_{clip_idx}.mp4")
        re_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            annotated_path,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-an",
            h264_path,
        ]
        if _run_ffmpeg(re_cmd):
            if os.path.exists(annotated_path):
                os.remove(annotated_path)
            if os.path.exists(clip_path) and clip_path != h264_path:
                os.remove(clip_path)
            return h264_path
        else:
            if os.path.exists(clip_path) and clip_path != annotated_path:
                os.remove(clip_path)
            return annotated_path

    except Exception as e:
        logger.error(
            f"Soccer analysis annotation failed for clip {clip_idx}: {e}", exc_info=True
        )
        return clip_path
