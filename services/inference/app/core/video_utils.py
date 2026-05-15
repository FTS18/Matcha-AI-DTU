import os
import logging
import subprocess
from pathlib import Path
from typing import Optional

from app.core.tts import tts_generate
from .video import (
    _run_ffmpeg,
    generate_silent_audio,
    _annotate_clip_with_soccer_analysis,
    build_clip_extraction_command,
    build_stitch_command,
    build_audio_mix_command,
    EVENT_CONFIG,
    VALID_TRANSITIONS,
    LOGO_PATH,
)

logger = logging.getLogger(__name__)


def create_highlight_reel(
    video_path: str,
    highlights: list,
    match_id: str,
    output_dir: str,
    music_dir: Path,
    tracking_data: Optional[list] = None,
    aspect_ratio: str = "16:9",
    language: str = "english",
) -> dict:
    """
    Production-grade highlight reel (robust version):
        1. Extracts per-highlight MP4 clips with event title overlay + Matcha AI watermark.
        2. Smart ball-follow crop for 9:16 vertical reels.
        3. Concatenates all clips with xfade transitions.
        4. Mixes TTS commentary + background music + crowd ambience + roar sfx.
    """
    if not highlights:
        return {"reel_url": None, "clip_urls": []}

    is_vertical = aspect_ratio == "9:16"
    has_logo = os.path.exists(LOGO_PATH)
    transition_duration = 0.8
    ar_tag = "_p" if is_vertical else ""  # suffix to avoid filename collisions

    logger.info(
        f"[Reel] Generating {'9:16' if is_vertical else '16:9'} reel "
        f"for {match_id} ({len(highlights)} clips)"
    )

    # ── Music assets ─────────────────────────────────────────────────────────
    music_path = str(music_dir / "music.mp3")
    crowd_path = str(music_dir / "crowd.mp3")
    roar_path = str(music_dir / "roar.mp3")
    if not os.path.exists(music_path):
        generate_silent_audio(music_path, duration=300.0)
    has_crowd = os.path.exists(crowd_path)
    has_roar = os.path.exists(roar_path)

    # ── Phase 1: Extract individual clips ────────────────────────────────────
    clip_details: list = []
    clip_public_urls: list = []

    for i, h in enumerate(highlights):
        start = float(h.get("startTime", 0))
        end = float(h.get("endTime", start + 10))
        clip_dur = max(end - start, 1.0)
        event_type = str(h.get("eventType") or "HIGHLIGHT").upper()
        commentary = str(h.get("commentary") or "")

        v_clip = os.path.join(output_dir, f"clip_{match_id}_{i}{ar_tag}.mp4")
        a_tts = os.path.join(output_dir, f"tts_{match_id}_{i}{ar_tag}.wav")

        # ── Build extraction command ──────────────────────────────────────
        clip_cmd = build_clip_extraction_command(
            video_path=video_path,
            start=start,
            end=end,
            event_type=event_type,
            output_path=v_clip,
            is_vertical=is_vertical,
            tracking_data=tracking_data,
        )

        # ── TTS commentary ────────────────────────────────────────────────
        has_tts = False
        if commentary:
            try:
                has_tts = tts_generate(commentary, a_tts, language=language)
            except Exception as _te:
                logger.debug(f"TTS skipped: {_te}")

        # ── Run video extraction ──────────────────────────────────────────
        ok = _run_ffmpeg(clip_cmd, timeout=120)

        if not ok or not os.path.exists(v_clip):
            # Fallback: simple extract without overlays
            logger.warning(f"Clip {i} overlay failed, retrying simple extract")
            ok = _run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(start),
                    "-to",
                    str(end),
                    "-i",
                    video_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "22",
                    "-an",
                    "-movflags",
                    "+faststart",
                    v_clip,
                ],
                timeout=120,
            )

        if ok and os.path.exists(v_clip) and os.path.getsize(v_clip) > 1024:
            # ── Soccer Analysis Overlay ──────────────────────────────────────
            # Annotate clip with player tracking, ellipses, speed & ball control
            v_clip = _annotate_clip_with_soccer_analysis(
                v_clip, output_dir, match_id, i
            )

            cfg = EVENT_CONFIG.get(event_type, EVENT_CONFIG["HIGHLIGHT"])
            pub_url = f"/uploads/{os.path.basename(v_clip)}"
            clip_public_urls.append(pub_url)
            clip_details.append(
                {
                    "video": v_clip,
                    "public_url": pub_url,
                    "tts": a_tts if has_tts else None,
                    "duration": clip_dur,
                    "transition": (
                        cfg["transition"]
                        if cfg["transition"] in VALID_TRANSITIONS
                        else "fade"
                    ),
                }
            )
            logger.info(f" clip {i}: {pub_url} ({clip_dur:.1f}s)")
        else:
            logger.warning(f" clip {i} failed completely, skipping")
            clip_public_urls.append(None)

    if not clip_details:
        logger.error("[Reel] No clips extracted — aborting reel")
        return {"reel_url": None, "clip_urls": clip_public_urls}

    # ── Phase 2: Stitch clips with xfade ─────────────────────────────────────
    if len(clip_details) == 1:
        v_stitched = clip_details[0]["video"]
    else:
        v_stitched = os.path.join(output_dir, f"v_trans_{match_id}{ar_tag}.mp4")
        stitch_cmd = build_stitch_command(
            clips=clip_details,
            output_path=v_stitched,
            transition_duration=transition_duration,
        )

        stitch_ok = _run_ffmpeg(stitch_cmd, timeout=300)
        if not stitch_ok:
            logger.warning("[Reel] Stitch failed — using first clip as fallback")
            v_stitched = clip_details[0]["video"]

    # ── Phase 3: Mix audio (TTS + music + crowd + roar — no game audio) ──
    ar_suffix = "_portrait" if is_vertical else ""
    final_reel = os.path.join(output_dir, f"highlight_reel_{match_id}{ar_suffix}.mp4")

    # Calculate start offsets
    start_offsets = [0.0]
    for j in range(len(clip_details) - 1):
        start_offsets.append(
            start_offsets[-1] + clip_details[j]["duration"] - transition_duration
        )
    total_dur = (
        start_offsets[-1] + clip_details[-1]["duration"] if clip_details else 10.0
    )

    audio_mix_cmd = build_audio_mix_command(
        v_stitched=v_stitched,
        music_path=music_path,
        crowd_path=crowd_path,
        roar_path=roar_path,
        clips=clip_details,
        output_path=final_reel,
        total_dur=total_dur,
        start_offsets=start_offsets,
        has_crowd=has_crowd,
        has_roar=has_roar,
    )

    audio_ok = _run_ffmpeg(audio_mix_cmd, timeout=300)

    if not audio_ok:
        # Fallback: copy video without complex audio
        logger.warning("[Reel] Audio mix failed, copying video-only")
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                v_stitched,
                "-c:v",
                "copy",
                "-an",
                "-movflags",
                "+faststart",
                final_reel,
            ],
            timeout=120,
        )

    # ── Cleanup temp files ────────────────────────────────────────────────────
    for c in clip_details:
        p = c.get("tts")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    trans_file = os.path.join(output_dir, f"v_trans_{match_id}{ar_tag}.mp4")
    if os.path.exists(trans_file) and trans_file != (
        clip_details[0]["video"] if clip_details else ""
    ):
        try:
            os.remove(trans_file)
        except Exception:
            pass

    if os.path.exists(final_reel) and os.path.getsize(final_reel) > 1024:
        reel_url = f"/uploads/highlight_reel_{match_id}{ar_suffix}.mp4"
        logger.info(f"[Reel] {final_reel} | clips={len(clip_public_urls)}")
        return {"reel_url": reel_url, "clip_urls": clip_public_urls}

    logger.error("[Reel] Final reel not created")
    return {"reel_url": None, "clip_urls": clip_public_urls}
