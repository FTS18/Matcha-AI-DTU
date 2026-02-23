import os
import logging
import subprocess
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from app.core.tts import tts_generate

logger = logging.getLogger(__name__)

# ── Logo path (top-right watermark) ──────────────────────────────────────────
LOGO_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "apps" / "web" / "public" / "favicons" / "logo.png")

# ── Event-specific overlay configs ───────────────────────────────────────────
EVENT_CONFIG = {
    "GOAL":        {"color": (0, 255, 100),  "emoji": "⚽", "title": "GOAL!",        "transition": "circleopen"},
    "SAVE":        {"color": (50, 180, 255), "emoji": "🧤", "title": "GREAT SAVE",   "transition": "fadeblack"},
    "TACKLE":      {"color": (255, 165, 0),  "emoji": "💥", "title": "TACKLE",       "transition": "slideleft"},
    "FOUL":        {"color": (255, 50, 50),  "emoji": "🟨", "title": "FOUL",         "transition": "fadeblack"},
    "CELEBRATION": {"color": (255, 215, 0),  "emoji": "🎉", "title": "CELEBRATION",  "transition": "circleopenclose"},
    "HIGHLIGHT":   {"color": (200, 200, 255),"emoji": "🔥", "title": "KEY MOMENT",   "transition": "fade"},
}


def generate_silent_audio(output_path: str, duration: float = 10.0) -> bool:
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={duration}",
            "-t", str(duration), "-q:a", "9", "-acodec", "libmp3lame",
            output_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to generate silent audio: {e}")
        return False


def _get_ball_focus_region(tracking_data: list, start: float, end: float) -> tuple:
    """
    Analyze ball positions in a time window and return the median (x, y) 
    normalized center point for smart crop / pan.
    """
    ball_xs, ball_ys = [], []
    for tf in tracking_data:
        t = tf.get("t", 0)
        if start <= t <= end and tf.get("b"):
            bx, by, bw, bh = tf["b"][0][:4]
            ball_xs.append(bx + bw / 2)
            ball_ys.append(by + bh / 2)
    
    if ball_xs:
        return float(np.median(ball_xs)), float(np.median(ball_ys))
    return 0.5, 0.5  # Center fallback


def _build_event_overlay_filter(event_type: str, duration: float, is_vertical: bool) -> str:
    """
    Build an FFmpeg drawtext filter for a stylish event-type title card 
    (e.g., "⚽ GOAL!" with animated fade-in).
    No more scrolling marquee — this is a centered, professional overlay.
    """
    cfg = EVENT_CONFIG.get(event_type, EVENT_CONFIG["HIGHLIGHT"])
    title = cfg["title"]
    r, g, b = cfg["color"]
    
    # Font sizes based on orientation
    title_size = 56 if not is_vertical else 44
    sub_size = 22 if not is_vertical else 18
    
    # Position: centered horizontally, upper-third vertically
    # Alpha animation: fade in from 0 to 1 over 0.5s, hold 3s, fade out over 0.5s
    alpha_expr = f"if(lt(t,0.5),t/0.5,if(lt(t,3.5),1,if(lt(t,4),(4-t)/0.5,0)))"
    
    filters = []
    
    # Title background bar (semi-transparent)
    if not is_vertical:
        filters.append(
            f"drawbox=x=0:y=ih*0.08:w=iw:h=80:color=black@0.6:t=fill"
            f":enable='between(t,0.3,4)'"
        )
    else:
        filters.append(
            f"drawbox=x=0:y=ih*0.06:w=iw:h=70:color=black@0.6:t=fill"
            f":enable='between(t,0.3,4)'"
        )

    # Main event title (centered)
    hex_color = f"{r:02x}{g:02x}{b:02x}"
    filters.append(
        f"drawtext=text='{title}'"
        f":fontcolor=0x{hex_color}:fontsize={title_size}"
        f":x=(w-text_w)/2:y={'ih*0.09' if not is_vertical else 'ih*0.065'}"
        f":alpha='{alpha_expr}'"
        f":borderw=2:bordercolor=black"
    )
    
    # Subtle "Matcha AI" watermark text (bottom-left)
    filters.append(
        f"drawtext=text='Matcha AI'"
        f":fontcolor=white@0.5:fontsize=16"
        f":x=15:y=h-30"
    )
    
    return ",".join(filters)


def _build_logo_overlay(logo_path: str, is_vertical: bool) -> str:
    """
    Returns the FFmpeg filter to overlay the Matcha logo (top-right corner).
    The logo is scaled down and placed with padding.
    """
    if not os.path.exists(logo_path):
        return ""
    
    logo_size = 60 if not is_vertical else 45
    pad = 15
    # This will be applied via a separate overlay input
    return f"scale={logo_size}:{logo_size}"


def create_highlight_reel(
    video_path: str, highlights: list, match_id: str, output_dir: str,
    music_dir: Path, tracking_data: Optional[list] = None,
    aspect_ratio: str = "16:9", language: str = "english"
) -> dict:
    """
    Production-grade highlight reel:
    1. Smart crop/pan to follow the ball (especially for 9:16 reels).
    2. Event-type title cards with animated fade-in/out.
    3. Matcha AI logo watermark (top-right).
    4. Professional xfade transitions per event type.
    5. Layered audio: Game sound + Music + Crowd + TTS commentary.
    Returns: {'reel_url': str|None, 'clip_urls': [str, ...]} for standalone playback.
    """
    if not highlights:
        return {'reel_url': None, 'clip_urls': []}

    is_vertical = (aspect_ratio == "9:16")
    logger.info(f"🎬 Generating {'9:16 vertical' if is_vertical else '16:9 horizontal'} reel "
                f"for match {match_id} ({len(highlights)} highlights)")

    # ── Audio Assets ────────────────────────────────────────────────────────
    music_path = str(music_dir / "music.mp3")
    crowd_path = str(music_dir / "crowd.mp3")
    roar_path  = str(music_dir / "roar.mp3")
    for p in [music_path, crowd_path, roar_path]:
        if not os.path.exists(p):
            generate_silent_audio(p, duration=10.0)

    # ── Logo ────────────────────────────────────────────────────────────────
    has_logo = os.path.exists(LOGO_PATH)

    # ── Phase 1: Extract individual clips ───────────────────────────────────
    clip_details = []
    clip_public_urls = []  # Per-clip public URLs kept for standalone playback
    transition_duration = 0.8

    for i, h in enumerate(highlights):
        start, end = h["startTime"], h["endTime"]
        text = h.get("commentary", "")
        event_type = h.get("eventType", "HIGHLIGHT")
        clip_dur = end - start

        # Use named clips saved to uploads dir for public access
        v_clip  = os.path.join(output_dir, f"clip_{match_id}_{i}.mp4")
        a_tts   = os.path.join(output_dir, f"tts_{match_id}_{i}.wav")
        a_game  = os.path.join(output_dir, f"game_{match_id}_{i}.wav")

        # ── Video Filters ───────────────────────────────────────────────
        v_filters = []

        # 1. Smart Ball-Follow Crop for 9:16
        if is_vertical and tracking_data:
            focus_x, focus_y = _get_ball_focus_region(tracking_data, start, end)
            # Crop a vertical slice (9:16 from 16:9 source)
            cw_norm = (9 / 16) / (16 / 9)  # ~0.3125
            x_start = max(0.0, min(1.0 - cw_norm, focus_x - cw_norm / 2))
            v_filters.append(f"crop=iw*{cw_norm:.4f}:ih:{x_start:.4f}*iw:0,scale=720:1280")
        elif is_vertical:
            v_filters.append("crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale=720:1280")

        # 2. Event title overlay (animated, no marquee)
        overlay_filter = _build_event_overlay_filter(event_type, clip_dur, is_vertical)
        v_filters.append(overlay_filter)

        v_filter_str = ",".join(v_filters) if v_filters else "null"

        # ── Build FFmpeg command ────────────────────────────────────────
        extract_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end),
            "-i", video_path,
        ]
        
        # Add logo as overlay input
        if has_logo:
            extract_cmd.extend(["-i", LOGO_PATH])
            logo_size = 55 if not is_vertical else 40
            pad = 12
            # Combine video filter + logo overlay
            full_filter = f"[0:v]{v_filter_str}[main];[1:v]scale={logo_size}:{logo_size},format=rgba[logo];[main][logo]overlay=W-w-{pad}:{pad}[out]"
            extract_cmd.extend([
                "-filter_complex", full_filter,
                "-map", "[out]",
                "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                v_clip
            ])
        else:
            extract_cmd.extend([
                "-vf", v_filter_str,
                "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                v_clip
            ])

        # ── TTS ─────────────────────────────────────────────────────────
        has_tts = tts_generate(text, a_tts, language=language) if text else False

        # ── Game Audio ──────────────────────────────────────────────────
        has_game_audio = False
        extract_a = [
            "ffmpeg", "-y",
            "-ss", str(start), "-to", str(end), "-i", video_path,
            "-vn", "-c:a", "pcm_s16le", a_game
        ]
        if subprocess.run(extract_a, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
            has_game_audio = True

        # Run video extraction
        res = subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(v_clip):
            cfg = EVENT_CONFIG.get(event_type, EVENT_CONFIG["HIGHLIGHT"])
            clip_public_url = f"/uploads/clip_{match_id}_{i}.mp4"
            clip_public_urls.append(clip_public_url)
            clip_details.append({
                "video": v_clip,
                "public_url": clip_public_url,
                "tts": a_tts if has_tts else None,
                "game_audio": a_game if has_game_audio else None,
                "duration": clip_dur,
                "event_type": event_type,
                "transition": cfg["transition"],
            })
        else:
            logger.warning(f"Clip {i} extraction failed (rc={res.returncode})")
            clip_public_urls.append(None)  # Keep index alignment

    if not clip_details:
        logger.error("No clips were extracted successfully")
        return {'reel_url': None, 'clip_urls': []}

    # ── Phase 2: Transition Chain ───────────────────────────────────────────
    # If only 1 clip, skip transitions
    if len(clip_details) == 1:
        v_output = clip_details[0]["video"]
    else:
        # Build xfade chain with per-event transition styles
        v_output = os.path.join(output_dir, f"v_trans_{match_id}.mp4")
        v_cmd = ["ffmpeg", "-y"]
        for c in clip_details:
            v_cmd.extend(["-i", c["video"]])

        filter_parts = []
        last_v = "[0:v]"
        current_offset = clip_details[0]["duration"] - transition_duration

        for i in range(1, len(clip_details)):
            next_v = f"[{i}:v]"
            out_v = f"[v{i}]"
            trans = clip_details[i]["transition"]
            filter_parts.append(
                f"{last_v}{next_v}xfade=transition={trans}"
                f":duration={transition_duration}:offset={current_offset:.2f}{out_v}"
            )
            last_v = out_v
            current_offset += clip_details[i]["duration"] - transition_duration

        v_cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", last_v,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            v_output
        ])

        res = subprocess.run(v_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            logger.error(f"Transition chain failed: {res.stderr[-500:]}")
            # Fallback: just use the first clip
            v_output = clip_details[0]["video"]

    # ── Phase 3: Audio Mixing ───────────────────────────────────────────────
    final_reel = os.path.join(output_dir, f"highlight_reel_{match_id}.mp4")

    final_cmd = ["ffmpeg", "-y", "-i", v_output]
    bg_paths = [music_path, crowd_path, roar_path]
    for p in bg_paths:
        final_cmd.extend(["-stream_loop", "-1", "-i", p])

    # Collect segment audio inputs
    seg_paths = []
    for c in clip_details:
        if c["tts"]:
            seg_paths.append(c["tts"])
        if c["game_audio"]:
            seg_paths.append(c["game_audio"])
    for p in seg_paths:
        final_cmd.extend(["-i", p])

    # Build audio filter
    filter_parts = []
    # Background layers
    filter_parts.append("[1:a]volume=0.08[m]")
    filter_parts.append("[2:a]volume=0.30[c]")
    filter_parts.append("[3:a]volume=0.20[r]")
    filter_parts.append("[m][c][r]amix=inputs=3:duration=first[base_bg]")

    # Segment audio with adelay
    start_offsets = [0.0]
    for j in range(len(clip_details) - 1):
        start_offsets.append(start_offsets[-1] + clip_details[j]["duration"] - transition_duration)

    seg_idx = 4
    seg_labels = []
    duck_exprs = []

    for j, c in enumerate(clip_details):
        offset_ms = int(start_offsets[j] * 1000)

        if c["game_audio"]:
            lbl = f"[ga{j}]"
            filter_parts.append(f"[{seg_idx}:a]adelay={offset_ms}|{offset_ms},volume=0.8{lbl}")
            seg_labels.append(lbl)
            seg_idx += 1

        if c["tts"]:
            lbl = f"[tts{j}]"
            filter_parts.append(f"[{seg_idx}:a]adelay={offset_ms}|{offset_ms},volume=1.5{lbl}")
            seg_labels.append(lbl)
            t_s = start_offsets[j]
            t_e = t_s + 5.0
            duck_exprs.append(f"between(t,{t_s:.1f},{t_e:.1f})")
            seg_idx += 1

    # Mix segments
    if seg_labels:
        filter_parts.append(f"{''.join(seg_labels)}amix=inputs={len(seg_labels)}:dropout_transition=0[segs]")
    else:
        filter_parts.append(f"anullsrc=r=44100:cl=stereo[segs]")

    # Duck background during TTS
    if duck_exprs:
        duck_str = "+".join(duck_exprs)
        filter_parts.append(f"[base_bg]volume='if({duck_str},0.25,1.0)':eval=frame[bg_ducked]")
    else:
        filter_parts.append("[base_bg]acopy[bg_ducked]")

    filter_parts.append("[bg_ducked][segs]amix=inputs=2:duration=first[final_a]")

    final_cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", "0:v", "-map", "[final_a]",
        "-shortest",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        final_reel
    ])

    res = subprocess.run(final_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        logger.error(f"Audio mix failed: {res.stderr[-500:]}")
        # Fallback: just copy video without mixed audio
        fallback_cmd = ["ffmpeg", "-y", "-i", v_output, "-c", "copy", final_reel]
        subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # ── Cleanup — only remove temp audio files, KEEP the clip MP4s ─────────
    for c in clip_details:
        for p in [c["tts"], c["game_audio"]]:
            if p and os.path.exists(p):
                try: os.remove(p)
                except: pass
    trans_file = os.path.join(output_dir, f"v_trans_{match_id}.mp4")
    if os.path.exists(trans_file):
        try: os.remove(trans_file)
        except: pass

    if os.path.exists(final_reel):
        logger.info(f"✅ Highlight reel generated: {final_reel} | {len(clip_public_urls)} individual clips")
        return {
            'reel_url': f"/uploads/highlight_reel_{match_id}.mp4",
            'clip_urls': clip_public_urls,
        }

    logger.error("Highlight reel generation failed")
    return {'reel_url': None, 'clip_urls': clip_public_urls}


def precompress_video(video_path: str, match_id: str, config: dict) -> str:
    try:
        if os.path.getsize(video_path) / (1024 * 1024) < config["COMPRESS_SIZE_THRESHOLD_MB"]:
            return video_path

        compressed_path = str(Path(video_path).parent / f"compressed_{match_id}.mp4")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"scale=-2:{config['COMPRESS_OUTPUT_HEIGHT']},fps={config['COMPRESS_OUTPUT_FPS']}",
            "-c:v", "libx264", "-crf", "28", "-preset", "ultrafast", "-an",
            compressed_path
        ]
        if subprocess.run(cmd, timeout=config["FFMPEG_COMPRESS_TIMEOUT"]).returncode == 0:
            return compressed_path
    except:
        pass
    return video_path
