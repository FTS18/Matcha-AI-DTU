import os
import logging
import subprocess
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

from app.core.tts import tts_generate

try:
    from app.core.soccer_analysis import process_clip_frames as _sa_process, is_available as _sa_available
except ImportError:
    _sa_process = None  # type: ignore
    _sa_available = None  # type: ignore

logger = logging.getLogger(__name__)

# ── Logo path (top-right watermark) ──────────────────────────────────────────
LOGO_PATH = str(Path(__file__).resolve().parent.parent.parent.parent.parent / "apps" / "web" / "public" / "favicons" / "logo.png")

# ── Event-specific overlay configs ───────────────────────────────────────────
EVENT_CONFIG = {
    "GOAL":        {"color": (0, 255, 100),  "title": "GOAL",        "transition": "circleopen"},
    "SAVE":        {"color": (50, 180, 255), "title": "GREAT SAVE",  "transition": "fadeblack"},
    "TACKLE":      {"color": (255, 165, 0),  "title": "TACKLE",      "transition": "slideleft"},
    "FOUL":        {"color": (255, 50, 50),  "title": "FOUL",        "transition": "fadeblack"},
    "CELEBRATION": {"color": (255, 215, 0),  "title": "CELEBRATION", "transition": "circleopenclose"},
    "HIGHLIGHT":   {"color": (200, 200, 255),"title": "KEY MOMENT",  "transition": "fade"},
}

VALID_TRANSITIONS = {
    "fade", "fadeblack", "fadewhite", "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "rectcrop", "distance", "pixelize", "diagtl", "diagtr", "diagbl",
    "diagbr", "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "fadegrays",
    "wipel", "wiper", "wipet", "wipeb",
}


def _annotate_clip_with_soccer_analysis(
    clip_path: str, output_dir: str, match_id: str, clip_idx: int
) -> str:
    """
    Read a raw video clip, run the soccer-analysis overlay pipeline on its
    frames (player ellipses, speed/distance, ball control %), then write the
    annotated frames back to a new file.  Returns the path to the annotated
    clip (or the original clip path on failure).
    """
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

        annotated_path = os.path.join(
            output_dir, f"sa_v_{match_id}_{clip_idx}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))
        for f in annotated:
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h))
            writer.write(f)
        writer.release()

        # Re-encode to H.264 so ffmpeg can concat/xfade it reliably
        h264_path = os.path.join(
            output_dir, f"sa_h264_{match_id}_{clip_idx}.mp4"
        )
        re_cmd = [
            "ffmpeg", "-y", "-i", annotated_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-an", h264_path,
        ]
        if subprocess.run(re_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
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
        logger.error(f"Soccer analysis annotation failed for clip {clip_idx}: {e}", exc_info=True)
        return clip_path


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
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timed out after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"ffmpeg error: {e}")
        return False


def generate_silent_audio(output_path: str, duration: float = 10.0) -> bool:
    """Generate a silent audio file of the given duration."""
    try:
        return _run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
            "-t", str(duration),
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ], timeout=30)
    except Exception as e:
        logger.error(f"Failed to generate silent audio: {e}")
        return False


def _get_ball_focus_region(tracking_data: list, start: float, end: float) -> tuple:
    """Return median ball (x, y) in [0,1] for smart crop."""
    ball_xs, ball_ys = [], []
    for tf in tracking_data:
        t = tf.get("t", 0)
        if start <= t <= end and tf.get("b"):
            bx, by, bw, bh = tf["b"][0][:4]
            ball_xs.append(bx + bw / 2)
            ball_ys.append(by + bh / 2)
    if ball_xs:
        return float(np.median(ball_xs)), float(np.median(ball_ys))
    return 0.5, 0.5


def _build_event_overlay_filter(event_type: str, clip_dur: float, is_vertical: bool) -> str:
    """
    Build FFmpeg drawtext filter for the event title card.
    Keeps it ASCII-safe (no emoji — those crash ffmpeg drawtext).
    """
    cfg = EVENT_CONFIG.get(event_type, EVENT_CONFIG["HIGHLIGHT"])
    title = cfg["title"]          # pure ASCII, safe for drawtext
    r, g, b = cfg["color"]
    hex_color = f"{r:02x}{g:02x}{b:02x}"

    title_size = 52 if not is_vertical else 40
    # Fade in 0→0.5s, hold to 3.5s, fade out to 4s
    alpha = "if(lt(t,0.5),t/0.5,if(lt(t,3.5),1,if(lt(t,4.0),(4.0-t)/0.5,0)))"
    # NOTE: drawtext uses 'h'/'w' (not 'ih'/'iw') for input dimensions
    y_pos = "h*0.09" if not is_vertical else "h*0.065"
    bar_h = 70 if not is_vertical else 55

    filters = [
        # Dark background bar  (drawbox CAN use ih/iw)
        f"drawbox=x=0:y=ih*0.07:w=iw:h={bar_h}:color=black@0.65:t=fill:enable='between(t,0.3,4.0)'",
        # Event title centred  (drawtext uses h/w, NOT ih/iw)
        (f"drawtext=text='{title}'"
         f":fontcolor=0x{hex_color}:fontsize={title_size}"
         f":x=(w-text_w)/2:y={y_pos}"
         f":alpha='{alpha}':borderw=2:bordercolor=black@0.8"),
        # Matcha AI watermark bottom-left
        ("drawtext=text='Matcha AI'"
         ":fontcolor=white@0.45:fontsize=15"
         ":x=14:y=h-28:borderw=1:bordercolor=black@0.6"),
    ]
    return ",".join(filters)


def create_highlight_reel(
    video_path: str, highlights: list, match_id: str, output_dir: str,
    music_dir: Path, tracking_data: Optional[list] = None,
    aspect_ratio: str = "16:9", language: str = "english"
) -> dict:
    """
    Production-grade highlight reel (robust version):
    1. Extracts per-highlight MP4 clips with event title overlay + Matcha AI watermark.
    2. Smart ball-follow crop for 9:16 vertical reels.
    3. Concatenates all clips with xfade transitions.
    4. Mixes TTS commentary + background music + crowd ambience + roar sfx.
       - Original game audio is MUTED — highlights use only produced audio.
       - Music fades in/out smoothly.
       - Crowd ambience runs throughout, volume rises on events.
       - Roar SFX fires at the start of each clip for big moments.
       - TTS is always prominent, music ducks under it.
    Returns: {'reel_url': str|None, 'clip_urls': [str|None, ...]}
    """
    if not highlights:
        return {'reel_url': None, 'clip_urls': []}

    is_vertical = (aspect_ratio == "9:16")
    has_logo = os.path.exists(LOGO_PATH)
    transition_duration = 0.8
    logo_size = 50 if not is_vertical else 38
    logo_pad = 10
    ar_tag = "_p" if is_vertical else ""   # suffix to avoid filename collisions

    logger.info(f"[Reel] Generating {'9:16' if is_vertical else '16:9'} reel "
                f"for {match_id} ({len(highlights)} clips)")

    # ── Music assets ─────────────────────────────────────────────────────────
    music_path = str(music_dir / "music.mp3")
    crowd_path = str(music_dir / "crowd.mp3")
    roar_path  = str(music_dir / "roar.mp3")
    if not os.path.exists(music_path):
        generate_silent_audio(music_path, duration=300.0)
    has_crowd = os.path.exists(crowd_path)
    has_roar  = os.path.exists(roar_path)

    # ── Phase 1: Extract individual clips ────────────────────────────────────
    clip_details: list = []
    clip_public_urls: list = []

    for i, h in enumerate(highlights):
        start = float(h.get("startTime", 0))
        end   = float(h.get("endTime", start + 10))
        clip_dur = max(end - start, 1.0)
        event_type = str(h.get("eventType") or "HIGHLIGHT").upper()
        commentary = str(h.get("commentary") or "")

        v_clip  = os.path.join(output_dir, f"clip_{match_id}_{i}{ar_tag}.mp4")
        a_game  = os.path.join(output_dir, f"game_{match_id}_{i}{ar_tag}.wav")
        a_tts   = os.path.join(output_dir, f"tts_{match_id}_{i}{ar_tag}.wav")

        # ── Video filter chain ────────────────────────────────────────────
        vf_parts: list = []

        # 1) Vertical crop (9:16)
        if is_vertical:
            if tracking_data:
                fx, fy = _get_ball_focus_region(tracking_data, start, end)
                cw_norm = (9 / 16) / (16 / 9)
                x_start = max(0.0, min(1.0 - cw_norm, fx - cw_norm / 2))
                vf_parts.append(
                    f"crop=iw*{cw_norm:.4f}:ih:{x_start:.4f}*iw:0,scale=720:1280"
                )
            else:
                vf_parts.append("crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale=720:1280")

        # 2) Event overlay (ASCII-safe)
        overlay_str = _build_event_overlay_filter(event_type, clip_dur, is_vertical)
        if overlay_str:
            vf_parts.append(overlay_str)

        vf_str = ",".join(vf_parts) if vf_parts else "null"

        # ── Build extraction command ──────────────────────────────────────
        if has_logo:
            full_vf = (
                f"[0:v]{vf_str}[_vf];"
                f"[1:v]scale={logo_size}:{logo_size},format=rgba[_logo];"
                f"[_vf][_logo]overlay=W-w-{logo_pad}:{logo_pad}[vout]"
            )
            clip_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-to", str(end), "-i", video_path,
                "-i", LOGO_PATH,
                "-filter_complex", full_vf,
                "-map", "[vout]", "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-movflags", "+faststart",
                v_clip,
            ]
        else:
            clip_cmd = [
                "ffmpeg", "-y",
                "-ss", str(start), "-to", str(end), "-i", video_path,
                "-vf", vf_str, "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-movflags", "+faststart",
                v_clip,
            ]

        # ── Skip original game audio — highlights use only crowd/roar/music/TTS
        has_game_audio = False

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
            ok = _run_ffmpeg([
                "ffmpeg", "-y",
                "-ss", str(start), "-to", str(end), "-i", video_path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-an", "-movflags", "+faststart",
                v_clip,
            ], timeout=120)

        if ok and os.path.exists(v_clip) and os.path.getsize(v_clip) > 1024:
            # ── Soccer Analysis Overlay ──────────────────────────────────────
            # Annotate clip with player tracking, ellipses, speed & ball control
            if _sa_process and _sa_available and _sa_available():
                v_clip = _annotate_clip_with_soccer_analysis(v_clip, output_dir, match_id, i)
            cfg = EVENT_CONFIG.get(event_type, EVENT_CONFIG["HIGHLIGHT"])
            pub_url = f"/uploads/clip_{match_id}_{i}{ar_tag}.mp4"
            clip_public_urls.append(pub_url)
            clip_details.append({
                "video": v_clip,
                "public_url": pub_url,
                "game_audio": a_game if has_game_audio else None,
                "tts": a_tts if has_tts else None,
                "duration": clip_dur,
                "transition": cfg["transition"] if cfg["transition"] in VALID_TRANSITIONS else "fade",
            })
            logger.info(f"  ✓ clip {i}: {pub_url} ({clip_dur:.1f}s)")
        else:
            logger.warning(f"  ✗ clip {i} failed completely, skipping")
            clip_public_urls.append(None)

    if not clip_details:
        logger.error("[Reel] No clips extracted — aborting reel")
        return {'reel_url': None, 'clip_urls': clip_public_urls}

    # ── Phase 2: Stitch clips with xfade ─────────────────────────────────────
    if len(clip_details) == 1:
        v_stitched = clip_details[0]["video"]
    else:
        v_stitched = os.path.join(output_dir, f"v_trans_{match_id}{ar_tag}.mp4")
        v_inputs = []
        for c in clip_details:
            v_inputs += ["-i", c["video"]]

        fp: list = []
        last_label = "[0:v]"
        offset = clip_details[0]["duration"] - transition_duration

        for idx in range(1, len(clip_details)):
            out_label = f"[v{idx}]"
            trans = clip_details[idx]["transition"]
            fp.append(
                f"{last_label}[{idx}:v]xfade=transition={trans}"
                f":duration={transition_duration:.2f}:offset={max(offset,0):.2f}{out_label}"
            )
            last_label = out_label
            offset += clip_details[idx]["duration"] - transition_duration

        stitch_ok = _run_ffmpeg(
            ["ffmpeg", "-y"]
            + v_inputs
            + ["-filter_complex", ";".join(fp),
               "-map", last_label,
               "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-movflags", "+faststart",
               v_stitched],
            timeout=300,
        )
        if not stitch_ok:
            logger.warning("[Reel] Stitch failed — using first clip as fallback")
            v_stitched = clip_details[0]["video"]

    # ── Phase 3: Mix audio (TTS + music + crowd + roar — no game audio) ──
    ar_suffix = "_portrait" if is_vertical else ""
    final_reel = os.path.join(output_dir, f"highlight_reel_{match_id}{ar_suffix}.mp4")

    # Calculate total reel duration and per-clip start offsets
    start_offsets = [0.0]
    for j in range(len(clip_details) - 1):
        start_offsets.append(
            start_offsets[-1] + clip_details[j]["duration"] - transition_duration
        )
    total_dur = start_offsets[-1] + clip_details[-1]["duration"] if clip_details else 10.0

    # ── Collect all audio input files ─────────────────────────────────────
    # Input 0 = video, Input 1 = music (looped), Input 2 = crowd (looped)
    extra_inputs: list = []
    extra_inputs += ["-stream_loop", "-1", "-i", music_path]       # input 1 = music
    if has_crowd:
        extra_inputs += ["-stream_loop", "-1", "-i", crowd_path]   # input 2 = crowd
    roar_input_idx = (2 if has_crowd else 1) + 1   # index of roar input
    if has_roar:
        extra_inputs += ["-i", roar_path]                          # input 2 or 3 = roar

    # Next available ffmpeg input index for TTS segments
    next_idx = (1   # music
                + (1 if has_crowd else 0)
                + (1 if has_roar else 0)
                + 1)  # +1 for the video itself at index 0

    filter_parts: list = []
    mix_labels: list = []

    # ── Background music: smooth 2s fade-in, 2s fade-out, low volume ─────
    fade_out_start = max(0, total_dur - 2.0)
    filter_parts.append(
        f"[1:a]volume=0.10,afade=t=in:st=0:d=2.0,afade=t=out:st={fade_out_start:.2f}:d=2.0"
        f",atrim=0:{total_dur:.2f},apad=whole_dur={total_dur:.2f}[bgm]"
    )
    mix_labels.append("[bgm]")

    # ── Crowd ambience: looped, 1.5s fade-in, 1.5s fade-out, moderate vol
    if has_crowd:
        filter_parts.append(
            f"[2:a]volume=0.30,afade=t=in:st=0:d=1.5,afade=t=out:st={fade_out_start:.2f}:d=1.5"
            f",atrim=0:{total_dur:.2f},apad=whole_dur={total_dur:.2f}[crowd]"
        )
        mix_labels.append("[crowd]")

    # ── Split roar input so it can be used once per clip ──────────────────
    num_clips = len(clip_details)
    if has_roar and num_clips > 0:
        if num_clips == 1:
            filter_parts.append(f"[{roar_input_idx}:a]acopy[_roar0]")
        else:
            split_labels = "".join(f"[_roar{j}]" for j in range(num_clips))
            filter_parts.append(f"[{roar_input_idx}:a]asplit={num_clips}{split_labels}")

    # ── Per-clip audio: TTS commentary, roar SFX (no game audio) ────────
    for j, c in enumerate(clip_details):
        off_ms = int(start_offsets[j] * 1000)
        clip_dur = c["duration"]

        # TTS commentary — prominent, starts 0.3s into clip for natural feel
        if c["tts"] and os.path.exists(c["tts"]):
            lbl = f"[tts{j}]"
            extra_inputs += ["-i", c["tts"]]
            tts_delay = off_ms + 300  # 0.3s after clip start
            filter_parts.append(
                f"[{next_idx}:a]volume=1.6,"
                f"afade=t=in:st=0:d=0.2,afade=t=out:st={max(0,clip_dur-0.5):.2f}:d=0.5,"
                f"adelay={tts_delay}|{tts_delay}{lbl}"
            )
            mix_labels.append(lbl)
            next_idx += 1

        # Roar SFX at the start of each clip (big moment emphasis)
        if has_roar:
            lbl = f"[roar{j}]"
            roar_delay = off_ms + 200  # 0.2s into clip
            filter_parts.append(
                f"[_roar{j}]volume=0.40,"
                f"afade=t=in:st=0:d=0.15,afade=t=out:st=1.5:d=1.0,"
                f"adelay={roar_delay}|{roar_delay}{lbl}"
            )
            mix_labels.append(lbl)

    # ── Final mix: all layers together ────────────────────────────────────
    if len(mix_labels) >= 2:
        filter_parts.append(
            f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}"
            f":duration=first:dropout_transition=3:normalize=0[final_a]"
        )
    elif len(mix_labels) == 1:
        # Only one audio source — just rename it
        filter_parts.append(f"{mix_labels[0]}acopy[final_a]")
    else:
        filter_parts.append("anullsrc=r=44100:cl=stereo[final_a]")

    audio_ok = _run_ffmpeg(
        ["ffmpeg", "-y",
         "-i", v_stitched]
        + extra_inputs
        + ["-filter_complex", ";".join(filter_parts),
           "-map", "0:v", "-map", "[final_a]",
           "-shortest",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart",
           final_reel],
        timeout=300,
    )

    if not audio_ok:
        # Fallback: copy video without complex audio
        logger.warning("[Reel] Audio mix failed, copying video-only")
        _run_ffmpeg(
            ["ffmpeg", "-y", "-i", v_stitched,
             "-c:v", "copy", "-an",
             "-movflags", "+faststart", final_reel],
            timeout=120,
        )

    # ── Cleanup temp files ────────────────────────────────────────────────────
    for c in clip_details:
        p = c.get("tts")
        if p and os.path.exists(p):
            try: os.remove(p)
            except: pass
    trans_file = os.path.join(output_dir, f"v_trans_{match_id}{ar_tag}.mp4")
    if os.path.exists(trans_file) and trans_file != (clip_details[0]["video"] if clip_details else ""):
        try: os.remove(trans_file)
        except: pass

    if os.path.exists(final_reel) and os.path.getsize(final_reel) > 1024:
        reel_url = f"/uploads/highlight_reel_{match_id}{ar_suffix}.mp4"
        logger.info(f"[Reel] ✓ {final_reel} | clips={len(clip_public_urls)}")
        return {'reel_url': reel_url, 'clip_urls': clip_public_urls}

    logger.error("[Reel] Final reel not created")
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
