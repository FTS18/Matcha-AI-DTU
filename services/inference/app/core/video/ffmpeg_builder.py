import numpy as np
import os
from .constants import EVENT_CONFIG, VALID_TRANSITIONS, LOGO_PATH


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


def _build_event_overlay_filter(
    event_type: str, clip_dur: float, is_vertical: bool
) -> str:
    """
    Build FFmpeg drawtext filter for the event title card.
    Keeps it ASCII-safe (no emoji — those crash ffmpeg drawtext).
    """
    cfg = EVENT_CONFIG.get(event_type, EVENT_CONFIG["HIGHLIGHT"])
    title = cfg["title"]  # pure ASCII, safe for drawtext
    r, g, b = cfg["color"]
    hex_color = f"{r:02x}{g:02x}{b:02x}"

    title_size = 52 if not is_vertical else 40
    # Fade in 0→0.5s, hold to 3.5s, fade out to 4s
    alpha = "if(lt(t,0.5),t/0.5,if(lt(t,3.5),1,if(lt(t,4.0),(4.0-t)/0.5,0)))"
    # NOTE: drawtext uses 'h'/'w' (not 'ih'/'iw') for input dimensions
    y_pos = "h*0.09" if not is_vertical else "h*0.065"
    bar_h = 70 if not is_vertical else 55

    filters = [
        # Dark background bar (drawbox CAN use ih/iw)
        f"drawbox=x=0:y=ih*0.07:w=iw:h={bar_h}:color=black@0.65:t=fill:enable='between(t,0.3,4.0)'",
        # Event title centred (drawtext uses h/w, NOT ih/iw)
        (
            f"drawtext=text='{title}'"
            f":fontcolor=0x{hex_color}:fontsize={title_size}"
            f":x=(w-text_w)/2:y={y_pos}"
            f":alpha='{alpha}':borderw=2:bordercolor=black@0.8"
        ),
        # Matcha AI watermark bottom-left
        (
            "drawtext=text='Matcha AI'"
            ":fontcolor=white@0.45:fontsize=15"
            ":x=14:y=h-28:borderw=1:bordercolor=black@0.6"
        ),
    ]
    return ",".join(filters)


def build_clip_extraction_command(
    video_path: str,
    start: float,
    end: float,
    event_type: str,
    output_path: str,
    is_vertical: bool = False,
    tracking_data: list = None,
    logo_path: str = LOGO_PATH,
) -> list:
    """Builds the FFmpeg command for extracting a single clip with overlays."""
    clip_dur = max(end - start, 1.0)
    vf_parts = []

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

    # 2) Event overlay
    overlay_str = _build_event_overlay_filter(event_type, clip_dur, is_vertical)
    if overlay_str:
        vf_parts.append(overlay_str)

    vf_str = ",".join(vf_parts) if vf_parts else "null"

    has_logo = os.path.exists(logo_path)
    logo_size = 50 if not is_vertical else 38
    logo_pad = 10

    if has_logo:
        full_vf = (
            f"[0:v]{vf_str}[_vf];"
            f"[1:v]scale={logo_size}:{logo_size},format=rgba[_logo];"
            f"[_vf][_logo]overlay=W-w-{logo_pad}:{logo_pad}[vout]"
        )
        return [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            video_path,
            "-i",
            logo_path,
            "-filter_complex",
            full_vf,
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-movflags",
            "+faststart",
            output_path,
        ]
    else:
        return [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-to",
            str(end),
            "-i",
            video_path,
            "-vf",
            vf_str,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-movflags",
            "+faststart",
            output_path,
        ]


def build_stitch_command(
    clips: list, output_path: str, transition_duration: float = 0.8
) -> list:
    """Builds the FFmpeg command for stitching clips with transitions."""
    if len(clips) < 2:
        return []

    v_inputs = []
    for c in clips:
        v_inputs += ["-i", c["video"]]

    fp = []
    last_label = "[0:v]"
    offset = clips[0]["duration"] - transition_duration

    for idx in range(1, len(clips)):
        out_label = f"[v{idx}]"
        trans = clips[idx].get("transition", "fade")
        if trans not in VALID_TRANSITIONS:
            trans = "fade"

        fp.append(
            f"{last_label}[{idx}:v]xfade=transition={trans}"
            f":duration={transition_duration:.2f}:offset={max(offset,0):.2f}{out_label}"
        )
        last_label = out_label
        offset += clips[idx]["duration"] - transition_duration

    return (
        ["ffmpeg", "-y"]
        + v_inputs
        + [
            "-filter_complex",
            ";".join(fp),
            "-map",
            last_label,
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )


def build_audio_mix_command(
    v_stitched: str,
    music_path: str,
    crowd_path: str,
    roar_path: str,
    clips: list,
    output_path: str,
    total_dur: float,
    start_offsets: list,
    has_crowd: bool,
    has_roar: bool,
) -> list:
    """Builds the complex FFmpeg command for mixing all audio layers."""
    extra_inputs = []
    extra_inputs += ["-stream_loop", "-1", "-i", music_path]
    if has_crowd:
        extra_inputs += ["-stream_loop", "-1", "-i", crowd_path]

    roar_input_idx = (2 if has_crowd else 1) + 1
    if has_roar:
        extra_inputs += ["-i", roar_path]

    next_idx = 1 + (1 if has_crowd else 0) + (1 if has_roar else 0) + 1

    filter_parts = []
    mix_labels = []

    # Music
    fade_out_start = max(0, total_dur - 2.0)
    filter_parts.append(
        f"[1:a]volume=0.10,afade=t=in:st=0:d=2.0,afade=t=out:st={fade_out_start:.2f}:d=2.0"
        f",atrim=0:{total_dur:.2f},apad=whole_dur={total_dur:.2f}[bgm]"
    )
    mix_labels.append("[bgm]")

    # Crowd
    if has_crowd:
        filter_parts.append(
            f"[2:a]volume=0.30,afade=t=in:st=0:d=1.5,afade=t=out:st={fade_out_start:.2f}:d=1.5"
            f",atrim=0:{total_dur:.2f},apad=whole_dur={total_dur:.2f}[crowd]"
        )
        mix_labels.append("[crowd]")

    # Roar & TTS
    num_clips = len(clips)
    if has_roar and num_clips > 0:
        if num_clips == 1:
            filter_parts.append(f"[{roar_input_idx}:a]acopy[_roar0]")
        else:
            split_labels = "".join(f"[_roar{j}]" for j in range(num_clips))
            filter_parts.append(f"[{roar_input_idx}:a]asplit={num_clips}{split_labels}")

        for j, c in enumerate(clips):
            off_ms = int(start_offsets[j] * 1000)
            clip_dur = c["duration"]

            if c.get("tts") and os.path.exists(c["tts"]):
                lbl = f"[tts{j}]"
                extra_inputs += ["-i", c["tts"]]
                tts_delay = off_ms + 300
                filter_parts.append(
                    f"[{next_idx}:a]volume=1.6,"
                    f"afade=t=in:st=0:d=0.2,afade=t=out:st={max(0,clip_dur-0.5):.2f}:d=0.5,"
                    f"adelay={tts_delay}|{tts_delay}{lbl}"
                )
                mix_labels.append(lbl)
                next_idx += 1

            if has_roar:
                lbl = f"[roar{j}]"
                roar_delay = off_ms + 200
                filter_parts.append(
                    f"[_roar{j}]volume=0.40,"
                    f"afade=t=in:st=0:d=0.15,afade=t=out:st=1.5:d=1.0,"
                    f"adelay={roar_delay}|{roar_delay}{lbl}"
                )
                mix_labels.append(lbl)

    if len(mix_labels) >= 2:
        filter_parts.append(
            f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}"
            f":duration=first:dropout_transition=3:normalize=0[final_a]"
        )
    elif len(mix_labels) == 1:
        filter_parts.append(f"{mix_labels[0]}acopy[final_a]")
    else:
        filter_parts.append("anullsrc=r=44100:cl=stereo[final_a]")

    return (
        ["ffmpeg", "-y", "-i", v_stitched]
        + extra_inputs
        + [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "0:v",
            "-map",
            "[final_a]",
            "-shortest",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )
