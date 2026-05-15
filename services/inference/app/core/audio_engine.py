import logging

logger = logging.getLogger(__name__)


def calculate_dynamic_audio_volumes(motion_score: float, emotion_score: float) -> dict:
    """
    Calculate audio volumes dynamically based on match intensity.
    Higher intensity = louder crowd, reduced music, emphasized commentary.
    """
    # Normalize inputs to 0-1 range
    motion = max(0.0, min(1.0, motion_score))
    emotion = max(0.0, min(1.0, emotion_score / 10.0))

    # Average intensity
    intensity = (motion + emotion) / 2.0

    # Dynamic volume adjustments
    volumes = {
        "music": max(0.02, 0.15 * (1.0 - intensity)),  # Fade out in intense moments
        "crowd": 0.25 + (0.35 * intensity),  # Ramp up with intensity
        "roar": 0.1 + (0.4 * intensity),  # Roar on big moments
        "commentary": 1.2 + (0.3 * intensity),  # Always prominent, boost on action
    }

    # Normalize to prevent clipping
    max_vol = max(volumes.values())
    if max_vol > 1.5:
        scale = 1.5 / max_vol
        volumes = {k: round(v * scale, 3) for k, v in volumes.items()}
    else:
        volumes = {k: round(v, 3) for k, v in volumes.items()}

    return volumes
