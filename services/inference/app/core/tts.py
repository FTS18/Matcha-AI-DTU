import os
import logging
import asyncio
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ── Kokoro local pipeline singleton ──────────────────────────────────────────
# Loaded once, reused for every TTS call — avoids the 327 MB reload overhead.
_kokoro_pipeline  = None          # KPipeline instance, False (disabled), or None (not yet loaded)
_kokoro_lock      = threading.Lock()
_kokoro_available = True          # flipped to False on hard failure

# Voice map: Kokoro (lang_code, voice) per language
_KOKORO_VOICES = {
    "english":    ("a", "af_sky"),
    "spanish":    ("e", "ef_dora"),
    "portuguese": ("p", "pf_dora"),
    "french":     ("f", "ff_siwis"),
}

# edge-tts fallback voice map
_EDGE_VOICES = {
    "english":    "en-GB-RyanNeural",
    "spanish":    "es-ES-AlvaroNeural",
    "portuguese": "pt-BR-AntonioNeural",
    "arabic":     "ar-SA-HamedNeural",
    "french":     "fr-FR-HenriNeural",
}


def _get_kokoro_pipeline(lang_code: str = "a"):
    """Return the shared KPipeline, loading it on first use (once per process)."""
    global _kokoro_pipeline, _kokoro_available
    if not _kokoro_available:
        return None
    with _kokoro_lock:
        if _kokoro_pipeline is not None and _kokoro_pipeline is not False:
            return _kokoro_pipeline
        if _kokoro_pipeline is False:
            return None
        try:
            import warnings
            from kokoro import KPipeline
            logger.info("Loading Kokoro-82M model (first use)…")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress PyTorch weight_norm deprecation
                _kokoro_pipeline = KPipeline(lang_code=lang_code)
            logger.info("Kokoro-82M pipeline ready ✓")
            return _kokoro_pipeline
        except Exception as e:
            logger.warning(f"Kokoro failed to load, disabling: {e}")
            _kokoro_pipeline = False
            _kokoro_available = False
            return None


def _kokoro_tts(text: str, output_path: str, language: str = "english") -> bool:
    """Generate speech with local Kokoro-82M. Returns True on success."""
    global _kokoro_pipeline, _kokoro_available
    if not _kokoro_available:
        return False
    try:
        import numpy as np
        import soundfile as sf

        lang_code, voice = _KOKORO_VOICES.get(language.lower(), ("a", "af_sky"))
        pipe = _get_kokoro_pipeline(lang_code)
        if pipe is None:
            return False

        frames = []
        gen = pipe(text, voice=voice)  # type: ignore[operator]
        for _, _, audio in gen:
            if audio is not None and len(audio) > 0:
                frames.append(audio)

        if not frames:
            logger.warning("Kokoro returned no audio frames")
            return False

        audio_data = np.concatenate(frames)
        sf.write(output_path, audio_data, 24000)

        size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        if size < 200:
            logger.warning(f"Kokoro output too small ({size} bytes)")
            return False

        logger.info(f"[TTS Kokoro] ✓ {output_path} ({size // 1024} KB, {language})")
        return True

    except Exception as e:
        err = str(e)
        logger.warning(f"Kokoro TTS failed: {err[:120]}")
        if any(k in err for k in ["CUDA out", "RuntimeError", "No module named"]):
            logger.warning("Kokoro disabled for this session due to hard failure")
            _kokoro_available = False
            _kokoro_pipeline = False
        return False


def _edge_tts(text: str, output_path: str, language: str = "english") -> bool:
    """Generate speech with Microsoft edge-tts (free, online). Fallback tier."""
    try:
        import edge_tts
        voice = _EDGE_VOICES.get(language.lower(), "en-GB-RyanNeural")

        async def _synth():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(asyncio.run, _synth()).result(timeout=30)
            else:
                loop.run_until_complete(_synth())
        except RuntimeError:
            asyncio.run(_synth())

        size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        if size < 200:
            return False
        logger.info(f"[TTS edge-tts] ✓ {output_path} ({language})")
        return True
    except Exception as e:
        logger.warning(f"edge-tts failed: {e}")
        return False


def tts_generate(text: str, output_path: str, language: str = "english") -> bool:
    """
    Generate TTS audio. Priority:
      1. Kokoro-82M (local, high quality, fast after first load)
      2. edge-tts   (online, Microsoft Neural, fallback)
    """
    if not text or not text.strip():
        return False

    if _kokoro_tts(text, output_path, language):
        return True

    if _edge_tts(text, output_path, language):
        return True

    logger.error(f"All TTS backends failed for: {text[:60]}")
    return False


def get_tts_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        from kokoro import KPipeline  # noqa: F401
        return True
    except ImportError:
        pass
    return False
