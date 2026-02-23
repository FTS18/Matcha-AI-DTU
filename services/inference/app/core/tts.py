import os
import logging
import asyncio
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

_KOKORO_MODEL   = "hexgrad/Kokoro-82M"
_KOKORO_VOICE   = "af_sky"
_EDGE_TTS_VOICE = "en-GB-RyanNeural"
_hf_token: Optional[str] = os.getenv("HF_TOKEN")

# Track Kokoro failures so we don't waste time retrying a dead endpoint
_kokoro_available = True

def _kokoro_tts(text: str, output_path: str) -> bool:
    global _kokoro_available
    if not _kokoro_available:
        return False
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(
            provider="hf-inference",
            api_key=_hf_token or "hf_anonymous",
        )
        audio_bytes = client.text_to_speech(text, model=_KOKORO_MODEL)
        if not audio_bytes:
            return False
        raw = audio_bytes if isinstance(audio_bytes, (bytes, bytearray)) else audio_bytes.read()
        if len(raw) < 100:
            return False
        with open(output_path, "wb") as f:
            f.write(raw)
        logger.info(f"[TTS Tier-1] Kokoro-82M generated: {output_path}")
        return True
    except Exception as e:
        err = str(e)
        # If 404/402/503 → model endpoint is down/paid, don't retry for this session
        if "404" in err or "402" in err or "503" in err or "Not Found" in err or "Payment Required" in err:
            logger.warning(f"Kokoro TTS endpoint unavailable, disabling: {e}")
            _kokoro_available = False
        else:
            logger.warning(f"Kokoro TTS failed: {e}")
        return False

_VOICE_MAP = {
    "english": "en-GB-RyanNeural",
    "spanish": "es-ES-AlvaroNeural",
    "portuguese": "pt-BR-AntonioNeural",
    "arabic": "ar-SA-HamedNeural",
}

def _edge_tts(text: str, output_path: str, language: str = "english") -> bool:
    try:
        import edge_tts
        voice = _VOICE_MAP.get(language.lower(), _EDGE_TTS_VOICE)
        async def _synth():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
        asyncio.run(_synth())
        logger.info(f"[TTS Tier-2] edge-tts generated ({language}): {output_path}")
        return True
    except Exception as e:
        logger.warning(f"edge-tts failed: {e}")
        return False

def tts_generate(text: str, output_path: str, language: str = "english") -> bool:
    # Kokoro is currently EN-only for af_sky, so we fallback to edge-tts for other languages
    if language.lower() == "english":
        if _kokoro_tts(text, output_path):
            return True
    
    if _edge_tts(text, output_path, language):
        return True
    logger.error("All TTS backends failed")
    return False

def get_tts_available() -> bool:
    try:
        import edge_tts
        return True
    except ImportError:
        return False
