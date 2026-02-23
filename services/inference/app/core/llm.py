import os
import io
import logging
import cv2
import time
import threading
import hashlib
import numpy as np
from typing import List, Optional, Dict, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Model selection ───────────────────────────────────────────────────────────
# gemini-2.0-flash                →  1500 RPD free tier — best for heavy workloads
# gemini-2.0-flash-lite           →  absolute fastest / lowest quota cost
# gemini-2.5-flash                →  highest quality but ONLY 20 RPD free tier!
_PREFERRED_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]

_gemini_model = None
_model_lock = threading.Lock()

# ── Quota circuit breaker ─────────────────────────────────────────────────────
# Once a *daily* quota error is detected, skip ALL Gemini calls for this process
# lifetime to avoid wasting minutes on doomed retries.
_quota_exhausted = False

# ── Rate-limiter: max N concurrent Gemini calls at once ───────────────────────
# Free tier is ~10-15 RPM; 3 concurrent + spacing keeps us well inside that
_GEMINI_SEMAPHORE = threading.Semaphore(3)

# ── Frame-level response cache (avoids re-calling identical crops) ────────────
_VISION_CACHE: Dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()
_MAX_CACHE = 256

def _frame_hash(frame: np.ndarray) -> str:
    """Fast perceptual hash: 16×9 grayscale thumb → MD5."""
    thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 9))
    return hashlib.md5(thumb.tobytes()).hexdigest()

def _cache_get(key: str) -> Optional[dict]:
    with _CACHE_LOCK:
        return _VISION_CACHE.get(key)

def _cache_put(key: str, val: dict):
    with _CACHE_LOCK:
        if len(_VISION_CACHE) >= _MAX_CACHE:
            # evict oldest quarter
            for k in list(_VISION_CACHE.keys())[:_MAX_CACHE // 4]:
                del _VISION_CACHE[k]
        _VISION_CACHE[key] = val


def _get_gemini():
    global _gemini_model
    if _quota_exhausted:
        return None
    with _model_lock:
        if _gemini_model is not None:
            return _gemini_model if _gemini_model else None
        if not GEMINI_API_KEY:
            logger.warning("Gemini API key not configured")
            _gemini_model = False
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            for model_name in _PREFERRED_MODELS:
                try:
                    _gemini_model = genai.GenerativeModel(model_name)
                    # Quick probe — list_models is cheap, no quota consumed
                    logger.info(f"Gemini loaded: {model_name} ✓")
                    return _gemini_model
                except Exception:
                    continue
            logger.warning("No Gemini model available")
            _gemini_model = False
            return None
        except Exception as e:
            logger.warning(f"Gemini unavailable: {e}")
            _gemini_model = False
            return None


def _call_gemini_with_retry(model, content, max_retries: int = 2, timeout: float = 30.0):
    """
    Call Gemini with:
    - Semaphore to cap concurrent calls (avoids burst 429s)
    - Exponential backoff on 429 / quota errors
    - Jitter to de-sync parallel workers
    - Overall per-call timeout to prevent pipeline from hanging
    """
    import random
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    global _quota_exhausted
    if _quota_exhausted:
        raise Exception("Gemini daily quota exhausted — skipping")
    base_delays = [2, 5, 10]
    with _GEMINI_SEMAPHORE:
        for attempt in range(max_retries + 1):
            try:
                # Wrap the blocking Gemini call in a thread with a timeout
                with ThreadPoolExecutor(max_workers=1) as _tp:
                    future = _tp.submit(model.generate_content, content)
                    response = future.result(timeout=timeout)
                return response
            except FuturesTimeout:
                logger.warning(f"Gemini call timed out after {timeout}s (attempt {attempt+1}/{max_retries+1})")
                if attempt < max_retries:
                    continue
                raise TimeoutError(f"Gemini call timed out after {max_retries+1} attempts")
            except Exception as e:
                err = str(e)
                # If model is not found (deprecated), reset cached model so next call tries fallback
                is_not_found = "404" in err or "not found" in err.lower() or "not supported" in err.lower()
                if is_not_found:
                    logger.warning(f"Gemini model not found/deprecated, resetting model cache: {err[:120]}")
                    global _gemini_model
                    with _model_lock:
                        _gemini_model = None  # force re-probe on next _get_gemini() call
                    raise
                is_rate = "429" in err or "quota" in err.lower() or "rate" in err.lower() or "resource_exhausted" in err.lower()
                if is_rate:
                    # Check if this is a DAILY quota (not just per-minute)
                    is_daily = "PerDay" in err or "limit: 0" in err
                    if is_daily:
                        logger.warning("🚫 Gemini DAILY quota exhausted — disabling Gemini for this session")
                        _quota_exhausted = True
                        raise
                    if attempt < max_retries:
                        wait = base_delays[min(attempt, len(base_delays) - 1)]
                        wait += random.uniform(0, wait * 0.3)   # jitter
                        logger.warning(f"Gemini rate limited (attempt {attempt+1}/{max_retries+1}), retrying in {wait:.1f}s…")
                        time.sleep(wait)
                        continue
                raise


def _get_gemini_vision():
    """Vision model is the same as text model for flash variants."""
    return _get_gemini()


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _pil_to_bytes(img: Image.Image, quality: int = 70) -> bytes:
    """JPEG-compress PIL image to cut token cost ~4-5x vs PNG."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


_VISION_PROMPT = (
    "You are a professional football/soccer match analyst AI.\n"
    "I will show you one or more numbered frames from a match. "
    "For EACH frame respond on its own line:\n"
    "  <N>|EVENT_TYPE|CONFIDENCE|SHORT_DESCRIPTION\n\n"
    "Rules (apply to every frame):\n"
    "- EVENT_TYPE: GOAL, SAVE, TACKLE, FOUL, CELEBRATION, or NONE\n"
    "- GOAL: ball CLEARLY inside the goal net OR players in unmistakable scoring celebration\n"
    "- SAVE: goalkeeper ACTIVELY diving/stretching/deflecting — standing near goal is NOT a save\n"
    "- TACKLE: two players in clear physical contact fighting for the ball\n"
    "- FOUL: player falling unnaturally AND referee pointing/holding a card\n"
    "- CELEBRATION: multiple players in group hug, jumping, or sliding on knees\n"
    "- If less than 70% sure → NONE\n"
    "- CONFIDENCE: 0.0-1.0 decimal (not percent)\n\n"
    "Example for 2 frames:\n"
    "1|NONE|0.1|Regular midfield passing\n"
    "2|GOAL|0.92|Ball clearly in back of net, players celebrating"
)


def _parse_vision_line(line: str) -> dict:
    """Parse: N|TYPE|CONF|DESC → dict."""
    _NONE = {"event_type": "NONE", "confidence": 0.0, "description": ""}
    parts = line.split("|")
    if len(parts) < 4:
        return _NONE
    event_type = parts[1].strip().upper()
    if event_type not in {"GOAL", "SAVE", "TACKLE", "FOUL", "CELEBRATION", "NONE"}:
        event_type = "NONE"
    try:
        conf = max(0.0, min(1.0, float(parts[2].strip())))
    except ValueError:
        conf = 0.0
    if conf < 0.65 and event_type != "NONE":
        event_type = "NONE"
        conf = 0.0
    return {"event_type": event_type, "confidence": conf, "description": "|".join(parts[3:]).strip()}


def analyze_frames_batch(frames_with_ts: List[Tuple[np.ndarray, float]]) -> List[dict]:
    """
    Send up to N frames in ONE Gemini request instead of N separate calls.
    Falls back to NONE for any frame that fails to parse.
    Cache avoids re-sending identical frames.
    """
    if not frames_with_ts:
        return []

    _none = {"event_type": "NONE", "confidence": 0.0, "description": "Vision AI unavailable"}
    gemini = _get_gemini_vision()
    if not gemini:
        return [_none.copy() for _ in frames_with_ts]

    keys = [_frame_hash(f) for f, _ in frames_with_ts]
    results: List[Optional[dict]] = [None] * len(frames_with_ts)
    uncached: List[int] = []

    for i, key in enumerate(keys):
        cached = _cache_get(key)
        if cached is not None:
            results[i] = cached
        else:
            uncached.append(i)

    if not uncached:
        logger.debug(f"Vision cache: all {len(frames_with_ts)} frames served from cache")
        return results  # type: ignore

    # Build multi-frame content list for one API call
    content: list = [_VISION_PROMPT]
    for seq, i in enumerate(uncached, 1):
        frame, _ = frames_with_ts[i]
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        img_bytes = _pil_to_bytes(_frame_to_pil(frame))
        content.append(f"Frame {seq}:")
        content.append({"mime_type": "image/jpeg", "data": img_bytes})

    try:
        response = _call_gemini_with_retry(gemini, content)
        raw_text = response.text if response else ""
        text = (raw_text or "").strip()
        parsed: Dict[int, dict] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            try:
                seq = int(line.split("|")[0].strip())
                parsed[seq] = _parse_vision_line(line)
            except (ValueError, IndexError):
                continue

        for pos, i in enumerate(uncached, 1):
            res = parsed.get(pos, {"event_type": "NONE", "confidence": 0.0, "description": "parse-error"})
            results[i] = res
            _cache_put(keys[i], res)

    except Exception as e:
        logger.warning(f"Batch vision call failed: {e}")
        for i in uncached:
            results[i] = {"event_type": "NONE", "confidence": 0.0, "description": str(e)[:60]}

    return results  # type: ignore


def analyze_frame_with_vision(frame: np.ndarray, timestamp: float, context: str = "") -> dict:
    """Single-frame wrapper — delegates to batch path (1-item batch)."""
    return analyze_frames_batch([(frame, timestamp)])[0]

def generate_commentary(event_type, final_score, timestamp, duration, context_events=None, language="english"):
    minute   = max(1, int(timestamp / 60))
    late     = duration > 0 and (timestamp / duration) > 0.85
    energy   = "HIGH INTENSITY" if final_score >= 7.5 else ("MODERATE" if final_score >= 5 else "low key")
    late_str = "in the dying minutes (CRUCIAL late-game moment!)" if late else f"at minute {minute}"
    ctx_str  = ""
    if context_events:
        near = [e for e in context_events if abs(e["timestamp"] - timestamp) < 60]
        if near:
            ctx_str = " Nearby events: " + ", ".join(e["type"] for e in near[:3]) + "."

    prompt = (
        f"You are an incredibly passionate and energetic football/soccer commentator. "
        f"Write a thrilling, continuous commentary script (around 40-60 words) describing a {event_type} "
        f"{late_str}. The intensity is {energy} (score {final_score:.1f}/10).{ctx_str} "
        f"Make it sound like a live broadcast, building up excitement, describing the build-up, the moment itself, "
        f"and the immediate aftermath. No quotes, no attribution, just the spoken words.\n\n"
        f"IMPORTANT: Respond ONLY in {language} language."
    )
    gemini = _get_gemini()
    if gemini:
        try:
            resp = _call_gemini_with_retry(gemini, prompt)
            raw_text = resp.text if resp else ""
            text = (raw_text or "").strip().strip('"').strip("'")
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini commentary error: {e}")

    # ── Fallback: generate a simple template commentary so pipeline never blocks ──
    return _fallback_commentary(event_type, minute, late)


def _fallback_commentary(event_type: str, minute: int, late: bool) -> str:
    """Template-based commentary when Gemini is unavailable."""
    _TEMPLATES = {
        "GOAL":        "And it's a GOAL! Minute {m}! The ball hits the back of the net and the crowd erupts!",
        "SAVE":        "What a SAVE at minute {m}! The keeper stretches to deny what looked like a certain goal!",
        "TACKLE":      "Crunching tackle at minute {m}! A perfectly timed challenge wins the ball back!",
        "FOUL":        "The referee blows the whistle at minute {m}! That's a foul and the free kick is awarded!",
        "CELEBRATION": "The players are celebrating at minute {m}! Pure joy on the pitch!",
    }
    tmpl = _TEMPLATES.get(event_type, "An exciting moment unfolds at minute {m}!")
    text = tmpl.format(m=minute)
    if late:
        text += " A crucial moment in the dying minutes of the match!"
    return text


def generate_commentary_parallel(scored_events: list, duration: float, language: str = "english",
                                  max_workers: int = 3) -> list:
    """
    Generate commentary for ALL events in parallel via a thread pool.
    Workers respect _GEMINI_SEMAPHORE so we never burst the quota.
    Returns the same list with 'commentary' keys filled in-place.
    Has a 60s overall timeout to prevent pipeline from hanging.
    """
    if not scored_events:
        return scored_events

    def _one(args):
        i, ev = args
        ctx = scored_events[max(0, i - 3):i] + scored_events[i + 1:i + 3]
        return i, generate_commentary(ev["type"], ev["finalScore"], ev["timestamp"], duration, ctx, language)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_one, (i, ev)): i for i, ev in enumerate(scored_events)}
        for fut in as_completed(futures, timeout=60):
            try:
                idx, commentary = fut.result(timeout=30)
                scored_events[idx]["commentary"] = commentary
            except Exception as e:
                logger.warning(f"Commentary worker failed: {e}")

    return scored_events

def generate_match_summary(scored_events, highlights, duration, language="english"):
    if not scored_events:
        return "No significant events were detected in this match footage."

    by_type: dict = {}
    for e in scored_events:
        by_type[e["type"]] = by_type.get(e["type"], 0) + 1

    top5  = sorted(scored_events, key=lambda x: x["finalScore"], reverse=True)[:5]
    tdesc = "; ".join(f"{e['type']} at {int(e['timestamp']//60)}:{int(e['timestamp']%60):02d} (score {e['finalScore']:.1f})" for e in top5)
    stats = ", ".join(f"{v} {k.lower()}s" for k, v in by_type.items())
    dur_m = int(duration // 60)

    prompt = (
        f"You are a football analyst AI. Write a 3-5 sentence match summary for a {dur_m}-minute match.\n"
        f"Event breakdown: {stats}.\n"
        f"Top 5 moments: {tdesc}.\n"
        f"Total events: {len(scored_events)} | Highlights: {len(highlights)}.\n"
        f"Use present tense, analytical but engaging. Describe match narrative — intense phases, "
        f"key moments, overall character. Don't invent player names or exact scorelines.\n\n"
        f"IMPORTANT: Respond ONLY in {language} language."
    )
    gemini = _get_gemini()
    if gemini:
        try:
            resp = _call_gemini_with_retry(gemini, prompt, max_retries=2, timeout=20)
            text = resp.text.strip()  # type: ignore[union-attr]
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini summary error: {e}")
    return f"A {dur_m}-minute match with {len(scored_events)} key events and {len(highlights)} highlights detected."
