import os
import logging
import cv2
import time
import numpy as np
from typing import List, Optional, Dict
from PIL import Image

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_gemini_model = None
_gemini_vision_model = None

def _get_gemini():
    global _gemini_model
    if _gemini_model is None:
        if not GEMINI_API_KEY:
            logger.warning("Gemini API key not configured")
            _gemini_model = False
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini 2.0 Flash loaded ✓")
        except Exception as e:
            logger.warning(f"Gemini unavailable: {e}")
            _gemini_model = False
    return _gemini_model if _gemini_model else None


def _call_gemini_with_retry(model, content, max_retries: int = 3):
    """Call Gemini generate_content with retry + exponential backoff for 429 errors."""
    delays = [5, 15, 30]  # seconds between retries
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(content)
            return response
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                if attempt < max_retries:
                    wait = delays[min(attempt, len(delays) - 1)]
                    logger.warning(f"Gemini rate limited (attempt {attempt+1}/{max_retries+1}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"Gemini rate limit exceeded after {max_retries+1} attempts")
                    raise
            else:
                raise

def _get_gemini_vision():
    """Vision model is the same as text model for gemini-2.0-flash."""
    return _get_gemini()

def _frame_to_pil(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def analyze_frame_with_vision(frame, timestamp: float, context: str = "") -> dict:
    gemini = _get_gemini_vision()
    if not gemini:
        return {"event_type": "NONE", "confidence": 0.0, "description": "Vision AI unavailable"}
    
    try:
        pil_image = _frame_to_pil(frame)
        prompt = """You are a professional football/soccer match analyst AI. Analyze this single frame.

CRITICAL RULES:
1. Be extremely strict. Most frames are regular gameplay — classify them as NONE.
2. GOAL: Ball CLEARLY inside the goal net, or players in unmistakable scoring celebration.
3. SAVE: Goalkeeper ACTIVELY diving, stretching, or deflecting ball. Standing near goal is NOT a save.
4. TACKLE: Two players in clear physical contact fighting for ball. Running near each other is NOT a tackle.
5. FOUL: Player falling unnaturally AND referee nearby pointing/holding card.
6. CELEBRATION: Multiple players in group hug, jumping, or sliding on knees.
7. If NOT at least 70% sure, classify as NONE.
8. Confidence must be 0.0-1.0 (e.g., 0.85). Do NOT use percentages.

Respond EXACTLY: EVENT_TYPE|CONFIDENCE|SHORT_DESCRIPTION
Example: NONE|0.1|Regular midfield passing sequence"""

        response = _call_gemini_with_retry(gemini, [prompt, pil_image])
        text = response.text.strip()
        parts = text.split("|")
        if len(parts) >= 3:
            event_type = parts[0].strip().upper()
            if event_type not in ["GOAL", "SAVE", "TACKLE", "FOUL", "CELEBRATION", "NONE"]:
                event_type = "NONE"
            try:
                confidence = float(parts[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except:
                confidence = 0.0
            # Reject low-confidence detections to reduce false positives
            if confidence < 0.65 and event_type != "NONE":
                logger.debug(f"Rejected low-confidence: {event_type} @ {confidence:.2f}")
                event_type = "NONE"
                confidence = 0.0
            description = "|".join(parts[2:]).strip()
            return {"event_type": event_type, "confidence": confidence, "description": description}
        return {"event_type": "NONE", "confidence": 0.0, "description": text[:100]}
    except Exception as e:
        logger.warning(f"Vision analysis failed: {e}")
        return {"event_type": "NONE", "confidence": 0.0, "description": str(e)[:50]}

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
            text = resp.text.strip().strip('"').strip("'")
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini commentary error: {e}")
    return None

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
            resp = _call_gemini_with_retry(gemini, prompt)
            text = resp.text.strip()
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini summary error: {e}")
    return None
