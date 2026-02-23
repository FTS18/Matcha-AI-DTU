import os
import logging
import cv2
import numpy as np
from typing import List, Optional, Dict
from PIL import Image

logger = logging.getLogger(__name__)

_gemini_model: Optional[object] = None
_gemini_model_initialized: bool = False
_gemini_vision_model: Optional[object] = None
_gemini_vision_model_initialized: bool = False

def _get_gemini_api_key() -> Optional[str]:
    """Lazy lookup so .env is loaded by main.py before first use."""
    return os.getenv("GEMINI_API_KEY")

def _get_gemini():
    global _gemini_model, _gemini_model_initialized
    if not _gemini_model_initialized:
        _gemini_model_initialized = True
        api_key = _get_gemini_api_key()
        if not api_key:
            logger.warning("Gemini API key not configured")
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Fix: Using correct model name gemini-2.0-flash
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini 2.0 Flash loaded ✓")
        except Exception as e:
            logger.warning(f"Gemini unavailable: {e}")
            _gemini_model = None
    return _gemini_model

def _get_gemini_vision():
    global _gemini_vision_model, _gemini_vision_model_initialized
    if not _gemini_vision_model_initialized:
        _gemini_vision_model_initialized = True
        api_key = _get_gemini_api_key()
        if not api_key:
            logger.warning("Gemini API key not configured")
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Fix: Using correct model name gemini-2.0-flash
            _gemini_vision_model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini Vision loaded ✓")
        except Exception as e:
            logger.warning(f"Gemini Vision unavailable: {e}")
            _gemini_vision_model = None
    return _gemini_vision_model

def _frame_to_pil(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def analyze_frame_with_vision(frame, timestamp: float, context: str = "") -> dict:
    gemini = _get_gemini_vision()
    if not gemini:
        return {"event_type": "NONE", "confidence": 0.0, "description": "Vision AI unavailable"}
    
    try:
        pil_image = _frame_to_pil(frame)
        prompt = """Analyze this football/soccer video frame. Determine if a significant game event is happening.
IMPORTANT: Be STRICT. Only classify as an event if you're confident it's actually happening in this frame.
Event types to look for:
- GOAL: Ball clearly entering/in the goal net, or immediate celebration after scoring
- SAVE: Goalkeeper making a save, diving, catching or deflecting the ball
- TACKLE: Clear physical challenge between players for the ball
- FOUL: Player being fouled, falling unnaturally, referee intervention
- CELEBRATION: Players clearly celebrating (arms raised, hugging, jumping)
- NONE: Regular gameplay, nothing significant, unclear, or can't determine
Respond in this exact format (one line):
EVENT_TYPE|CONFIDENCE|DESCRIPTION"""
        
        response = gemini.generate_content([prompt, pil_image])
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
                confidence = 0.3
            description = "|".join(parts[2:]).strip()
            return {"event_type": event_type, "confidence": confidence, "description": description}
        return {"event_type": "NONE", "confidence": 0.0, "description": text[:100]}
    except Exception as e:
        logger.warning(f"Vision analysis failed: {e}")
        return {"event_type": "NONE", "confidence": 0.0, "description": str(e)[:50]}

def generate_commentary(event_type, final_score, timestamp, duration, context_events=None, language="english"):
    minute   = max(1, int(timestamp / 60))
    late         = duration > 0 and (timestamp / duration) > 0.85
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
        f"CRITICAL: Do NOT invent player names, team names, shirt numbers, or scorelines. "
        f"Use only generic terms like 'the striker', 'the midfielder', 'the goalkeeper', 'the attacking team'. "
        f"Describe only what could plausibly happen during a {event_type} — do not fabricate details.\n\n"
        f"IMPORTANT: Respond ONLY in {language} language."
    )
    gemini = _get_gemini()
    if gemini:
        try:
            resp = gemini.generate_content(prompt)
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
        f"key moments, overall character.\n\n"
        f"CRITICAL: Do NOT invent player names, team names, or exact scorelines. "
        f"Refer to teams generically (e.g. 'the attacking side', 'the home team'). "
        f"Only describe events that are in the data above — do not hallucinate extra events.\n\n"
        f"IMPORTANT: Respond ONLY in {language} language."
    )
    gemini = _get_gemini()
    if gemini:
        try:
            resp = gemini.generate_content(prompt)
            text = resp.text.strip()
            if text:
                return text
        except Exception as e:
            logger.warning(f"Gemini summary error: {e}")
    return None
