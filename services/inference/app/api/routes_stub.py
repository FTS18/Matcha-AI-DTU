from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging
import random
import os

logger = logging.getLogger(__name__)
router = APIRouter()

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:4000/api/v1")

# Try to import the real Gemini analyzer
try:
    from app.core.gemini_analyzer import analyze_video_with_gemini
    GEMINI_AVAILABLE = True
    logger.info("Gemini video analyzer loaded ✓ — real video analysis enabled")
except ImportError as e:
    GEMINI_AVAILABLE = False
    logger.warning(f"Gemini analyzer not available: {e} — using mock fallback")


class VideoAnalysisRequest(BaseModel):
    match_id: str
    video_url: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    language: str = "english"
    aspect_ratio: str = "16:9"


def _generate_emotion_scores(duration: float, interval: float = 5.0):
    """Generate realistic emotion/intensity scores across the match timeline."""
    scores = []
    t = 0.0
    base_audio = 0.3
    base_motion = 0.25
    while t <= duration:
        # Create natural-looking peaks and valleys
        noise_a = random.uniform(-0.15, 0.15)
        noise_m = random.uniform(-0.15, 0.15)
        # Add excitement peaks near goal/event moments
        audio = max(0.0, min(1.0, base_audio + noise_a + 0.2 * abs(random.gauss(0, 0.3))))
        motion = max(0.0, min(1.0, base_motion + noise_m + 0.2 * abs(random.gauss(0, 0.3))))
        context_weight = max(0.0, min(1.0, 0.5 + random.uniform(-0.2, 0.2)))
        final = max(0.0, min(10.0, (audio * 4 + motion * 4 + context_weight * 2)))
        scores.append({
            "timestamp": round(t, 1),
            "audioScore": round(audio, 3),
            "motionScore": round(motion, 3),
            "contextWeight": round(context_weight, 3),
            "finalScore": round(final, 2),
        })
        t += interval
    return scores


async def analyze_video_stub(
    video_url: str,
    match_id: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    language: str = "english",
    aspect_ratio: str = "16:9"
):
    """Stub analysis — sends realistic mock events, highlights, emotion scores, and duration to orchestrator."""
    logger.info(f"[Stub] Starting analysis for match {match_id}")

    # Simulated match duration (90-120 seconds for uploaded clips)
    duration = round(random.uniform(85.0, 130.0), 1)

    # ── Rich mock events with finalScore ──
    mock_events = [
        {
            "timestamp": round(duration * 0.08, 1),
            "type": "TACKLE",
            "confidence": 0.89,
            "finalScore": 6.2,
            "commentary": "Strong defensive tackle in the opening minutes. The defender reads the play perfectly and dispossesses the attacker."
        },
        {
            "timestamp": round(duration * 0.18, 1),
            "type": "FOUL",
            "confidence": 0.84,
            "finalScore": 4.5,
            "commentary": "A cynical foul to stop a promising counter-attack. The referee shows a yellow card."
        },
        {
            "timestamp": round(duration * 0.30, 1),
            "type": "SAVE",
            "confidence": 0.91,
            "finalScore": 7.8,
            "commentary": "Outstanding reflex save! The goalkeeper dives to his right to tip the curling shot around the post."
        },
        {
            "timestamp": round(duration * 0.45, 1),
            "type": "GOAL",
            "confidence": 0.97,
            "finalScore": 9.5,
            "commentary": "GOAL! A beautifully crafted team move finished with a clinical strike into the bottom corner. The crowd erupts!"
        },
        {
            "timestamp": round(duration * 0.55, 1),
            "type": "TACKLE",
            "confidence": 0.86,
            "finalScore": 5.8,
            "commentary": "Sliding tackle at the edge of the box. Perfectly timed, winning the ball cleanly."
        },
        {
            "timestamp": round(duration * 0.65, 1),
            "type": "FOUL",
            "confidence": 0.82,
            "finalScore": 5.1,
            "commentary": "Late challenge from behind. Free kick in a dangerous position for the attacking team."
        },
        {
            "timestamp": round(duration * 0.78, 1),
            "type": "SAVE",
            "confidence": 0.93,
            "finalScore": 8.2,
            "commentary": "Incredible double save! The keeper blocks the initial header and then scrambles to deny the follow-up."
        },
        {
            "timestamp": round(duration * 0.88, 1),
            "type": "GOAL",
            "confidence": 0.96,
            "finalScore": 9.1,
            "commentary": "GOAL! A thunderous long-range strike that swerves past the wall and into the top corner. Unstoppable!"
        },
    ]

    # ── Highlights derived from top-scoring events ──
    mock_highlights = []
    for ev in sorted(mock_events, key=lambda e: e["finalScore"], reverse=True)[:4]:
        mock_highlights.append({
            "startTime": max(0, ev["timestamp"] - 3.0),
            "endTime": min(duration, ev["timestamp"] + 5.0),
            "score": ev["finalScore"],
            "eventType": ev["type"],
            "commentary": ev["commentary"],
            "videoUrl": None,
        })
    mock_highlights.sort(key=lambda h: h["startTime"])

    # ── Emotion / Intensity scores (audio + motion) ──
    emotion_scores = _generate_emotion_scores(duration)

    # Spike intensity around event timestamps for realism
    for ev in mock_events:
        for sc in emotion_scores:
            if abs(sc["timestamp"] - ev["timestamp"]) < 8.0:
                boost = 0.3 if ev["type"] == "GOAL" else 0.15
                sc["audioScore"] = min(1.0, round(sc["audioScore"] + boost, 3))
                sc["motionScore"] = min(1.0, round(sc["motionScore"] + boost * 0.8, 3))
                sc["finalScore"] = min(10.0, round(
                    sc["audioScore"] * 4 + sc["motionScore"] * 4 + sc["contextWeight"] * 2, 2
                ))

    # ── AI Summary ──
    summary = (
        "Match Analysis Summary\n\n"
        f"Duration: {int(duration)}s | Events Detected: {len(mock_events)} | Highlights: {len(mock_highlights)}\n\n"
        "Tactical Overview:\n"
        "The match featured an intense display of attacking football with two well-taken goals. "
        "Defensive resilience was evident through several crucial tackles and outstanding goalkeeping saves. "
        "The first half saw measured build-up play with a clinical finish, while the second half opened up "
        "with end-to-end action culminating in a spectacular long-range strike.\n\n"
        "Key Observations:\n"
        "- High pressing game from both sides creating multiple turnovers\n"
        "- Goalkeeper performance was exceptional with multiple high-quality saves\n"
        "- Defensive discipline maintained despite attacking pressure\n"
        "- Set pieces created dangerous opportunities throughout the match"
    )

    try:
        import requests as req_lib
        orchestrator_url = ORCHESTRATOR_URL

        # Phase 1: Signal processing started (20%)
        await asyncio.sleep(0.5)
        req_lib.post(f"{orchestrator_url}/matches/{match_id}/progress", json={"progress": 10}, timeout=5)
        logger.info(f"[Stub] Progress 10% for {match_id}")

        # Phase 2: Send live events one by one (20-80%)
        for i, event in enumerate(mock_events):
            progress = 20 + int((i / len(mock_events)) * 60)
            req_lib.post(f"{orchestrator_url}/matches/{match_id}/progress", json={"progress": progress}, timeout=5)
            req_lib.post(
                f"{orchestrator_url}/matches/{match_id}/live-event",
                json=event,
                timeout=5,
            )
            logger.info(f"[Stub] Sent event {event['type']} @ {event['timestamp']}s ({progress}%)")
            await asyncio.sleep(0.4)

        # Phase 3: Signal near-completion (90%)
        req_lib.post(f"{orchestrator_url}/matches/{match_id}/progress", json={"progress": 90}, timeout=5)
        await asyncio.sleep(0.3)

        # Phase 4: Send complete payload with ALL fields
        complete_payload = {
            "events": mock_events,
            "highlights": mock_highlights,
            "emotionScores": emotion_scores,
            "duration": duration,
            "summary": summary,
            "teamColors": [[220, 50, 50], [50, 100, 220]],
            "topSpeedKmh": round(random.uniform(28.0, 36.0), 1),
            "heatmapUrl": None,
            "thumbnailUrl": None,
            "highlightReelUrl": None,
            "trackingData": None,
            "videoUrl": None,
        }

        resp = req_lib.post(
            f"{orchestrator_url}/matches/{match_id}/complete",
            json=complete_payload,
            timeout=15,
        )
        logger.info(f"[Stub] Complete payload sent for {match_id} — status {resp.status_code}")
        logger.info(f"[Stub] Analysis complete: {len(mock_events)} events, {len(mock_highlights)} highlights, {len(emotion_scores)} emotion samples, duration={duration}s")

    except Exception as e:
        logger.error(f"[Stub] Analysis failed for {match_id}: {e}", exc_info=True)
        try:
            import requests as req_lib
            req_lib.post(f"{ORCHESTRATOR_URL}/matches/{match_id}/progress", json={"progress": -1}, timeout=5)
        except Exception:
            pass


@router.post("/analyze")
async def analyze_match(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze a video — uses Gemini AI for real detection, falls back to mock if unavailable."""
    logger.info(f"Received analysis request for match {request.match_id} (video: {request.video_url})")

    if GEMINI_AVAILABLE:
        logger.info(f"Using Gemini AI for real video analysis (match {request.match_id})")
        background_tasks.add_task(
            analyze_video_with_gemini,
            request.video_url,
            request.match_id,
            start_time=request.start_time,
            end_time=request.end_time,
            language=request.language,
        )
    else:
        logger.warning(f"Gemini unavailable — using mock analysis for match {request.match_id}")
        background_tasks.add_task(
            analyze_video_stub,
            request.video_url,
            request.match_id,
            start_time=request.start_time,
            end_time=request.end_time,
            language=request.language,
            aspect_ratio=request.aspect_ratio,
        )

    return {"status": "processing", "match_id": request.match_id, "mode": "gemini" if GEMINI_AVAILABLE else "stub"}


@router.get("/health")
def health():
    return {"status": "ok", "mode": "gemini" if GEMINI_AVAILABLE else "stub"}
