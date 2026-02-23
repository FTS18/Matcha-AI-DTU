from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from app.core.analysis import analyze_video

router = APIRouter()

class VideoAnalysisRequest(BaseModel):
    match_id: str
    video_url: str
    start_time:   Optional[float] = None
    end_time:     Optional[float] = None
    language:     str = "english"
    aspect_ratio: str = "16:9"

@router.post("/analyze")
async def analyze_match(request: VideoAnalysisRequest, background_tasks: BackgroundTasks):
    # Offload to background task
    background_tasks.add_task(
        analyze_video, 
        request.video_url, 
        request.match_id,
        start_time=request.start_time,
        end_time=request.end_time,
        language=request.language,
        aspect_ratio=request.aspect_ratio
    )
    return {"status": "processing", "match_id": request.match_id}


@router.get("/yt-info")
async def youtube_info(url: str = Query(..., description="YouTube URL")):
    """Fetch YouTube video metadata (title, duration, thumbnail) without downloading."""
    try:
        import yt_dlp
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": "best",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration", 0),          # seconds
                "thumbnail": info.get("thumbnail", ""),
                "channel": info.get("uploader", ""),
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch video info: {e}")
