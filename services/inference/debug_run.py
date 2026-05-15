import sys
import logging
from app.core.analysis import analyze_video
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# find a video in uploads
uploads = (
    list(Path("../../workspace/uploads").glob("*.mp4"))
    + list(Path("../../uploads").glob("*.mp4"))
    + list(Path("../uploads").glob("*.mp4"))
    + list(Path("../../Matcha-AI-DTU/uploads").glob("*.mp4"))
)
print("Uploads found:", uploads)

if uploads:
    try:
        analyze_video(str(uploads[0]), "test_123")
        print("Success")
    except Exception as e:
        import traceback

        traceback.print_exc()
else:
    print("No video found")
