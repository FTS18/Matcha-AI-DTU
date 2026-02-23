import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Load .env file BEFORE importing anything that reads env vars
from pathlib import Path
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
from app.api.routes import router as api_router
from app.core.analysis import _get_tts

app = FastAPI(title="Matcha AI Inference Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
def startup_event():
    # Pre-load TTS model in background to avoid blocking startup
    threading.Thread(target=_get_tts, daemon=True).start()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "inference"}

if __name__ == "__main__":
    try:
        print("🚀 Starting Inference Server on http://0.0.0.0:8000")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()

