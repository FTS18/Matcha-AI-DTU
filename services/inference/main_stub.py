from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Use stub routes that don't require heavy dependencies
from app.api.routes_stub import router as api_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Matcha AI Inference Service (Stub Mode)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "inference", "mode": "stub"}

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Inference Server (Stub Mode) on http://0.0.0.0:8000")
        uvicorn.run("main_stub:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        import traceback
        traceback.print_exc()
