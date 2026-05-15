# Matcha-AI-DTU Inference Engine (`services/inference`)

The pure Python powerhouse behind analyzing soccer videos within the Matcha-AI-DTU platform. Exposed via **FastAPI** (`uvicorn`), this microservice digests videos forwarded by the Orchestrator, running a 5-phase pipeline from state-of-the-art Computer Vision through AI LLM commentary to neural audio synthesis and post-game analytics.

## 5-Phase Pipeline Overview

When a video hits the Inference API, it undergoes a sophisticated sequential analysis pipeline (`app/core/analysis.py`):

1. **Phase 1 — Video Decoding & YOLO Detection**:

- Pre-compresses large videos (>100MB) via FFmpeg to 480p @ 1fps for faster processing.
- `YOLOv8s.pt` detects `sports ball` and `person` bounding boxes per frame.
- `GoalDetectionEngine`: Kalman filter + homography projection + finite state machine auto-calibrates to goal lines and confirms goals with high precision.
- Jersey crops are extracted per player for team colour clustering.
- All bounding boxes stored as `track_frames` (normalised coords) for Phase 5.

2. **Phase 2 — Action Event Recognition**:

- **SoccerNet** trained model detects 11 event types: GOAL, SAVE, TACKLE, FOUL, CORNER, YELLOW_CARD, RED_CARD, PENALTY, OFFSIDE, CELEBRATION, HIGHLIGHT.
- Motion-peak fallback engaged if SoccerNet returns < 3 events.
- Every event scored 0–10 via weighted formula: event importance + motion intensity + temporal position + confidence.
- Events emitted live to Orchestrator WebSocket as they are detected.

3. **Phase 3 — Generative AI Commentary (Google Gemini 2.0 Flash)**:

- Per-event: 40–60 word live-broadcast style commentary prompt, with late-game intensity boost.
- Full match: 3–5 sentence analytical narrative summary of the entire game.
- Fallback template library used when Gemini is unavailable.

4. **Phase 4 — Highlight Reel & Neural TTS Synthesis**:

- Top-N non-overlapping highlight clips selected based on context score.
- **3-tier TTS pipeline**:
- **Kokoro-82M** (`hexgrad/Kokoro-82M`) — #1 ranked in TTS-Spaces-Arena, British female voice (`af_sky`). Requires `HF_TOKEN`.
- **edge-tts** (`en-GB-RyanNeural`) — British male broadcaster, no API key needed.
- **FFmpeg silence** — absolute fallback.
- FFmpeg mixes TTS audio + crowd ambience + background music onto each clip.
- All clips concatenated into `highlight_reel_{matchId}.mp4`.

5. **Phase 5 — Post-Game Analytics** (`app/core/heatmap.py`): _(NEW)_

- **Player Density Heatmap**: Accumulates YOLO player centroids into 2D NumPy grid per team. Gaussian-blurred and rendered as an OpenCV pitch diagram PNG (`800×520px`). Saved to `/uploads/heatmap_{matchId}.png`.
- **Ball Speed Estimation**: Consecutive ball positions → pixel delta → metres (105m pitch) → km/h. Returns 95th-percentile to suppress noise. Clamped 0–200 km/h.
- **Team Colour Detection**: NumPy K-Means (2 centroids, 20 iterations) on jersey crops from all frames. Results drive heatmap overlay colours and frontend swatch display.

## Hardware Support Matrix

| Mode     | Performance                    | Requirement                |
| -------- | ------------------------------ | -------------------------- |
| CPU-only | ~2–5 min per minute of footage | Default — works everywhere |
| CUDA GPU | ~10–30x faster                 | NVIDIA GPU with CUDA 12.4  |

```python
import torch
print(torch.cuda.is_available()) # Check if GPU is detected
print(torch.cuda.get_device_name(0)) # e.g., "NVIDIA GeForce RTX 3050"
```

## Startup Environment

### Setting Requirements

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install CUDA-enabled PyTorch (if NVIDIA GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install all requirements (includes lapx, huggingface-hub, edge-tts, opencv, etc.)
pip install -r requirements.txt
```

> **Note**: `piper-tts` is **no longer needed**. It has been replaced by Kokoro-82M + edge-tts in the 3-tier neural TTS system.

### Configuration (`.env`)

Create `services/inference/.env`:

```env
GEMINI_API_KEY=your_google_ai_studio_key
HF_TOKEN=hf_your_huggingface_token
ORCHESTRATOR_URL=http://localhost:4000
```

| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Google AI Studio key — enables Gemini commentary and match summary |
| `HF_TOKEN` | Recommended | HuggingFace token — enables Kokoro-82M TTS (Tier 1). Free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `ORCHESTRATOR_URL` | Optional | Default: `http://localhost:4000` |

### Launch Service

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

_Wait for the service to log ` Starting Inference Server` and `Heatmap module loaded `. YOLO weights are loaded into memory on first request._

## Performance Notes

- **Pre-compression**: Videos >100MB are automatically downscaled to 480p @ 1fps before the analysis loop, reducing frame count dramatically without significant accuracy loss.
- **YOLO skip optimisation**: Frames with very low motion score (`< 0.15`) skip YOLO inference entirely, saving GPU time on static scenes.
- **Phase 5 analytics** runs on already-collected `track_frames` — no additional video re-read needed after the main loop.
- **Auto-rescaling**: Frames wider than 800px are downscaled before the YOLO inference call, protecting VRAM limits.

## Key Files

| File                             | Purpose                                             |
| -------------------------------- | --------------------------------------------------- |
| `app/core/analysis.py`           | Main 5-phase pipeline orchestrator (1458+ lines)    |
| `app/core/heatmap.py`            | Phase 5: player heatmap PNG + ball speed estimation |
| `app/core/goal_detection.py`     | Kalman + homography + FSM goal detection engine     |
| `app/core/soccernet_detector.py` | SoccerNet-trained football event detector           |
| `app/api/routes.py`              | FastAPI route: `POST /api/v1/analyze`               |
| `main.py`                        | FastAPI app entrypoint + CORS + health check        |
| `yolov8s.pt`                     | YOLOv8 small model weights (22MB)                   |
| `requirements.txt`               | Python dependencies                                 |
