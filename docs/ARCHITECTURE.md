# System Architecture

Matcha-AI-DTU utilizes a microservice-like **Monorepo** architecture leveraging specific languages for their native strengths:

- **TypeScript/React** for dynamic, real-time UI mapping.
- **TypeScript/NestJS** for strictly-typed API gateways and WebSockets.
- **Python/FastAPI** for deep learning AI inference processing.

This document serves to visually explain how data flows across the monorepo when a user initiates a request.

---

## The Core Video Analysis Data flow

The primary end-to-end operation is ingesting a raw video, processing it through YOLO Computer Vision AI models and Large Language Models, and returning a generated Sports Highlight audio synthetic file dynamically to the browser.

```mermaid
sequenceDiagram

autonumber

actor User as Client (Next.js App)

participant Env as @matcha/env (Validator)

participant Orch as Orchestrator (NestJS)

participant Cont as @matcha/contracts (Zod)

participant Redis as Redis Cache

participant DB as @matcha/database (Prisma)

participant ML as Inference Engine (Python)

%% -------------------------------------

%% Stage 1: Upload & Registration Focus

%% -------------------------------------

User->>Env: Validate Public Env (NEXT_PUBLIC_*)

User->>Orch: POST /matches (Uploads Video File)

activate Orch

Orch->>Cont: Validate Payload (Zod)

Orch->>DB: Create Match Record & Return ID

Orch->>Redis: Set Processing State [Analysis Queue]

Orch->>ML: POST /analyze { videoPath, matchId }

activate ML

Orch-->>User: HTTP 201 Created { matchId }

deactivate Orch

%% -------------------------------------

%% Stage 2: Processing & Live Updates

%% -------------------------------------

note over ML,User: Real-Time Event Loop via WebSockets

loop Per Video Frame Batch

ML->>ML: Run Phase 1b: Goal Detection Pipeline

ML->>ML: Run Phase 2: Action parsing

ML->>Orch: POST /callbacks/progress { matchId, progress: x% }

Orch->>User: WS Event "analysisProgress" [Progress Bar Update]

opt Event Detected (e.g. GOAL!)

ML->>Orch: POST /callbacks/event { ...EventData }

Orch->>Cont: Validate Callback Object

Orch->>DB: Save Event Record

Orch->>User: WS Event "eventDetected" [Live UI Popup]

end

end

%% -------------------------------------

%% Stage 3: Generative AI & Audio Render

%% -------------------------------------

ML->>ML: Formulate script using Google Gemini

ML->>ML: Render voiceover via Piper TTS (.wav)

ML->>Orch: POST /callbacks/complete { highlightMetadata }

deactivate ML

activate Orch

Orch->>DB: Update Match Status = "COMPLETED"

Orch->>Redis: Clear Cache Queue

Orch-->>User: WS Event "analysisComplete" [Refresh App Data]

deactivate Orch

User->>Orch: GET /matches/{id}

Orch-->>User: Return full payload with AI Audio URL
```

---

## Database Schema Diagram

We utilize Prisma ORM for type-safe database queries. The central entities revolve around `Match` (a video entity) and its children `MatchEvent`.

```mermaid
erDiagram

MATCH {

String id PK

String title

String videoUrl

String ttsAudioUrl

String status

DateTime createdAt

DateTime updatedAt

}

MATCH_EVENT {

String id PK

String matchId FK

String eventType "GOAL, TACKLE, FOUL, etc."

Float timestampSeconds

Float confidence

String description

}

MATCH ||--o{ MATCH_EVENT : "has many"
```

---

## WebSockets Implementation (Socket.io)

For real-time progression we decouple the heavy Python operations from holding HTTP connections open using Callbacks and WebSockets.

1. **NestJS** mounts a Socket.IO Gateway on port `4000`.
2. When the **Inference (Python)** script processes frames, it issues a synchronous _fire-and-forget_ HTTP request to the Orchestrator (`/callbacks/progress`).
3. The **Orchestrator** translates this HTTP payload into an active Socket Event mapped to all connected Next.js users listening to that namespace.
4. If a WebSocket disconnects, the analysis **continues uninterrupted** inside the Python environment, preventing dropped progress upon a user refreshing their browser.

---

## Updated Database Schema (v2 — Full Analytics Platform)

Since the initial architecture, the Prisma schema has been significantly expanded to support the full analytics platform. Migration `20260221180240_add_heatmap_speed` was the most recent, adding heatmap and speed fields.

The complete current schema is documented below:

```mermaid
erDiagram

MATCH {

String

id

PK

String

userId

FK "Links securely to authenticated user via JWT"

String

uploadUrl

String

status

"UPLOADED | PROCESSING | COMPLETED | FAILED"

Int

progress

"0–100 processing percent"

Float

duration

"total video length in seconds"

String

summary

"AI Gemini 3-5 sentence narrative"

String

highlightReelUrl "URL to stitched MP4 highlight reel"

Json

trackingData

"YOLO bbox frames: [{t, b[], p[]}]"

Json

teamColors

"[[R,G,B],[R,G,B]] KMeans jersey clusters"

String

heatmapUrl

"URL to player density heatmap PNG (NEW)"

Float

topSpeedKmh

"95th-pct peak ball speed km/h (NEW)"

DateTime createdAt

DateTime updatedAt

}

USER {

String

id

PK

String

email

UNIQUE

String

password

"bcrypt hashed"

String

name

DateTime createdAt

}

USER

||--o{ MATCH

: "owns"

EVENT {

String

id

PK

String

matchId

FK

Float

timestamp "seconds, 0.01s precision"

String

type

"GOAL | FOUL | TACKLE | SAVE | CELEBRATION | HIGHLIGHT | PENALTY | RED_CARD | YELLOW_CARD | CORNER | OFFSIDE"

Float

confidence "0.0–1.0 detection confidence"

Float

finalScore "0–10 composite weighted context score"

String

commentary "Gemini-generated broadcast commentary text"

DateTime createdAt

}

EMOTION_SCORE {

String

id

PK

String

matchId

FK

Float

timestamp

Float

audioScore

"0.0–1.0 motion-derived intensity"

Float

motionScore

"0.0–1.0 OpenCV frame-diff score"

Float

contextWeight "0.0–1.0 time-position temporal weight"

Float

finalScore

"0–10 composite"

DateTime createdAt

}

HIGHLIGHT {

String

id

PK

String

matchId

FK

Float

startTime "clip start in seconds"

Float

endTime

"clip end in seconds"

Float

score

"0–10 highlight significance"

String

eventType "GOAL | SAVE | TACKLE etc."

String

commentary "highlight description"

String

videoUrl

"optional clip URL"

DateTime createdAt

}

MATCH ||--o{ EVENT

: "has many"

MATCH ||--o{ EMOTION_SCORE : "has many"

MATCH ||--o{ HIGHLIGHT

: "has many"
```

---

## Analytics Pipeline — Phase 5 (NEW)

After the main event detection loop completes, a **Phase 5 Post-Processing Analytics** pass runs entirely within the inference service using the `track_frames` data accumulated by YOLO during the primary loop. No additional video re-read is needed.

```mermaid
graph TD

A["YOLO Frame Loop\n(Phases 1–4 complete)"] --> TF["track_frames list\ncollected during the loop"]

TF --> H["generate_heatmap()\napp/core/heatmap.py"]

TF --> S["estimate_ball_speed()\napp/core/heatmap.py"]

TF --> TC["_cluster_teams()\napp/core/analysis.py\nK-Means jersey clustering"]

H --> HI["heatmap_matchId.png\nOpenCV top-down pitch render\nGaussian-blurred density overlay\nsaved → /uploads/"]

S --> SP["topSpeedKmh float\n95th-percentile ball speed\n0–200 km/h clamped"]

TC --> CO["teamColors [[R,G,B],[R,G,B]]\nused for heatmap colour coding\nand frontend swatch display"]

HI --> PL["Orchestrator PATCH\nPOST /matches/:id/complete"]

SP --> PL

CO --> PL

PL --> DB["PostgreSQL\nheatmapUrl topSpeedKmh teamColors"]

DB --> FE["Frontend Analytics Tab\n Ball Speed · Team Colors · Heatmap"]
```

### Phase 5 Implementation Details

| Component | File | Algorithm |
| --- | --- | --- |
| Heatmap generator | `app/core/heatmap.py` | For each tracked player, accumulate centroid `+1` into a 2D NumPy grid. Run `cv2.GaussianBlur(kernel=51)`, then overlay blended colour per team onto a pitch background drawn with OpenCV primitives |
| Ball speed estimator | `app/core/heatmap.py` | Extract consecutive ball `(cx_norm, cy_norm)` positions. Multiply by real pitch dims (105m × 68m) to get metres. Divide by `Δt → m/s → km/h`. Return 95th-percentile to suppress tracking noise spikes |
| Team colour detector | `app/core/analysis.py` | `_crop_jersey()` extracts the torso region (30–70% height, 20–80% width) of each detected player box. `_dominant_colour()` computes the median RGB. `_cluster_teams()` runs NumPy K-Means (20 iterations, 2 centroids) to split players into two jersey-colour groups |
| Analytics tab (frontend) | `apps/web/app/matches/[id]/page.tsx` | New `"analytics"` tab state. Renders: speed with amber glow text, team hex swatches with box-shadow glow, `<img>` heatmap tag loaded from `match.heatmapUrl` |

---

## TTS (Text-to-Speech) Architecture — 3-Tier System (UPDATED)

The commentary voiceover system has been upgraded from a single `edge-tts` implementation to a **3-tier cascading fallback chain** ensuring the highest quality audio is always used while guaranteeing production is never blocked.

```mermaid
graph TD

T["tts_generate(text, output_path)\ncalled once per highlight clip"] --> K

K[" Tier 1: Kokoro-82M\nhexgrad/Kokoro-82M\nHuggingFace Inference API\nVoice: af_sky (British female)\n#1 ranked — TTS Arena Leaderboard"]

K -- " HTTP 200 + audio bytes" --> OUT

K -- " Rate-limited / ImportError / timeout" --> E

E[" Tier 2: Microsoft edge-tts\nen-GB-RyanNeural\nBritish male broadcaster\nNo API key required — always available"]

E -- " asyncio.run + .save()" --> OUT

E -- " Network error / not installed" --> S

S[" Tier 3: FFmpeg anullsrc\nSilent 44100Hz stereo audio\nAbsolute fallback — never fails"]

S --> OUT

OUT["Audio file written to disk\nMixed into highlight reel via FFmpeg"]
```

**Configuration constants in `analysis.py`:**

```python
_KOKORO_MODEL

= "hexgrad/Kokoro-82M"

# Top TTS Arena model
_KOKORO_VOICE

= "af_sky"

# British female sports commentator
_EDGE_TTS_VOICE = "en-GB-RyanNeural"

# British male broadcaster (fallback)
_hf_token

= os.getenv("HF_TOKEN") # Set in services/inference/.env
```

---

## Frontend Component Architecture

```
apps/web/
├── app/
│

├── page.tsx

Hero landing page
│

│

Full-screen video bg + overlay stats
│

├── layout.tsx

Root layout — Navbar + Footer injected globally
│

├── globals.css

CSS design tokens, animations
│

│

.hide-scrollbar utility class (NEW)
│

└── matches/
│

├── page.tsx

Match Dashboard
│

│

Status filters: UPLOADED / PROCESSING / COMPLETED / FAILED
│

│

Responsive card grid (Android-optimised)
│

└── [id]/
│

└── page.tsx

Match Detail Page
│

├── Video Player

Seeks to YOLO-detected event timestamps on click
│

├── Top 5 Moments Horizontally scrollable podium cards (gold/silver/bronze)
│

├── Intensity Chart Motion score sparkline over match duration
│

├── Highlights Tab Highlight clip cards with TTS commentary
│

├── Events Tab

Full event timeline + live WebSocket feed during processing
│

└── Analytics Tab ← NEW
│

├── Ball Speed

95th-pct peak km/h from YOLO tracking
│

├── Team Colors Auto-detected jersey swatches (hex codes)

│

└── Heatmap

OpenCV player density PNG with pitch overlay
│

└── PDF Report Generator (@react-pdf/renderer)
├── components/
│

├── layout/
│

│

├── Navbar.tsx

Responsive top nav with mobile hamburger
│

│

└── Footer.tsx

Global footer
│

├── match-dashboard.tsx

Filterable card grid
│

│

Responsive filter tabs + stat pills (Android-optimised)
│

└── ui/

shadcn/ui component primitives
```

---

## Environment Variables Reference

All services load `.env` from their own service directory. Never commit these files.

### `services/orchestrator/.env`

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `DATABASE_URL` | Yes | — | PostgreSQL connection string (`postgresql://user:pass@host:port/db`) |
| `HF_TOKEN` | Recommended | anonymous | HuggingFace token — raises Kokoro TTS rate limits significantly |
| `PORT` |  | `4000` | NestJS HTTP + WebSocket port |
| `CORS_ORIGIN` |  | `http://localhost:3000` | Comma-separated list of allowed frontend origins |
| `INFERENCE_URL` |  | `http://localhost:8000` | Base URL to the Python inference service |
| `REQUEST_TIMEOUT` |  | `30000` | HTTP request timeout in milliseconds |

### `services/inference/.env`

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | — | Google AI Studio API key for Gemini 2.0 Flash (commentary + summary) |
| `HF_TOKEN` | Recommended | anonymous | HuggingFace token for Kokoro-82M TTS (Tier 1) |
| `ORCHESTRATOR_URL` |  | `http://localhost:4000` | Orchestrator base URL for progress callbacks |

---

## Complete Monorepo Directory Layout

```
Matcha-AI-DTU/
├── apps/
│

├── web/

Next.js 14 frontend (App Router)
│

│

├── app/

Pages and layouts
│

│

├── components/

Shared React components
│

│

└── public/

Static assets
│

└── mobile/

Expo React Native (future)
│
├── packages/
│

└── shared/

Universal TS Types, APIs, Socket enums @matcha/shared
│
├── services/
│

├── orchestrator/

NestJS API gateway (port 4000)
│

│

├── src/
│

│

│

├── matches/

Match upload, analysis, retrieval module
│

│

│

│

├── matches.controller.ts
│

│

│

│

└── matches.service.ts
│

│

│

└── events/

Socket.IO real-time gateway
│

│

│

└── events.gateway.ts
│

│

└── prisma/
│

│

├── schema.prisma

Full Prisma ORM schema
│

│

└── migrations/

All applied DB migrations
│

│
│

└── inference/

Python FastAPI AI engine (port 8000)
│

├── app/
│

│

├── core/
│

│

│

├── analysis.py

Main 5-phase pipeline orchestrator
│

│

│

├── heatmap.py

Player heatmap + ball speed (NEW)
│

│

│

├── goal_detection.py Kalman + Homography + FSM goal engine
│

│

│

└── soccernet_detector.py SoccerNet trained event detector
│

│

└── api/
│

│

└── routes.py

FastAPI route definitions
│

├── yolov8n.pt

YOLOv8 nano model weights
│

├── yolov8s.pt

YOLOv8 small model weights (default)
│

├── requirements.txt

Python dependencies
│

└── AI_PIPELINE.md

Detailed pipeline documentation
│
├── uploads/

Shared video + generated asset directory
│

(match videos, heatmap PNGs, highlight reels)
│
├── docs/
│

├── ARCHITECTURE.md

← This file
│

├── API_REFERENCE.md

HTTP + WebSocket API contracts
│

└── CONTRIBUTING.md

Contribution guidelines
│
├── turbo.json

Turborepo task pipeline config
└── package.json

Monorepo root workspace config
```
