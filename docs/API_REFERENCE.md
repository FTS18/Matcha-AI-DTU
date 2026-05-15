# API & Events Reference Guide

This document outlines the strictly-typed contracts established between **Next.js** $\rightarrow$ **Orchestrator** $\leftrightarrow$ **Inference Engine**.

> **Standardized Safety**: Since the Mono-Refinement, every API payload is validated at the gateway level using **`@matcha/contracts`**. If a payload does not strictly match the Zod schema, the request is rejected with a 400 Bad Request before hitting the service logic.

---

## 1. Orchestrator API (NestJS — Port 4000)

These are operations exposed primarily to the Frontend Client (Next.js).

### `POST /matches`

Initiates a new video processing task.

- **Headers**: `Authorization: Bearer <JWT_TOKEN>`
- **Content-Type**: `multipart/form-data`
- **Body**:

- `file`: `(Binary)` The video file (e.g. .mp4, .mov, .mkv). Max 5GB.

- `title`: `(String)` Optional display title.
- **Response**: `201 Created`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",

  "status": "UPLOADED",

  "uploadUrl": "http://localhost:4000/uploads/1708532400000-match.mp4",

  "progress": 0,

  "duration": 0,

  "createdAt": "2024-02-21T18:30:00.000Z"
}
```

> After the 201 response, the orchestrator immediately fires `triggerInference()` in the background with up to 5 retry attempts (exponential backoff: 2s, 4s, 8s, 16s).

### `GET /matches`

Retrieves all matches for the authenticated user, ordered by most recent.

- **Headers**: `Authorization: Bearer <JWT_TOKEN>`
- **Response**: `200 OK` — Array of match objects with event and highlight counts.

```json
[
  {
    "id": "550e8400-...",

    "status": "COMPLETED",

    "progress": 100,

    "duration": 5400.0,

    "createdAt": "2024-02-21T18:30:00.000Z",

    "_count": {
      "events": 12,

      "highlights": 5
    }
  }
]
```

### `GET /matches/:id`

Retrieves fully analyzed data for a specific match including all events, highlights, emotion scores, and analytics fields. Applies user possession checks via context.

- **Headers**: `Authorization: Bearer <JWT_TOKEN>`
- **Response**: `200 OK` — Full match detail object.

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",

  "status": "COMPLETED",

  "progress": 100,

  "uploadUrl": "http://localhost:4000/uploads/match.mp4",

  "duration": 5400.0,

  "summary": "A pulsating 90-minute encounter that saw relentless pressure from both sides...",

  "highlightReelUrl": "http://localhost:4000/uploads/highlight_reel_550e8400.mp4",

  "trackingData": [
    {
      "t": 0.0,
      "b": [[0.51, 0.62, 0.03, 0.04, 0.88]],
      "p": [[0.3, 0.5, 0.08, 0.2, 1, 0]]
    }
  ],

  "teamColors": [
    [220, 50, 50],
    [50, 80, 220]
  ],

  "heatmapUrl": "http://localhost:4000/uploads/heatmap_550e8400.png",

  "topSpeedKmh": 78.4,

  "events": [
    {
      "id": "evt-001",

      "timestamp": 42.5,

      "type": "GOAL",

      "confidence": 0.95,

      "finalScore": 9.2,

      "commentary": "GOAAAL! What a thunderous strike! The ball cannons into the top right corner..."
    }
  ],

  "emotionScores": [
    {
      "timestamp": 30.0,

      "audioScore": 0.72,

      "motionScore": 0.65,

      "contextWeight": 0.65,

      "finalScore": 6.8
    }
  ],

  "highlights": [
    {
      "startTime": 32.5,

      "endTime": 62.5,

      "score": 9.2,

      "eventType": "GOAL",

      "commentary": "GOAAAL! What a thunderous strike...",

      "videoUrl": null
    }
  ]
}
```

### `DELETE /matches/:id`

Permanently deletes a match and all related records (events, highlights, emotion scores) from the database. Requires JWT possession verification.

- **Headers**: `Authorization: Bearer <JWT_TOKEN>`
- **Response**: `200 OK` — `{ "ok": true }`

### `POST /matches/:id/reanalyze`

Wipes all previous analysis results and re-triggers the inference pipeline on the original uploaded video. The match status is reset to `PROCESSING` and progress to `0`. Previous `heatmapUrl`, `topSpeedKmh`, `summary`, and all events are cleared.

- **Response**: `200 OK` — `{ "ok": true }`

---

## 2. Orchestrator WebSocket Events (Socket.IO — Port 4000)

Clients connect to `ws://localhost:4000` and join a match-specific room by emitting `joinMatch` with the match ID. All match-specific events are scoped to that room.

### Client → Server

#### `joinMatch`

Joins the real-time room for a specific match to receive live analysis updates.

```json
{ "matchId": "550e8400-..." }
```

#### `leaveMatch`

Leaves the match room.

```json
{ "matchId": "550e8400-..." }
```

### Server → Client

#### `progress`

Emitted continuously as the Python inference engine processes video frames. Fired approximately every 5 processed frames.

```json
{
  "matchId": "550e8400-...",

  "progress": 45
}
```

- `progress` ranges from `0` to `100`. A value of `-1` signals a fatal processing failure (match status will be `FAILED`).

#### `matchEvent`

Emitted immediately (fire-and-forget) when Phase 2 detects an event during processing — before the full analysis completes. This allows the frontend to populate the event feed in real-time while the video is still being processed.

```json
{
  "matchId": "550e8400-...",

  "event": {
    "timestamp": 42.5,

    "type": "GOAL",

    "confidence": 0.95,

    "finalScore": 9.2,

    "commentary": "GOAAAL! A breathtaking late winner...",

    "source": "soccernet"
  }
}
```

#### `complete`

Emitted when Phase 4 synthesis is complete and all data has been saved to the database. The frontend should re-fetch `GET /matches/:id` after receiving this to load the full payload including analytics, highlight reel URL, heatmap, and team colours.

```json
{
  "matchId": "550e8400-...",

  "eventCount": 12,

  "highlightCount": 5
}
```

---

## 3. Inference Engine API (FastAPI — Port 8000)

Operations exposed natively _(Orchestrator $\rightarrow$ Python)_. These are internal endpoints not intended to be called directly by the frontend.

### `GET /health`

Health check endpoint. Returns `200 OK` if the inference service is running.

- **Response**: `{ "status": "ok", "service": "inference" }`

### `POST /api/v1/analyze`

Triggers the main 5-phase inference pipeline asynchronously using FastAPI `BackgroundTasks`. Returns immediately with `202 Accepted` — the actual analysis runs in the background.

- **Content-Type**: `application/json`
- **Body**:

```json
{
  "match_id": "550e8400-...",

  "video_url": "/absolute/filesystem/path/to/match.mp4"
}
```

> Note: `video_url` is a **filesystem path**, not an HTTP URL. The orchestrator passes the local file path, while the publicly-accessible URL is stored separately in `uploadUrl`.

- **Response**: `200 OK`

```json
{ "status": "processing", "match_id": "550e8400-..." }
```

---

## 4. Orchestrator Internal Callbacks (Port 4000)

Private routes called by the Python inference engine to push updates back to NestJS during analysis. These are fire-and-forget from Python's perspective (1–2 second timeouts).

### `POST /matches/:id/progress`

Updates processing progress and broadcasts via WebSocket.

- **Body**: `{ "progress": 45 }` — Integer 0–100. `-1` signals failure.

### `POST /matches/:id/live-event`

Immediately broadcasts a newly-detected event via WebSocket to all connected frontend clients watching this match. The event is **not** saved to the database here — only broadcast. Final persistence happens in `complete`.

- **Body**:

```json
{
  "timestamp": 42.5,

  "type": "GOAL",

  "confidence": 0.95,

  "finalScore": 9.2,

  "commentary": "GOAAAL!",

  "source": "soccernet"
}
```

### `POST /matches/:id/complete`

Called once at the end of Phase 4 (after highlight reel generation). Saves all analysis results to PostgreSQL in a single Prisma `$transaction` and broadcasts the `complete` WebSocket event to connected clients.

- **Body** (full payload):

```json
{
  "events": [
    {
      "timestamp": 42.5,

      "type": "GOAL",

      "confidence": 0.95,

      "finalScore": 9.2,

      "commentary": "GOAAAL! A thunderous late winner..."
    }
  ],

  "highlights": [
    {
      "startTime": 32.5,

      "endTime": 62.5,

      "score": 9.2,

      "eventType": "GOAL",

      "commentary": "GOAAAL!..."
    }
  ],

  "emotionScores": [
    {
      "timestamp": 30.0,

      "audioScore": 0.72,

      "motionScore": 0.65,

      "contextWeight": 0.65,

      "finalScore": 6.8
    }
  ],

  "duration": 5400.0,

  "summary": "A pulsating 90-minute encounter...",

  "highlightReelUrl": "http://localhost:4000/uploads/highlight_reel_550e.mp4",

  "trackingData": [{ "t": 0.02, "b": [], "p": [] }],

  "teamColors": [
    [220, 50, 50],
    [50, 80, 220]
  ],

  "heatmapUrl": "http://localhost:4000/uploads/heatmap_550e8400.png",

  "topSpeedKmh": 78.4
}
```

---

## 5. Event Types Reference

The following `EventType` enum values are valid across all endpoints:

| Type          | Weight (0–10) | Source                          | Description                     |
| ------------- | ------------- | ------------------------------- | ------------------------------- |
| `GOAL`        | 10.0          | GoalDetectionEngine / SoccerNet | Ball crossed the goal line      |
| `PENALTY`     | 9.5           | SoccerNet                       | Penalty kick awarded            |
| `RED_CARD`    | 9.0           | SoccerNet                       | Player sent off                 |
| `SAVE`        | 8.0           | SoccerNet / Vision AI           | Goalkeeper saves a shot         |
| `YELLOW_CARD` | 7.0           | SoccerNet                       | Player booked                   |
| `CELEBRATION` | 6.5           | SoccerNet / Vision AI           | Players celebrating             |
| `FOUL`        | 6.0           | SoccerNet                       | Foul committed                  |
| `HIGHLIGHT`   | 5.5           | Motion Fallback                 | Generic high-action moment      |
| `TACKLE`      | 5.0           | SoccerNet / Vision AI           | Physical challenge for the ball |
| `CORNER`      | 4.0           | SoccerNet                       | Corner kick                     |
| `OFFSIDE`     | 2.5           | SoccerNet                       | Offside flag raised             |

---

## 6. Context Score Formula

Every detected event is assigned a `finalScore` (0–10) via `compute_context_score()`:

```
ew

= EVENT_WEIGHTS[type] / 10.0

# event type weight
audio = min(motionScore × 1.3, 1.0)

# motion-derived audio intensity proxy
tw

= time_context_weight(timestamp)

# temporal weight (late game = higher)
base = (ew × 0.40) + (audio × 0.20) + (motionScore × 0.25) + (tw × 0.15)
score = base × (0.5 + 0.5 × confidence)

# confidence scaling

# Special boosts:
if timestamp/duration > 0.85 and type == "GOAL": score *= 2.0 # late goal
if timestamp/duration < 0.08 and type in ("SAVE", "TACKLE"): score *= 1.3 # frantic start
```

Temporal weight table:

| Match Position          | Weight |
| ----------------------- | ------ |
| Injury time (>92%)      | 1.00   |
| Final 10 min (>85%)     | 0.95   |
| Last quarter (>70%)     | 0.85   |
| Second half (>50%)      | 0.75   |
| Around half-time (>45%) | 0.60   |
| First half              | 0.65   |

---

## 7. Static File Serving

The Orchestrator serves uploaded videos, generated highlight reels, and heatmap images from the shared `/uploads/` directory at:

```
GET http://localhost:4000/uploads/<filename>
```

File naming conventions: | File | Naming Pattern | Format | |---|---|---| | Uploaded match video | `{timestamp}-{original_name}.mp4` | MP4 | | Compressed intermediate | `compressed_{matchId}.mp4` | MP4 (deleted after analysis) | | Highlight reel | `highlight_reel_{matchId}.mp4` | MP4 | | Player heatmap | `heatmap_{matchId}.png` | PNG (800×520px) | | TTS audio clip | `temp_audio_{matchId}_{i}.wav` | WAV (deleted after muxing) |
