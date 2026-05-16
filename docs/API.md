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
  "uploadUrl": "http://localhost:4000/uploads/match.mp4",
  "progress": 0,
  "duration": 0,
  "createdAt": "2024-02-21T18:30:00.000Z"
}
```

### `GET /matches`
Retrieves all matches for the authenticated user.

- **Response**: `200 OK` — Array of match objects.

```json
[
  {
    "id": "550e8400-...",
    "status": "COMPLETED",
    "progress": 100,
    "duration": 5400.0,
    "_count": { "events": 12, "highlights": 5 }
  }
]
```

### `GET /matches/:id`
Retrieves full analyzed data for a specific match.

- **Response**: `200 OK` — Full match detail object.

---

## 2. Orchestrator WebSocket Events (Socket.IO)

### Client → Server
- `joinMatch`: `{ "matchId": "string" }`
- `leaveMatch`: `{ "matchId": "string" }`

### Server → Client
- `progress`: `{ "matchId": "string", "progress": number }`
- `matchEvent`: `{ "matchId": "string", "event": { ...Data } }`
- `complete`: `{ "matchId": "string", "eventCount": number, "highlightCount": number }`

---

## 3. Inference Engine API (FastAPI — Port 8000)

### `POST /api/v1/analyze`
Triggers the main 5-phase inference pipeline.

- **Body**:
```json
{
  "match_id": "550e8400-...",
  "video_url": "/absolute/path/to/match.mp4"
}
```

---

## 4. Internal Callbacks (Port 4000)

- `POST /matches/:id/progress`: Updates integer progress (0-100).
- `POST /matches/:id/live-event`: Broadcasts a newly-detected event.
- `POST /matches/:id/complete`: Final payload delivery + Persistence.

---

## 5. Event Types Reference

| Type | Weight | Source | Description |
| --- | --- | --- | --- |
| `GOAL` | 10.0 | GoalEngine | Ball crossed the line |
| `PENALTY` | 9.5 | SoccerNet | Penalty kick awarded |
| `RED_CARD` | 9.0 | SoccerNet | Player sent off |
| `SAVE` | 8.0 | Vision AI | Keeper save |
| `FOUL` | 6.0 | SoccerNet | Foul committed |

---

## 6. Context Score Formula

Every event is assigned a `finalScore` (0–10) via `compute_context_score()`:

```text
Score = (ew * 0.40) + (audio * 0.20) + (motion * 0.25) + (tw * 0.15)
Where:
- ew: Event Weight (GOAL=1.0, FOUL=0.6)
- audio: Audio intensity proxy
- motion: OpenCV frame difference
- tw: Temporal Weight (Late game = higher)
```

---

## 7. Static File Serving

Files are served from `/uploads/`:
- **Match Video**: `{timestamp}-{name}.mp4`
- **Highlight Reel**: `highlight_reel_{matchId}.mp4`
- **Heatmap**: `heatmap_{matchId}.png`
