# Application Walkthrough

This document provides a detailed, step-by-step walkthrough of the Matcha-AI-DTU platform from the perspective of a user. It describes every screen, interaction, and system behavior that a user encounters from the moment they open the application to the moment they view and download their complete match analysis. This document is intended to serve as both a user guide and a functional specification for contributors building new features.

---

## Table of Contents

1. [Prerequisites and Initial State](#1-prerequisites-and-initial-state)
2. [Landing Page and Authentication](#2-landing-page-and-authentication)
3. [Registration Flow](#3-registration-flow)
4. [Login Flow](#4-login-flow)
5. [Match Dashboard — Overview](#5-match-dashboard--overview)
6. [Uploading a Match Video](#6-uploading-a-match-video)
7. [Real-Time Analysis Progress](#7-real-time-analysis-progress)
8. [Match Detail Page — Video Player](#8-match-detail-page--video-player)
9. [Match Detail Page — Top 5 Moments](#9-match-detail-page--top-5-moments)
10. [Match Detail Page — Highlights Tab](#10-match-detail-page--highlights-tab)
11. [Match Detail Page — Events Tab](#11-match-detail-page--events-tab)
12. [Match Detail Page — Analytics Tab](#12-match-detail-page--analytics-tab)
13. [PDF Match Report Generation](#13-pdf-match-report-generation)
14. [Error States and Edge Cases](#14-error-states-and-edge-cases)
15. [Data Flow Summary](#15-data-flow-summary)

---

## 1. Prerequisites and Initial State

Before the application is usable, the following services must be running:

| Service                 | Port | Status Indicator                                                  |
| ----------------------- | ---- | ----------------------------------------------------------------- |
| PostgreSQL (Docker)     | 5433 | `docker ps` shows `matcha_postgres` as `Up`                       |
| Redis (Docker)          | 6380 | `docker ps` shows `matcha_redis` as `Up`                          |
| NestJS Orchestrator     | 4000 | Terminal shows `Nest application successfully started`            |
| Python Inference Engine | 8000 | Terminal shows `Uvicorn running on http://0.0.0.0:8000`           |
| Next.js Frontend        | 3000 | Terminal shows `Ready in Xms` with `Local: http://localhost:3000` |

If any of these services are not running, the corresponding functionality will fail. See `./SETUP.md` for the full startup sequence.

---

## 2. Landing Page and Authentication

### What the User Sees

When a user navigates to `http://localhost:3000` for the first time, they arrive at the Hero Landing Page. This page serves as both a marketing introduction and the entry point to the application.

The landing page contains:

- A full-screen background (video or gradient) establishing the sports analytics theme.
- A headline that describes the application's core value proposition.
- Two call-to-action buttons: one linking to the login page and one linking to the registration page.
- A statistics section showing example analytics numbers to demonstrate the platform's capability.
- A navigation bar at the top with links to Login and Register.

If the user already has a valid JWT token stored in their browser (from a previous session), the navigation bar will show a link to the Dashboard instead of Login/Register, and clicking the main call-to-action will redirect directly to the match dashboard.

---

## 3. Registration Flow

### Navigating to Registration

The user clicks "Get Started" or "Register" from the landing page, arriving at `/register`.

### The Registration Form

The registration form contains three fields:

- Full Name
- Email Address
- Password

All fields are required. Validation is performed on both the client side (immediate inline feedback) and the server side (Zod schema validation in `packages/contracts`).

Client-side validation rules:

- Name must be at least 2 characters.
- Email must be a syntactically valid email address.
- Password must be at least 8 characters.

### What Happens on Submission

1. The frontend sends a `POST /auth/register` request to `http://localhost:4000/api/v1/auth/register` with `{ name, email, password }`.
2. The NestJS `AuthModule` validates the payload using the Zod schema from `@matcha/contracts`.
3. If the email is already registered, the server returns `409 Conflict` and the frontend displays an inline error: "An account with this email already exists."
4. If validation passes, the password is hashed using `bcrypt` (12 salt rounds) and a new `User` record is created in PostgreSQL.
5. The server returns a JWT access token.
6. The frontend stores the token (in a cookie or `localStorage`, depending on implementation) and redirects the user to `/matches`.

---

## 4. Login Flow

### The Login Form

The login page at `/login` contains two fields:

- Email Address
- Password

### What Happens on Submission

1. The frontend sends `POST /auth/login` with `{ email, password }`.
2. The NestJS `AuthModule` finds the user by email using Prisma and compares the submitted password to the stored bcrypt hash.
3. If credentials are incorrect, the server returns `401 Unauthorized`. The frontend displays: "Invalid email or password."
4. If credentials are correct, the server returns a signed JWT token containing the user's ID.
5. The frontend stores the token and redirects to `/matches`.

### Token Persistence

The JWT is stored client-side and automatically attached to all subsequent API requests in the `Authorization: Bearer <token>` header. The token's expiry determines how long the user stays logged in before being prompted to log in again.

---

## 5. Match Dashboard — Overview

### What the User Sees

The match dashboard at `/matches` is the user's home base within the application. It shows all matches that the authenticated user has uploaded.

The dashboard layout consists of:

- A header with the user's name and a "Upload New Match" button.
- A filter tab bar with options: All, Uploaded, Processing, Completed, Failed.
- A responsive grid of match cards.

### Match Cards

Each match card displays:

- The match title (user-provided or auto-generated from the filename).
- A status badge color-coded by state:

- Grey: UPLOADED (waiting for analysis to start)

- Blue with an animated pulse: PROCESSING (analysis currently running)

- Green: COMPLETED (analysis finished, all data available)

- Red: FAILED (analysis encountered an unrecoverable error)
- A progress bar showing the percentage complete during PROCESSING state.
- The upload date.
- A brief summary if the match is COMPLETED.

Clicking any match card navigates to `/matches/[id]` — the match detail page.

### Filtering

The filter tabs query `GET /matches?status=PROCESSING` (or the relevant status). Only matches belonging to the authenticated user are returned. The tab switches are instant because they re-query the API.

---

## 6. Uploading a Match Video

### The Upload Flow

Clicking "Upload New Match" opens an upload dialog or navigates to the upload section of the landing page, depending on whether the user is already authenticated.

The upload interface consists of:

- A file input (or drag-and-drop zone in future — see ROADMAP.md item 3.5).
- A text field for the match title (optional — the system uses the filename if left blank).
- An "Analyze Match" button.

### Accepted File Formats

The system accepts `.mp4`, `.mov`, and `.avi` video files. Other formats should return a client-side validation error before the file is sent to the server.

### What Happens on Submission

1. The frontend sends a `multipart/form-data` `POST /matches` request with the video file and title.
2. The NestJS `MatchesController` receives the upload and calls `MatchesService.create()`.
3. The service saves the video file to the `uploads/` directory with a UUID-based filename to prevent collisions.
4. A new `Match` record is created in PostgreSQL with `status: UPLOADED`, `progress: 0`, and the file path stored as `uploadUrl`.
5. The service sends an asynchronous `POST /analyze` request to the Python inference engine at `http://localhost:8000/analyze` with `{ videoPath, matchId }`.
6. The NestJS service does not wait for the inference to complete. It immediately returns `HTTP 201 Created` with `{ matchId }` to the frontend.
7. The frontend navigates to `/matches/[matchId]` to show the real-time analysis progress.

**Important**: Steps 5-7 happen immediately. The actual video analysis runs asynchronously in the Python process and can take anywhere from 2 minutes to 20 minutes depending on video length and hardware.

---

## 7. Real-Time Analysis Progress

### The 5-Phase Pipeline

Once the Python inference engine begins processing, it runs through 5 sequential phases. Each phase reports progress back to the frontend in real time via WebSockets.

**Phase 1a — YOLO Frame Analysis** The inference engine opens the video file using OpenCV and reads it frame by frame. For every Nth frame (where N is defined by `CONFIG["FRAME_SKIP"]`), it runs YOLOv8 object detection to identify players and the ball. Each detection is stored in `track_frames` — a list of frame objects containing bounding boxes for all detected objects.

During this phase, the frontend shows a progress bar advancing from 0% to approximately 60%.

**Phase 1b — Goal Detection** Running concurrently with or immediately after Phase 1a, the custom `GoalDetectionEngine` processes the ball's positional data. It applies a Kalman filter to smooth the ball trajectory, then uses a homography matrix to project ball positions from pixel space into real-world pitch coordinates. A finite state machine determines when the ball has definitively crossed the goal line.

**Phase 2 — Action Recognition and Event Scoring** The system runs the SoccerNet-trained action recognition model on the accumulated tracking data. It detects 11 event types: GOAL, SAVE, TACKLE, FOUL, CORNER, YELLOW_CARD, RED_CARD, PENALTY, OFFSIDE, CELEBRATION, HIGHLIGHT.

For each detected event, a composite `contextScore` is computed using a weighted formula:

```
finalScore = eventWeight * motionIntensity * temporalWeight * confidenceScore
```

Where `eventWeight` varies by type (GOAL = 10.0, SAVE = 7.0, TACKLE = 4.0, etc.) and `temporalWeight` gives a 1.5x boost to events in the final 20% of the match.

As each event is detected, the inference engine immediately sends `POST /callbacks/event` to the orchestrator. The orchestrator saves the event to PostgreSQL and pushes a `eventDetected` WebSocket event to the frontend, which displays a live notification to the user.

**Phase 3 — Gemini Commentary Generation** For each confirmed event, the inference engine calls the Google Gemini 2.0 Flash API with the event metadata (type, timestamp, motion score, match context, formation analysis from Phase 2). The prompt requests a 40-60 word broadcast-style commentary sentence.

The system also generates a 3-5 sentence overall match summary using Gemini after all individual event commentaries are written.

**Phase 4 — Highlight Reel Construction (TTS + FFmpeg)** The system selects the top-N highest-scoring non-overlapping events as highlight clips. For each clip:

1. The `tts_generate()` function is called to synthesize the event's commentary as audio.
2. The 3-tier TTS system attempts Kokoro-82M (Tier 1), falls back to edge-tts (Tier 2), and falls back to silent audio (Tier 3).
3. FFmpeg cuts the video clip from `startTime` to `endTime`, overlays scrolling text with the commentary, mixes in the TTS audio, crowd ambience, and background music, then renders a standalone `.mp4` clip.
4. All clips are concatenated into a single highlight reel `.mp4`.

**Phase 5 — Analytics Post-Processing** After the main loop completes, `generate_heatmap()` and `estimate_ball_speed()` run over the accumulated `track_frames` data. The heatmap is rendered as a PNG using OpenCV and saved to `uploads/`. The team color detector runs K-Means clustering on jersey RGB samples.

**Completion Callback** The inference engine sends `POST /callbacks/complete` to the orchestrator with:

- `highlightReelUrl` — the generated highlight reel file path
- `heatmapUrl` — the generated heatmap image file path
- `topSpeedKmh` — the estimated peak ball speed
- `teamColors` — the two detected jersey color RGB arrays
- `summary` — the Gemini match summary text

The orchestrator updates the Match record to `status: COMPLETED` and sends an `analysisComplete` WebSocket event to the frontend.

### What the Frontend Shows During Processing

The match detail page continuously receives WebSocket events:

- `analysisProgress` events update the circular progress indicator from 0% to 100%.
- `eventDetected` events cause a toast notification to appear briefly, showing the detected event type and timestamp.
- `analysisComplete` causes the page to reload its data and reveal all completed tabs.

---

## 8. Match Detail Page — Video Player

### Layout

The top section of the match detail page (`/matches/[id]`) contains a video player. The player renders the originally uploaded video file (not the highlight reel — that is separate in the Highlights tab).

### Event Timestamp Seeking

Below or beside the video player, the Top 5 Moments and the event list both show timestamps. Clicking a timestamp or event card calls `videoRef.current.currentTime = timestampSeconds` to seek the video player to that exact moment. This allows users to immediately watch the raw footage at the point where any detected event occurred.

### Video Player Controls

The video player uses the browser's native HTML5 `<video>` element with custom styling overlaid. Controls include: play/pause, volume, fullscreen, playback speed (0.5x, 1x, 1.5x, 2x).

---

## 9. Match Detail Page — Top 5 Moments

### What the User Sees

Directly below the video player is a horizontally scrollable row of 5 cards labeled "Top Moments." These represent the 5 events with the highest `finalScore`.

The cards are styled using a podium ranking theme:

- Rank 1: Gold styling
- Rank 2: Silver styling
- Rank 3: Bronze styling
- Ranks 4-5: Standard styling

Each card shows:

- The rank number.
- The event type (e.g., GOAL, SAVE).
- The timestamp formatted as `MM:SS`.
- The `finalScore` displayed out of 10.
- The first sentence of the Gemini-generated commentary.

Clicking a card seeks the video player to the event's timestamp.

---

## 10. Match Detail Page — Highlights Tab

### Content

The Highlights tab shows the generated highlight reel video and a list of all highlight clip cards.

At the top of the tab, the full highlight reel `.mp4` is embedded in a `<video>` player. This is the concatenated reel of all highlight clips with TTS commentary mixed in.

Below the reel, each individual highlight is shown as a card with:

- The event type.
- The clip start time and end time.
- The `score` (0-10).
- The full Gemini commentary text for that clip.
- The TTS voice tier that was used (Kokoro / edge-tts / Silent).

---

## 11. Match Detail Page — Events Tab

### Content

The Events tab is a chronological list of every detected event, not just the top 5 or the highlights. This gives users a complete picture of the match.

Each event row shows:

- Event type badge (color-coded).
- Timestamp.
- Confidence score (0.0 to 1.0, shown as a percentage).
- Final weighted score (0-10).
- The full Gemini commentary text.

The list is sorted by timestamp ascending (earliest events first).

### Live Event Streaming During Analysis

When the user is on the Events tab during active processing, new events appear in real time as the `eventDetected` WebSocket events arrive. Each new event animates in at the bottom of the list.

---

## 12. Match Detail Page — Analytics Tab

### Ball Speed

The first card in the Analytics tab shows the estimated peak ball speed in km/h. This is the 95th percentile of all ball speed measurements calculated from the YOLO tracking data (to suppress noise from outlier tracking errors). The value is displayed prominently with an amber glow style to draw attention.

The formula used:

```
delta_metres = sqrt((x2 - x1)^2 + (y2 - y1)^2) * real_pitch_scale
speed_kmh = (delta_metres / delta_time_seconds) * 3.6
```

### Team Colors

The second card shows the two automatically detected jersey colors as color swatches alongside the hex color codes. These are computed by the K-Means clustering algorithm (`_cluster_teams()`) which groups all detected players' median jersey RGB values into two clusters.

### Player Heatmap

The third card shows the player density heatmap as an image. This is a top-down diagram of a standard football pitch with a Gaussian-blurred color overlay showing where players from each team concentrated their movement during the match.

The heatmap image is fetched from `match.heatmapUrl` which serves the file from the `uploads/` directory.

The interpretation: areas of intense red/warm color indicate heavy player concentration. Cooler areas indicate less activity. The two team colors (from the jersey color detection) are used to color each team's density layer differently.

---

## 13. PDF Match Report Generation

### Triggering Report Generation

A "Download PDF Report" button is visible on the match detail page for completed matches. Clicking it triggers client-side PDF generation using `@react-pdf/renderer` (no server round-trip required).

### Report Contents

The generated PDF contains:

- A title page with the match name, analysis date, and Matcha-AI branding.
- A summary section with the Gemini-generated match narrative.
- A statistics summary: total events detected, highlight count, peak ball speed, match duration.
- A "Top 5 Moments" section with event type, timestamp, and full commentary for each.
- A complete event timeline table with all detected events sorted by timestamp.
- The analytics data (ball speed, team colors, note that heatmap image embedding requires the URL to be publicly accessible).

The PDF is downloaded immediately to the user's browser as `match-report-[matchId].pdf`.

---

## 14. Error States and Edge Cases

### Analysis Failed

If the inference engine encounters an unrecoverable error (corrupted video, out of memory, network timeout to Gemini), it sends `POST /callbacks/failed` to the orchestrator. The orchestrator updates the Match to `status: FAILED`. The frontend shows a red error state on the match card and on the detail page with the error message from the inference logs.

### WebSocket Disconnection

If the user closes the browser or loses their network connection during analysis, the analysis continues running uninterrupted in the Python process. When the user reopens the match detail page later, the current state is fetched via `GET /matches/:id`, which returns the latest `progress` and `status`. If the analysis has completed, all results are available immediately.

### Video Upload Failures

If the video upload fails (network error, file too large for the configured `JSON_BODY_LIMIT`), the frontend shows an error toast and the match record is not created.

### No Events Detected

If the video is too short, too low quality, or does not contain any recognizable sports action, the SoccerNet model and motion-peak fallback may detect zero events. In this case, the match will complete successfully but the Highlights tab and Top 5 Moments will show a "No highlights detected" empty state.

### Kokoro TTS Failure

If the `HF_TOKEN` environment variable is not set or the HuggingFace API is rate-limiting the account, Kokoro TTS will fail. The system automatically falls through to edge-tts. This is logged in the inference terminal as `Kokoro failed, falling back to edge-tts`. The user does not see this failure — the highlight reel is still generated with the fallback TTS voice.

---

## 15. Data Flow Summary

The following summarizes the end-to-end data flow for a complete analysis cycle:

```
User Browser

|

| 1. POST /matches (multipart video upload)

v
NestJS Orchestrator (port 4000)

|

| 2. Save video to uploads/

| 3. Create Match record (status=UPLOADED)

| 4. POST /analyze (async, fire-and-forget)

|

| 5. HTTP 201 Created { matchId } --> User Browser

v
Python Inference Engine (port 8000)

|

| Phase 1: YOLO tracking (frames 0..N)

|

|--> POST /callbacks/progress (every X frames) --> Orchestrator --> WS --> Browser

|

| Phase 2: Event detection

|

|--> POST /callbacks/event (each event) --> Orchestrator --> DB + WS --> Browser

|

| Phase 3: Gemini commentary

|

| Phase 4: TTS synthesis + FFmpeg highlight reel

|

| Phase 5: Heatmap + speed + team colors

|

| POST /callbacks/complete { highlightUrl, heatmapUrl, topSpeedKmh, teamColors, summary }

v
NestJS Orchestrator

|

| Update Match (status=COMPLETED, all fields populated)

| WS "analysisComplete" --> Browser

v
User Browser

|

| GET /matches/:id (refresh all data)

v
Full Match Detail Page (all tabs populated)
```
