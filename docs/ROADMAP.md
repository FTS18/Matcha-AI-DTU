# Project Roadmap

This document outlines the planned development trajectory for Matcha-AI-DTU. It serves as the official guide for contributors looking for meaningful tasks and for stakeholders tracking the project's progress. Items are organized by phase, priority, and estimated complexity. This is a living document and will be updated as features ship and priorities evolve.

---

## How to Use This Roadmap

If you are a contributor looking for something to work on:

1. Find a task that interests you and matches your skill level.
2. Check the Issues tab to see if a corresponding issue already exists.
3. If no issue exists, open one and reference this roadmap item.
4. Comment on the issue to express your interest before starting work.
5. Follow the contribution workflow in `CONTRIBUTING.md`.

Complexity labels used throughout this document:

- `[Beginner]` — No deep understanding of the codebase required. Good for first-time contributors.
- `[Intermediate]` — Requires familiarity with at least one service (Next.js, NestJS, or Python/FastAPI).
- `[Advanced]` — Requires deep understanding of multiple services and the data flow between them.
- `[Research]` — Involves exploring new libraries, models, or techniques before implementation.

---

## Current Status: Phase 1 and Phase 2 — Complete

The following capabilities are already fully implemented and production-ready:

**AI Analysis Pipeline (5 phases)**

- Phase 1a: YOLO v8 player and ball tracking across all video frames
- Phase 1b: Custom GoalDetectionEngine using Kalman filtering, homography projection, and a finite state machine
- Phase 2: SoccerNet-trained action recognition for 11 event types with motion-peak fallback
- Phase 3: Google Gemini 2.0 Flash commentary generation with late-game intensity boost
- Phase 4: 3-Tier TTS pipeline (Kokoro-82M, Microsoft edge-tts, FFmpeg silence fallback)
- Phase 5: Player heatmap generation, ball speed estimation, and team color detection via K-Means

**Phase 2 Enhancements (Complete)**

- Enhanced ball trajectory smoothing with sliding window averaging
- Context-aware commentary using team formation analysis
- Dynamic audio mixing based on match intensity scores
- Smart highlight selection with narrative grouping of related events

**Platform Infrastructure**

- JWT authentication via NestJS AuthModule
- Real-time progress tracking via Socket.IO WebSockets
- PostgreSQL database with full Prisma ORM schema and migrations
- Redis caching for job state management
- Monorepo architecture with Turborepo, shared packages, and strict TypeScript

**Frontend**

- Next.js 15 web application with full analysis dashboard
- Video player with event timestamp seeking
- Top 5 Moments podium with score ranking
- Intensity sparkline chart
- Highlights, Events, and Analytics tabs
- PDF match report generation via @react-pdf/renderer

---

## Phase 3 — Frontend and UX Polish

These tasks focus on improving the user-facing experience without requiring changes to the AI pipeline.

### 3.1 Dark and Light Theme Toggle `[Beginner]`

**Status**: Not started **Description**: The application currently only supports a dark theme. Implement a full light theme using the existing `@matcha/theme` design token system and add a toggle button in the Navbar. **Implementation Notes**:

- The `@matcha/theme` package already defines brand colors. Extend it to export a `lightTheme` object.
- Use CSS custom properties (`--color-background`, `--color-surface`, etc.) so that toggling the theme class on `<html>` switches the entire UI.
- Persist the user's preference in `localStorage`.
- The toggle button should be in `components/layout/Navbar.tsx`. **Files to Modify**: `packages/theme/src/index.ts`, `apps/web/app/globals.css`, `apps/web/components/layout/Navbar.tsx`

---

### 3.2 Skeleton Loading States for Match Cards `[Beginner]`

**Status**: Not started **Description**: When the match dashboard fetches data, the cards appear all at once after a delay, causing a jarring layout shift. Replace with animated skeleton placeholders that show while data is loading. **Implementation Notes**:

- Create a `MatchCardSkeleton` component in `apps/web/components/`.
- Use CSS `@keyframes` pulse animation (already defined in `globals.css`) rather than an external library.
- Render 6 skeleton cards while `isLoading` is true in the match dashboard. **Files to Modify**: `apps/web/components/match-dashboard.tsx`, `apps/web/app/globals.css`

---

### 3.3 Infinite Scroll for Match List `[Intermediate]`

**Status**: Not started **Description**: The match list currently loads all matches at once. For users with many matches, this is slow. Implement cursor-based pagination on the backend and infinite scroll on the frontend. **Implementation Notes**:

- Add a `cursor` and `limit` query parameter to `GET /matches` in the NestJS controller.
- Use Prisma cursor-based pagination: `findMany({ take: 20, cursor: { id: lastId }, skip: 1 })`.
- On the frontend, use the Intersection Observer API to detect when the user reaches the bottom of the list and trigger the next page fetch.
- Add the pagination parameters to the Zod contract in `packages/contracts`. **Files to Modify**: `services/orchestrator/src/matches/matches.controller.ts`, `services/orchestrator/src/matches/matches.service.ts`, `apps/web/components/match-dashboard.tsx`, `packages/contracts/src/match.ts`

---

### 3.4 Match Search and Advanced Filtering `[Intermediate]`

**Status**: Not started **Description**: Add a search bar to the match dashboard that filters matches by title, date range, and status. Currently only status filtering exists. **Implementation Notes**:

- Add `search`, `dateFrom`, `dateTo` query parameters to `GET /matches`.
- Use Prisma `where` clauses: `{ title: { contains: search, mode: 'insensitive' } }`.
- On the frontend, debounce the search input by 300ms before firing the API call to avoid excessive requests.
- Add date pickers using the existing `shadcn/ui` components already installed in the project. **Files to Modify**: `services/orchestrator/src/matches/matches.service.ts`, `apps/web/components/match-dashboard.tsx`

---

### 3.5 Drag-and-Drop Video Upload with Progress Bar `[Intermediate]`

**Status**: Not started **Description**: The current upload experience is a basic file input. Replace it with a drag-and-drop zone that shows a real-time upload progress bar. **Implementation Notes**:

- Implement using the native HTML5 Drag and Drop API and the `XMLHttpRequest` `upload.onprogress` event (or the `fetch` API with a `ReadableStream` wrapper).
- Show a circular or linear progress indicator during upload.
- Display the file name and estimated size before the user confirms upload.
- Handle file type validation (only accept `.mp4`, `.mov`, `.avi`) on the frontend before sending to the server. **Files to Modify**: `apps/web/app/page.tsx` or a new `UploadZone.tsx` component.

---

### 3.6 Responsive Mobile Layout Improvements `[Beginner]`

**Status**: Ongoing **Description**: Several components have known mobile layout issues on smaller viewports (below 375px). Audit and fix all mobile responsiveness problems. **Implementation Notes**:

- Test every page at 375px (iPhone SE), 390px (iPhone 14), and 414px (iPhone Pro Max) widths.
- Follow the existing conventions in `./CONTRIBUTING.md` section 9: use `text-[10px] sm:text-sm`, `size-4 sm:size-5`, and `hide-scrollbar` utilities.
- Pay special attention to the Analytics tab heatmap image, which can overflow on narrow screens. **Files to Modify**: Various files in `apps/web/`

---

### 3.7 Match Detail Page — Event Timeline Visualization `[Intermediate]`

**Status**: Not started **Description**: Replace the current flat event list with a visual timeline that shows events plotted along a horizontal match duration bar. Clicking an event on the timeline seeks the video player to that timestamp. **Implementation Notes**:

- Use SVG or an HTML Canvas element for the timeline bar.
- Events are represented as dots or icons positioned at `(timestamp / duration) * barWidth` pixels from the left.
- Color-code dots by event type (GOAL = gold, FOUL = red, SAVE = blue, etc.).
- On dot click, call the existing `seekTo(timestamp)` function already available on the video player ref. **Files to Modify**: `apps/web/app/matches/[id]/page.tsx`

---

### 3.8 PDF Match Report — Design and Content Enhancement `[Intermediate]`

**Status**: Partially implemented (basic PDF exists) **Description**: The current PDF report is functional but visually basic. Enhance it with the team color swatches, heatmap image, event timeline, and Matcha brand styling. **Implementation Notes**:

- The PDF is generated using `@react-pdf/renderer` in `apps/web/`.
- Add team color swatches as colored rectangles using the `teamColors` data from the match.
- Embed the heatmap image using `<Image>` from `@react-pdf/renderer` (requires the image URL to be publicly accessible).
- Add an event breakdown table with columns for timestamp, event type, confidence score, and commentary. **Files to Modify**: `apps/web/app/matches/[id]/page.tsx` or a dedicated PDF component file.

---

## Phase 4 — AI Pipeline Enhancements

These tasks require working with the Python inference engine in `services/inference/`.

### 4.1 Support for Additional Sports `[Research]` `[Advanced]`

**Status**: Not started **Description**: The current pipeline is optimized for soccer/football. Investigate and implement support for at least one additional sport — basketball is the recommended starting point due to available YOLO training data. **Implementation Notes**:

- Research: Identify a suitable pre-trained YOLO model for basketball player and ball detection (e.g., Roboflow Universe datasets).
- Add a `sport` field to the match upload schema in `packages/contracts` and the Prisma `Match` model.
- In `analysis.py`, route the pipeline to sport-specific event detection logic based on the `sport` field.
- Implement basketball-specific event types: `THREE_POINTER`, `DUNK`, `STEAL`, `BLOCK`, `REBOUND`.
- Update the `EventType` enum in `packages/database/prisma/schema.prisma`. **Files to Modify**: `services/inference/app/core/analysis.py`, `packages/database/prisma/schema.prisma`, `packages/contracts/src/match.ts`

---

### 4.2 Player Re-identification Across Frames `[Advanced]`

**Status**: Not started **Description**: Currently, YOLO assigns track IDs to players, but these IDs can reset or switch between frames, causing the heatmap and team assignment to be inaccurate for long videos. Implement a re-identification (Re-ID) system to maintain consistent player identity across the entire video. **Implementation Notes**:

- Research: Explore `torchreid` or `deep-sort-realtime` Python libraries.
- The core challenge is matching a player's track ID when they leave and re-enter the frame or when YOLO loses tracking momentarily.
- A simpler approach: use the player's last known position and jersey color (from `_dominant_colour()`) to re-assign track IDs when a new detection appears within a threshold distance. **Files to Modify**: `services/inference/app/core/analysis.py`, `services/inference/app/core/heatmap.py`

---

### 4.3 Gemini Vision — Frame-Level Commentary `[Advanced]`

**Status**: Not started **Description**: The current commentary is text-only, generated from event metadata. Upgrade it to use Gemini Vision by sending actual video frames to the model so it can describe what it literally sees rather than only what the event detector reports. **Implementation Notes**:

- At each detected event timestamp, extract the corresponding video frame using OpenCV (`cap.read()` at the target frame index).
- Encode the frame as a base64 JPEG string.
- Send the frame alongside the event metadata in the Gemini API call using the multimodal input format.
- Add error handling for cases where frame extraction fails (fall back to text-only commentary).
- Be mindful of Gemini API rate limits — implement a per-event delay or a rate-limiting queue. **Files to Modify**: `services/inference/app/core/analysis.py`

---

### 4.4 Confidence Threshold Calibration UI `[Intermediate]`

**Status**: Not started **Description**: The `CONFIG` dictionary in `analysis.py` controls detection sensitivity, but users currently have no way to adjust it without editing source code. Add a per-match configuration UI on the upload page where users can set sensitivity sliders. **Implementation Notes**:

- Add an `analysisConfig` JSON field to the match upload API payload.
- Pass this configuration from the orchestrator to the inference engine in the `POST /analyze` request.
- In `analysis.py`, merge user-provided config with the default `CONFIG` dict at the start of analysis.
- On the frontend, add a collapsible "Advanced Settings" panel on the upload form with sliders for `MOTION_PEAK_THRESHOLD` (0.2 to 0.8) and `HIGHLIGHT_COUNT` (3 to 10). **Files to Modify**: `services/inference/app/core/analysis.py`, `services/orchestrator/src/matches/matches.service.ts`, `apps/web/app/page.tsx`, `packages/contracts/src/match.ts`

---

### 4.5 Multi-Language Commentary `[Intermediate]`

**Status**: Not started **Description**: The current commentary is generated only in English. Add support for generating commentary in multiple languages (Spanish, French, Hindi, Portuguese, Arabic) by passing a language parameter to the Gemini prompt. **Implementation Notes**:

- Add a `commentaryLanguage` field to the match upload schema.
- Modify the Gemini prompt in `analysis.py` to include the target language: `"Generate commentary in {language}. Commentary should be..."`
- For TTS, update the language selection logic: `edge-tts` supports many language voices (e.g., `es-ES-AlvaroNeural` for Spanish). Map each supported language to an appropriate `edge-tts` voice identifier.
- Note that Kokoro-82M (Tier 1) currently only supports English. The system should automatically fall through to Tier 2 for non-English languages. **Files to Modify**: `services/inference/app/core/analysis.py`, `packages/contracts/src/match.ts`

---

### 4.6 Offside Detection Using Homography `[Advanced]` `[Research]`

**Status**: Not started **Description**: The existing goal detection engine already implements a homography transformation to project pixel coordinates onto real-world pitch coordinates. Extend this capability to detect offside situations by computing the position of attacking players relative to the last defender at the moment of a pass. **Implementation Notes**:

- Research: The existing homography pipeline in `goal_detection.py` maps frame pixel positions to pitch coordinates. Offside detection requires knowing which players are attackers, which are defenders, and the exact moment of a through-ball pass.
- Detect a "pass" event by looking for a sudden change in ball possession (direction reversal combined with player proximity to the ball).
- At the moment of the pass, find the second-to-last defender (the last defender is typically the goalkeeper) using their projected Y-coordinate on the pitch.
- Determine if any attacker's projected Y-coordinate is beyond the second-to-last defender's Y-coordinate. **Files to Modify**: `services/inference/app/core/goal_detection.py`, `services/inference/app/core/analysis.py`

---

### 4.7 Automated Video Compression and Format Validation `[Beginner]` `[Intermediate]`

**Status**: Partially implemented (size-based compression exists) **Description**: The current pre-compression only triggers above a file size threshold. Improve this by adding resolution validation (auto-downscale 4K videos to 1080p), frame rate normalization (standardize to 30fps), and codec validation (ensure H.264 encoding). **Implementation Notes**:

- Use FFprobe (`subprocess.run(['ffprobe', ...])`) to extract video metadata before processing.
- If resolution is above 1920x1080, add `-vf scale=1920:-2` to the FFmpeg pre-processing command.
- If frame rate exceeds 60fps, add `-r 30` to standardize.
- If the codec is not H.264, add `-c:v libx264` to re-encode. **Files to Modify**: `services/inference/app/core/analysis.py`

---

## Phase 5 — Infrastructure and DevOps

### 5.1 Docker Containerization of All Services `[Advanced]`

**Status**: Not started (Docker is currently only used for databases) **Description**: Containerize the NestJS orchestrator and Python inference engine in Docker so the entire stack can be started with a single `docker-compose up` command, without requiring local Node.js or Python installations. **Implementation Notes**:

- Create `services/orchestrator/Dockerfile` using a `node:20-alpine` base image.
- Create `services/inference/Dockerfile` using a `python:3.11-slim` base image. This Dockerfile must install all system dependencies including `ffmpeg`, `libgl1`, and `libglib2.0-0` (required by OpenCV).
- Update `docker-compose.yml` to add `orchestrator` and `inference` services with appropriate build contexts, volume mounts for `uploads/`, and environment variable passthrough.
- The inference container will be large (~2-4GB) due to PyTorch. Document this clearly. **Files to Modify**: `docker-compose.yml`, new `services/orchestrator/Dockerfile`, new `services/inference/Dockerfile`

---

### 5.2 GitHub Actions CI Pipeline `[Intermediate]`

**Status**: Not started **Description**: Add automated testing and linting on every pull request using GitHub Actions so that broken code cannot be merged. **Implementation Notes**:

- Create `.github/workflows/ci.yml`.
- The workflow should trigger on `push` to `dev` and on all `pull_request` events targeting `dev` or `main`.
- Steps: checkout code, install Node.js 20, run `npm install`, run `npx turbo run lint`, run `npx turbo run build`.
- Add a separate job for Python: install Python 3.11, install `requirements.txt`, run `python -m pytest services/inference/tests/` if test files exist.
- Add status badges to the README. **Files to Create**: `.github/workflows/ci.yml`

---

### 5.3 End-to-End Test Suite `[Intermediate]` `[Advanced]`

**Status**: Not started **Description**: There are currently no automated tests in the project. Add unit tests for the most critical logic and at least one end-to-end integration test for the upload-to-analysis pipeline. **Implementation Notes**:

- Python unit tests: Test `calculate_dynamic_audio_volumes()`, `smooth_ball_trajectory()`, `estimate_ball_speed()`, and `_cluster_teams()` in isolation using `pytest`. Use mock YOLO tracking data.
- NestJS unit tests: Test the `MatchesService` using Jest with a mocked Prisma client (`jest.mock('@prisma/client')`).
- E2E test: Use a short test video (5-10 seconds of synthesized frames) and run the full pipeline, asserting that the match status reaches `COMPLETED` and at least one event is returned. **Files to Create**: `services/inference/tests/`, `services/orchestrator/src/matches/matches.service.spec.ts`

---

### 5.4 MinIO Integration for Scalable File Storage `[Advanced]`

**Status**: Not started (MinIO is in docker-compose but not used) **Description**: The current file system uses a local `uploads/` directory, which does not scale and cannot be shared between containerized services. Integrate MinIO (already defined in `docker-compose.yml`) as the object storage backend for all uploaded videos, generated heatmaps, and highlight reels. **Implementation Notes**:

- Install the MinIO JavaScript SDK (`minio`) in `services/orchestrator`.
- Create a `StorageModule` in NestJS that wraps the MinIO client with `uploadFile(bucket, key, buffer)` and `getSignedUrl(bucket, key)` methods.
- On video upload (`POST /matches`), stream the incoming file to MinIO instead of writing to `uploads/`.
- Pass the MinIO object key (not a local path) to the inference engine. The inference engine will need to download the file from MinIO, process it, and upload results back to MinIO.
- Install the MinIO Python SDK (`minio`) in `services/inference/requirements.txt`. **Files to Modify**: `services/orchestrator/src/`, `services/inference/app/core/analysis.py`, `docker-compose.yml`

---

## Phase 6 — New Product Features

### 6.1 Live Match Streaming Analysis `[Advanced]` `[Research]`

**Status**: Research phase **Description**: Instead of only analyzing pre-recorded videos, investigate supporting live video stream analysis where the AI pipeline processes frames in real time as a match is being broadcast. **Implementation Notes**:

- Research: OpenCV can read from RTSP streams (`cv2.VideoCapture('rtsp://...')`). The challenge is processing frames faster than they arrive.
- The pipeline would need to run in a significantly reduced mode (fewer YOLO detections per second, simplified event detection) to achieve near-real-time performance.
- This is a significant architectural change. It likely requires the inference service to maintain a persistent connection to the stream rather than processing a file.

---

### 6.2 Coach and Team Dashboard `[Advanced]`

**Status**: Not started **Description**: Add a "Team" mode where multiple matches can be grouped into a season or tournament. Aggregate analytics across all matches (e.g., total goals, average ball speed per game, player heatmap across the full season) and present them in a dedicated coaching dashboard. **Implementation Notes**:

- Add `Team` and `Season` Prisma models with relationships to `Match`.
- Create a new `teams` module in the NestJS orchestrator.
- Create a new `/teams` and `/seasons` frontend route in Next.js.
- The season-level heatmap would be generated by overlaying multiple per-match heatmaps using NumPy array addition followed by normalization.

---

### 6.3 Expo React Native Mobile App `[Advanced]`

**Status**: Scaffolded (apps/mobile/ directory exists but is empty) **Description**: Build out the Expo mobile application that was planned in the original architecture. The `@matcha/shared` package already exports the API client and TypeScript interfaces that the mobile app should use. **Implementation Notes**:

- Initialize the Expo app in `apps/mobile/` using `npx create-expo-app`.
- Implement authentication screens (login, register) that call the same orchestrator endpoints as the web app.
- Implement a match list screen and a match detail screen with a video player (`expo-av`).
- Use `@matcha/shared` for all API calls and type definitions — do not duplicate these in the mobile app.

---

### 6.4 Match Sharing and Social Features `[Intermediate]`

**Status**: Not started **Description**: Allow users to share a public link to a completed match analysis. Implement a shareable read-only view of the match detail page that does not require authentication. **Implementation Notes**:

- Add a `shareToken` UUID field to the `Match` Prisma model.
- Add a `POST /matches/:id/share` endpoint that generates and returns the `shareToken`.
- Add a `GET /shared/:shareToken` public endpoint that returns match data without requiring a JWT.
- Create a `apps/web/app/shared/[token]/page.tsx` Next.js route that renders the read-only match view.

---

### 6.5 User Profile and Match History Dashboard `[Intermediate]`

**Status**: Not started **Description**: Add a user profile page that shows account information, total matches analyzed, cumulative stats (total goals detected, average match duration processed), and account settings. **Implementation Notes**:

- Add aggregate query endpoints to the NestJS `users` module.
- Use Prisma aggregate functions: `_count`, `_avg`, `_sum` on the `Match` model grouped by `userId`.
- Create `apps/web/app/profile/page.tsx`.

---

### 6.6 Notification System for Long Analysis Jobs `[Intermediate]`

**Status**: Not started **Description**: Video analysis can take 5-15 minutes for long matches. Add browser push notifications and optional email notifications that alert the user when their analysis completes. **Implementation Notes**:

- Browser push: Use the Web Push API (`web-push` npm package in the orchestrator). The frontend subscribes via `navigator.serviceWorker.ready.pushManager.subscribe()`. Store the subscription object per user in the database.
- Email: Integrate Nodemailer or Resend in the NestJS orchestrator. Send an email template when `match.status` transitions to `COMPLETED`.
- The notification should contain the match title and a direct link to the match detail page.

---

## Phase 7 — Documentation Improvements

### 7.1 API Documentation with Swagger UI `[Beginner]`

**Status**: Not started **Description**: The NestJS orchestrator has Swagger integration available via `@nestjs/swagger` but it is not currently enabled or configured. Enable it so the API is self-documenting at `/api/docs`. **Implementation Notes**:

- Install `@nestjs/swagger` and `swagger-ui-express` if not already present.
- Add `SwaggerModule.setup('api/docs', app, document)` in `services/orchestrator/src/main.ts`.
- Add `@ApiProperty()` decorators to all DTOs.
- Add `@ApiOperation()` and `@ApiResponse()` decorators to all controller endpoints. **Files to Modify**: `services/orchestrator/src/main.ts`, all DTO files and controller files.

---

### 7.2 Video Tutorial and Demo Recording `[Beginner]`

**Status**: Not started **Description**: Record a short screen-capture video (5-10 minutes) demonstrating the full user journey: uploading a video, watching the real-time analysis progress, reviewing the generated highlights, and exploring the analytics tab. Embed this in the README.

---

### 7.3 Storybook Component Library `[Intermediate]`

**Status**: Not started **Description**: The `@matcha/ui` package contains shared React components. Set up Storybook so contributors can browse, test, and develop these components in isolation without running the full stack. **Implementation Notes**:

- Initialize Storybook in `packages/ui/` using `npx storybook@latest init`.
- Write story files for each component: `ScoreBadge.stories.tsx`, `VideoPlayer.stories.tsx`, `CopyButton.stories.tsx`.
- Add a `storybook` script to `packages/ui/package.json`.

---

## Known Issues and Bug Backlog

The following are known bugs that need fixing. They are all good candidates for first-time contributors.

| Issue | Severity | Description | Likely File |
| --- | --- | --- | --- |
| Next.js workspace root warning | Low | Build warns about multiple lockfiles causing incorrect workspace root detection. Add `outputFileTracingRoot` to `next.config.js`. | `apps/web/next.config.js` |
| CORS_ORIGIN port mismatch | Medium | If Next.js starts on port 3001 (because 3000 is busy), the orchestrator CORS will block requests. Add auto-detection or clear documentation. | `services/orchestrator/.env` |
| Prisma generate EPERM on Windows | Medium | Running `npx prisma generate` while the orchestrator is running fails on Windows because the DLL is locked. Document the workaround (stop the service first). | `./SETUP.md` |
| ESLint Next.js plugin warning | Low | Build warns that the Next.js ESLint plugin is not configured. Add `plugin:@next/next/recommended` to `.eslintrc`. | `apps/web/.eslintrc.js` |
| MinIO defined but not used | Low | The `docker-compose.yml` starts a MinIO container that nothing connects to. Either implement Phase 5.4 or document MinIO's intended future role. | `docker-compose.yml` |

---

## Version History

| Version | Date | Description |
| --- | --- | --- |
| 1.0 | February 2026 | Initial release. Full 5-phase AI pipeline, NestJS orchestrator, Next.js frontend, JWT auth. |
| 1.1 | February 2026 | Phase 5 analytics: player heatmap, ball speed estimation, team color detection. |
| 1.2 | February 2026 | Phase 2 enhancements: ball trajectory smoothing, context-aware commentary, dynamic audio mixing, smart highlight selection with narrative grouping. |
| 1.3 | May 2026 | GSSOC'26 preparation: comprehensive documentation suite, GitHub community files, contributor templates, roadmap. |

---

This roadmap is maintained by the Matcha-AI-DTU core team. For questions about any roadmap item, open a GitHub Discussion or comment on the relevant Issue.
