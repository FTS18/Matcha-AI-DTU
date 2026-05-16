# Contributing to Matcha-AI-DTU

Thank you for considering a contribution to Matcha-AI-DTU. This project is a complex monorepo blending Next.js, NestJS, and a Python computer vision and LLM pipeline. Every contribution — from a one-line documentation fix to a full feature implementation — is valued.

---

## New to Open Source? Start Here

If you have never contributed to an open-source project before, read `ONBOARDING.md` before this file. It explains forking, branching, committing, and pull requests in plain English without assuming any prior experience.

Not familiar with a term in this file? Check `ARCHITECTURE.md#terminology--concepts` for plain-English definitions of every technical term used in this project.

---

## Good First Issues

The following are specific, well-scoped tasks that are appropriate for contributors who are new to this codebase. All of them are labeled `good first issue` on GitHub. Open the corresponding issue, comment to claim it, and wait for assignment before starting work.

| Issue Title | Area | What You Will Learn |
| --- | --- | --- |
| Add skeleton loading states to the match dashboard grid | Frontend (Next.js) | React components, loading state management |
| Add a "Copy Match ID" button to the match detail page | Frontend (Next.js) | Clipboard API, React state |
| Display the match duration on each match card | Frontend (Next.js) | Date formatting, NestJS API response |
| Add a "No highlights detected" empty state to the Highlights tab | Frontend (Next.js) | Conditional rendering, UI components |
| Add a character counter to the match title input field | Frontend (Next.js) | Controlled inputs, inline validation |
| Add a Windows-specific note to the venv activation step in `SETUP.md` | Documentation | Markdown, Windows path syntax |
| Expand the FAQ with answers to the top 5 Docker setup questions | Documentation | Technical writing, Docker basics |
| Add a `FAQ.md` entry for the "analysis stuck at 0%" error | Documentation | Markdown, understanding the callback flow |
| Add inline JSDoc comments to the `MatchesService` methods | Backend (NestJS) | TypeScript, NestJS service pattern |
| Add input length validation for the match title field in the NestJS controller | Backend (NestJS) | NestJS validation pipes, class-validator |

If none of these appeal to you, browse `ROADMAP.md` for all planned features tagged `[Beginner]`.

---

## 1. Where to Start?

If you are looking for a place to contribute:

1. Check the **Issues** tab on GitHub and filter by `good first issue` or `help wanted`.
2. Browse `ROADMAP.md` for planned features. All items are labeled by difficulty.
3. Read `ARCHITECTURE.md` to understand the overarching data flow before jumping in.
4. Read `../services/inference/AI_PIPELINE.md` if you are working on the Python inference pipeline.

## 2. Fork & Clone

1. Fork the repository on GitHub.
2. Clone your forked repo to your local machine.
3. Add the upstream remote: `git remote add upstream [repository url]`

## 3. Branch Naming Convention

We strictly follow a structured branch naming paradigm to keep CI/CD pipelines happy:

- `feature/issue-number-short-description` (e.g. `feature/42-add-tts-voices`)
- `bugfix/issue-number-short-description` (e.g. `bugfix/99-fix-websocket-crash`)
- `docs/short-description`
- `refactor/component-name`

## 4. Commit Message Standard

We utilize **Conventional Commits**:

- `feat: [description]` for new features.
- `fix: [description]` for bug fixes.
- `docs: [description]` for documentation alterations.
- `style: [description]` styling/formatting (prettier, eslint changes).
- `refactor: [description]` refactoring existing logic without breaking API boundaries.
- `chore: [description]` updating dependencies or CI pipelines.

_Example_: `feat: integrate YOLOv8n object detection model for goal calibration`

## 5. Development Workflow Recommendations

- **Always run linters** before opening a PR. Ensure `npm run lint` and `npm run format` pass successfully in the root directory.
- For Python code in `services/inference`, please use appropriate type hinting for FastAPI Pydantic models. We try to keep Python styling PEP 8 compliant.

## 6. Pull Requests

1. All Development takes place on the `dev` branch. Pull requests should target `dev`, not `main`.
2. When creating a PR, the `.github/PULL_REQUEST_TEMPLATE.md` will automatically populate. **Please fill it out fully.**
3. Reference the Issue number utilizing closing terminology (e.g. "Closes #42").
4. Wait for Code Reviews. At least 1 approval is needed before merging.

---

## 7. Local Development Setup (Full Stack)

Follow these steps to run the full Matcha AI stack locally on Windows.

### Prerequisites

- **Node.js** >= 18 (LTS recommended)
- **Python** >= 3.11 (3.14 compatible)
- **PostgreSQL** running on port `5433` (or Docker)
- **FFmpeg** in your system `PATH`
- A **Gemini API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
- A **HuggingFace token** (free) from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — for Kokoro TTS

### Step 1: Install Node Dependencies

```bash
npm install
```

### Step 2: Set Up the Orchestrator & Database

All database logic is centralized in `@matcha/database`.

```bash
# Generate the shared Prisma client
npx turbo run generate

# Deploy migrations to your local Docker Postgres
npx turbo run db:migrate
```

### Step 3: Set Up the Inference Service

```bash
cd services/inference
python -m venv venv
venv\Scripts\activate

# Windows
# source venv/bin/activate

# Linux/macOS
pip install -r requirements.txt
```

Create `services/inference/.env`:

```
GEMINI_API_KEY=your_gemini_key_here
HF_TOKEN=hf_your_huggingface_token
ORCHESTRATOR_URL=http://localhost:4000
```

### Step 4: Run the Full Stack

```bash
# From the monorepo root
npx turbo run dev
```

This starts:

- `apps/web` on `http://localhost:3000`
- `services/orchestrator` on `http://localhost:4000`
- `services/inference` on `http://localhost:8000`

---

## 8. Working on the Python Inference Pipeline

The inference service lives entirely in `services/inference/`. Key files:

| File                             | Purpose                                                       |
| -------------------------------- | ------------------------------------------------------------- |
| `app/core/analysis.py`           | **Main pipeline** — 5-phase video analysis orchestrator       |
| `app/core/heatmap.py`            | Phase 5 analytics: player heatmap PNG + ball speed estimation |
| `app/core/goal_detection.py`     | Kalman filter + Homography + FSM goal detection engine        |
| `app/core/soccernet_detector.py` | SoccerNet-based football event detector                       |
| `app/api/routes.py`              | FastAPI route handlers                                        |
| `main.py`                        | FastAPI app entrypoint                                        |

### CONFIG dict

Most tunable parameters are in the `CONFIG` dict at the top of `analysis.py`. Edit there to adjust sensitivity:

```python
CONFIG = {
    "MOTION_PEAK_THRESHOLD": 0.45,  # Raise to detect fewer, more dramatic highlights
    "HIGHLIGHT_COUNT": 5,           # Number of clips in the highlight reel
    "COMPRESS_SIZE_THRESHOLD_MB": 100,  # Pre-compress videos larger than this
}
```

### Adding a New Event Type

1. Add the new type to the `EventType` enum in `packages/database/prisma/schema.prisma`
2. Add a weight to `EVENT_WEIGHTS` in `services/inference/app/core/analysis.py`
3. Update the Zod schema in `packages/contracts/src/match.ts`
4. Run `npx turbo run db:migrate` to update the database
5. Run `npx turbo run generate` to update the Prisma client across the monorepo

### Adding a New Analytics Metric

1. Add the computation logic to `app/core/heatmap.py` (or create a new module in `app/core/`)
2. Add the result to the `payload` dict in `analyze_video()` in `analysis.py`
3. Update the `AnalysisPayload` interface and `MatchDetail` interface located inside `packages/shared/src/types.ts`.
4. Add the field to the Prisma `Match` model in `schema.prisma` and run a migration
5. Display it in the "Analytics" tab in `apps/web/app/matches/[id]/page.tsx`

### TTS Voice Configuration

The 3-tier TTS system selects quality automatically. To change voices:

```python
# In services/inference/app/core/analysis.py
_KOKORO_MODEL   = "hexgrad/Kokoro-82M"   # Change to any HF TTS model
_KOKORO_VOICE   = "af_sky"               # Kokoro voice ID
_EDGE_TTS_VOICE = "en-GB-RyanNeural"     # edge-tts voice name
```

Available Kokoro voice IDs: `af_sky`, `af_bella`, `am_adam`, `am_michael`, `bf_emma`, `bm_george`, `bm_lewis`

---

## 9. Working on the Frontend (Next.js)

The frontend lives in `apps/web/`. Key locations:

| File/Directory                   | Purpose                                                  |
| -------------------------------- | -------------------------------------------------------- |
| `app/page.tsx`                   | Hero landing page                                        |
| `app/matches/page.tsx`           | Match dashboard with filter tabs                         |
| `app/matches/[id]/page.tsx`      | Match detail page (video, events, highlights, analytics) |
| `components/match-dashboard.tsx` | Reusable match card grid component                       |
| `components/layout/Navbar.tsx`   | Global navigation bar                                    |
| `app/globals.css`                | Global CSS, design tokens, utility classes               |

### Adding a section to the Analytics Tab

The Analytics tab is in `apps/web/app/matches/[id]/page.tsx`. To add a new metric:

1. Ensure the orchestrator returns it in the API response.
2. Update the Zod schema and TypeScript interfaces in **`packages/contracts`** and **`packages/shared`**.
3. Add a new UI card in the React component.

### Monorepo Best Practices

- **Shared Package**: Avoid defining types, constants, or WebSocket event strings locally within `apps/web` or `services/orchestrator`. Always export them from `packages/shared` so the mobile app and backend remain strictly synchronized.

### Mobile Responsiveness Guidelines

- Always use `text-[10px] sm:text-sm` for text that appears in filter tabs or compact layouts
- Always use `size-4 sm:size-5` for icons that appear in headings
- Add `hide-scrollbar` class to any horizontally scrolling container (`<div>`)
- Test on at least 375px viewport width (iPhone SE size) before opening a PR

---

## 10. Environment Variables Cheat Sheet

Never commit `.env` files. All secrets stay local.

```bash
# services/orchestrator/.env
DATABASE_URL="postgresql://user:pass@localhost:5433/matcha_db?schema=public"
HF_TOKEN=hf_your_token_here
CORS_ORIGIN=http://localhost:3000
INFERENCE_URL=http://localhost:8000
PORT=4000

# services/inference/.env

GEMINI_API_KEY=your_google_ai_studio_key
HF_TOKEN=hf_your_token_here
ORCHESTRATOR_URL=http://localhost:4000
```

---

Thank you! Let's build the best sports ML platform.

---

## Maintenance & Security Policy

This section outlines the current architectural decisions regarding framework upgrades and security maintenance for the Matcha-AI-DTU project.

### Current Architectural Status

As of May 2026, the project has undergone a comprehensive stabilization and hardening phase. To ensure maximum reliability and performance of the core AI pipeline and visual reporting features, we have made the strategic decision to pin the following major versions:

| Component | Current Version | Maintenance Strategy |
| :--- | :--- | :--- |
| **Frontend Framework** | Next.js 14.2.x | Pinned (Stability over feature-parity) |
| **UI Runtime** | React 18.2.0 | Pinned (Peer dependency compatibility) |
| **Backend Framework** | NestJS 11.x | Pinned (Stability) |
| **Database ORM** | Prisma 5.22.x | Pinned (Environment compatibility) |

### Deferred Upgrades

The following upgrades are **NOT** in the immediate plan and are intentionally deferred until further notice:

#### 1. Next.js 15/16 & React 19
While Next.js 16 is available, upgrading to it would force a migration to React 19.
*   **Rationale**: React 19 introduces breaking changes to the internal component life-cycle and type-definitions that are currently incompatible with critical third-party libraries used in this project, specifically `@react-pdf/renderer` (for match reports) and `react-dropzone` (for video uploads).
*   **Constraint**: Until these libraries provide stable, production-ready support for React 19, we will remain on the Next.js 14.2.x branch.

#### 2. NestJS 11
Upgrading to NestJS 11 is deferred to maintain backend stability and avoid a full-scale refactor of decorators and module architectures.

### Security Posture

We maintain a "Green" status on our CI/CD pipeline and have addressed all critical local build and linting failures.

#### Known Vulnerabilities
You may notice **5** High/Moderate vulnerabilities reported by `npm audit` (specifically regarding `next` and `postcss`). 
*   **Status**: These are known and documented. 
*   **Reason**: These vulnerabilities exist within the core source code of the Next.js 14 line and its required sub-dependencies. Since we are already on the latest available patch for the 14.x branch (`14.2.35`), these cannot be resolved without a major upgrade to Next.js 15/16.
*   **Mitigation**: We have prioritized **Functional Stability** and **Build Correctness**. The security risk is managed by ensuring the application environment is hardened and only necessary ports are exposed in production.

### Optimizations (May 2026)

The following optimizations were implemented to ensure production readiness:

1. **Dependency Hoisting**: Common tools (TypeScript, ESLint, Prettier, Prisma) were moved to the root to reduce redundancy and install times.
2. **Docker Multi-Stage Builds**: All service Dockerfiles now use multi-stage builds and `turbo prune` to create minimal, secure images.
3. **Next.js Standalone Mode**: The web application is configured for `standalone` output, reducing image size by up to 90%.
4. **Graceful Shutdown**: The Orchestrator now implements NestJS shutdown hooks to properly close database and Redis connections on SIGTERM.
