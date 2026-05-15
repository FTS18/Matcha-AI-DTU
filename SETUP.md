# Matcha AI — Complete Setup Guide (Beginner Friendly)

> This guide walks you through **every single step** needed to get the **entire Matcha AI stack running** on your local machine — from scratch. It is written for someone who knows basic terminal usage but may not be familiar with monorepos, Docker, Python virtual environments, or Node.js workspaces. Every command is explained so you understand _why_ you are running it, not just _what_ to type.

---

## 📖 Table of Contents

1. [What is this project?](#what-is-this-project)
2. [System Requirements & Prerequisites](#-system-requirements--prerequisites)
3. [Step 1: Clone the Repository](#-step-1-clone-the-repository)
4. [Step 2: Start the Infrastructure (Docker)](#-step-2-start-the-infrastructure-docker)
5. [Step 3: Set Up Node.js & Install Dependencies](#-step-3-set-up-nodejs--install-dependencies)
6. [Step 4: Set Up the Python AI Inference Service](#-step-4-set-up-the-python-ai-inference-service)
7. [Step 5: Configure Environment Variables](#-step-5-configure-environment-variables)
8. [Step 6: Run the Full Stack](#-step-6-run-the-full-stack)
9. [Step 7: Verify Everything is Working](#-step-7-verify-everything-is-working)
10. [Troubleshooting Common Errors](#-troubleshooting-common-errors)

---

## What is this project?

Matcha AI is a **monorepo** — a single Git repository that contains multiple services that work together:

| Service              | Technology          | What it does                                                |
| :------------------- | :------------------ | :---------------------------------------------------------- |
| **Web Frontend**     | Next.js (React)     | The UI that users interact with in their browser.           |
| **Orchestrator**     | NestJS (Node.js)    | The backend API server — handles auth, matches, uploads.    |
| **Inference Engine** | FastAPI (Python)    | The AI brain — runs YOLO, Gemini, and generates highlights. |
| **Database**         | PostgreSQL (Docker) | Stores users, matches, and events persistently.             |
| **Cache**            | Redis (Docker)      | Fast in-memory store for tracking active analysis jobs.     |

All of them need to be running simultaneously for the app to work correctly.

---

## 💻 System Requirements & Prerequisites

Before you start, install these tools on your machine. Check if you already have them by running the commands shown.

### 1. Node.js (v18 or higher)

Node.js is the JavaScript runtime that powers the frontend and the backend (Orchestrator).

```bash
# Check if you already have it:
node --version
# Should print something like: v20.11.0

# Check npm (comes with Node):
npm --version
# Should print something like: 10.2.4
```

If not installed, download from: **https://nodejs.org** (choose the LTS version).

---

### 2. Python 3.9 or higher

Python is used to run the AI Inference engine (YOLO video analysis, Gemini AI, heatmap generation).

```bash
# Check if you have it:
python3 --version
# Should print: Python 3.9.x or higher
```

> **macOS Note**: macOS ships with Python 3.9 from Xcode Command Line Tools. If you see `3.8` or lower, install a newer version via [python.org](https://www.python.org/downloads/) or [Homebrew](https://brew.sh/) (`brew install python@3.11`).
>
> **Important**: The project uses Python 3.9 compatible syntax. Do NOT use 3.7 or 3.8.

---

### 3. Docker Desktop

Docker runs PostgreSQL and Redis in isolated containers — you don't need to install these databases manually, Docker handles everything.

```bash
# Check if Docker is installed:
docker --version
# Should print: Docker version 24.x.x or similar
```

If not installed, download from: **https://www.docker.com/products/docker-desktop/**

> After installing Docker Desktop, **make sure it's running** (look for the Docker whale icon in your system tray/menu bar). Docker must be open and active before you run any `docker` commands.

---

### 4. FFmpeg

FFmpeg is used by the AI Inference engine to cut video clips, generate highlight reels, and process audio for commentary.

```bash
# Check if you have it:
ffmpeg -version
```

If not installed:

- **macOS**: `brew install ffmpeg`
- **Ubuntu/Linux**: `sudo apt install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html and add to PATH.

---

### 5. Git

You'll need Git to clone the repository.

```bash
git --version
```

If not installed: https://git-scm.com/downloads

---

## 📁 Step 1: Clone the Repository

Open your terminal and navigate to a folder where you want to put the project (e.g., your home directory or a `Projects` folder), then clone:

```bash
# Clone the repository
git clone https://github.com/YOUR-ORG/Matcha-AI-DTU.git

# Navigate into the project folder
cd Matcha-AI-DTU
```

> From this point on, **all commands assume you are inside the `Matcha-AI-DTU` folder** unless stated otherwise.

---

## 🐳 Step 2: Start the Infrastructure (Docker)

The project needs a running PostgreSQL database and a Redis cache. Docker Compose starts both with a single command.

```bash
# Start all infrastructure services in the background (-d = "detached" mode)
docker-compose up -d --build
```

**What this does:**

- Downloads the official `postgres:16-alpine` and `redis:7-alpine` Docker images (only on first run, takes a minute).
- Starts a PostgreSQL server accessible at `localhost:5433`.
- Starts a Redis server accessible at `localhost:6380`.
- The `--build` flag ensures any custom Dockerfile steps are executed.

**Verify they're running:**

```bash
docker ps
```

You should see two containers running: `matcha_postgres` and `matcha_redis`. Both should show `Up` in the `STATUS` column.

> **Troubleshooting**: If Docker says "Cannot connect to the Docker daemon", Docker Desktop is not running. Open the Docker Desktop app first.

---

## 📦 Step 3: Set Up Node.js & Install Dependencies

This project is a **monorepo** managed by **Turborepo**. All JavaScript packages (frontend, orchestrator, and shared libraries like `@matcha/ui`, `@matcha/env`, etc.) are installed and linked together with a single command from the root.

### 3.1 — Create the `uploads` directory

The Orchestrator needs a folder to store uploaded video files. Create it manually:

```bash
mkdir uploads
```

> This folder is in `.gitignore` so it won't be committed to Git — you have to create it yourself on every fresh clone.

### 3.2 — Install all Node.js dependencies

```bash
npm install
```

**What this does:**

- Reads `package.json` in the root and in every `apps/*`, `packages/*`, and `services/*` folder.
- Downloads and installs all required Node.js packages into `node_modules`.
- **Links monorepo packages** — so `@matcha/ui`, `@matcha/env`, `@matcha/shared`, etc., are all available to each service without publishing to npm.

This might take 1-2 minutes on the first run.

### 3.3 — Build the shared packages

Shared TypeScript packages (`@matcha/env`, `@matcha/shared`, `@matcha/ui`, etc.) need to be compiled to JavaScript before the apps can use them.

```bash
npx turbo run build
```

**What this does:**

- Turborepo figures out which packages depend on which (e.g., `web` depends on `@matcha/env`, so `@matcha/env` must be built first).
- Compiles TypeScript source files (`src/*.ts`) into JavaScript (`dist/*.js`) for every shared package.
- Results are cached — if nothing changed, subsequent builds are instant.

> **Common error**: If you see `Cannot find module '@t3-oss/env-core'` during build, it means the `@matcha/env` package's dependencies weren't properly resolved. Run `npm install` again from the root and then retry `npx turbo run build`.

### 3.4 — Run Database migrations

The Orchestrator uses **Prisma ORM** to manage the database schema. You need to apply the schema to your PostgreSQL database once:

```bash
cd services/orchestrator
npx prisma migrate deploy
cd ../..
```

If you're setting up for the first time and the database is fresh, you might need:

```bash
cd services/orchestrator
npx prisma migrate dev
cd ../..
```

> **What's the difference?** `migrate dev` creates the migration file AND applies it (for development). `migrate deploy` only applies existing migration files (for production or CI). Use `migrate dev` on a fresh local setup.

---

## 🧠 Step 4: Set Up the Python AI Inference Service

The AI engine is a Python service that runs separately. It has its own dependencies (PyTorch, OpenCV, Ultralytics YOLO, etc.) that need to be installed in an **isolated virtual environment** so they don't conflict with system Python packages.

### 4.1 — Navigate to the inference directory

```bash
cd services/inference
```

### 4.2 — Create a Python virtual environment

A virtual environment is like a "clean room" for Python — it installs packages only for this project without touching your system Python.

```bash
python3 -m venv venv
```

This creates a folder called `venv/` inside `services/inference/`. You'll only need to do this once.

### 4.3 — Activate the virtual environment

Every time you open a new terminal and want to work with the Inference service, you must activate the venv:

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
.\venv\Scripts\activate
```

> You'll know it's activated when you see `(venv)` at the beginning of your terminal prompt, like: `(venv) keshav@mac services/inference %`

### 4.4 — Upgrade pip

`pip` is Python's package installer. It's good practice to upgrade it before installing packages:

```bash
pip install --upgrade pip
```

### 4.5 — Install Python dependencies

```bash
pip install -r requirements.txt
```

This reads `requirements.txt` and installs everything: FastAPI, Uvicorn (the HTTP server), OpenCV, NumPy, PyTorch, Ultralytics YOLO, edge-tts, and more. This can take **5-10 minutes** on the first run as it downloads large packages like PyTorch.

> **GPU Support (Optional)**: If you have an NVIDIA GPU with CUDA 12.4, you'll get much faster video analysis. In that case, instead of the above, install PyTorch with CUDA support first:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
> pip install -r requirements.txt
> ```

### 4.6 — Go back to the project root

```bash
cd ../..
```

---

## 🔐 Step 5: Configure Environment Variables

Environment variables are secret configuration values (like API keys and database passwords) that are kept out of the git repository for security reasons. You must create these files manually.

### 5.1 — Orchestrator environment file

Create the file `services/orchestrator/.env`:

```bash
# On macOS/Linux:
cp services/orchestrator/.env.example services/orchestrator/.env
```

Then open `services/orchestrator/.env` in your text editor and fill in the values:

```env
# ── Server ──────────────────────────────────────────────────
PORT=4000
REQUEST_TIMEOUT=30000
JSON_BODY_LIMIT=100mb
URLENCODED_BODY_LIMIT=1mb

# ── CORS ────────────────────────────────────────────────────
# This is the URL of your frontend. In development, Next.js runs on 3000.
CORS_ORIGIN=http://localhost:3000

# ── Database ─────────────────────────────────────────────────
# This connects to the PostgreSQL container started by Docker in Step 2.
# Note the port is 5433 (not the default 5432) to avoid conflicts with any local Postgres.
DATABASE_URL="postgresql://matcha_user:matcha_password@localhost:5433/matcha_db?schema=public"

# ── AI Integration ───────────────────────────────────────────
# Get your free Gemini API key from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY="your_google_ai_studio_api_key_here"

# (Optional) Get from https://huggingface.co/settings/tokens — increases TTS quality/rate limits
HF_TOKEN="hf_your_huggingface_token_here"

# ── Services ─────────────────────────────────────────────────
# The URL where the Python Inference engine is running
INFERENCE_URL=http://localhost:8000
```

### 5.2 — Web App environment file

The file `apps/web/.env.local` already exists in the repo with the correct defaults. Verify it contains:

```env
# This tells the frontend where to send API requests
NEXT_PUBLIC_API_URL=http://localhost:4000/api/v1
```

If the file doesn't exist, create it:

```bash
echo "NEXT_PUBLIC_API_URL=http://localhost:4000/api/v1" > apps/web/.env.local
```

### 5.3 — Inference Service environment file

The file `services/inference/.env` already exists. Verify it contains at minimum:

```env
ORCHESTRATOR_URL=http://localhost:4000/api/v1
GEMINI_API_KEY="your_google_ai_studio_api_key_here"
HF_TOKEN="hf_your_huggingface_token_here"
```

---

## Step 6: Run the Full Stack

Now that all dependencies and environment variables are set up, you can launch everything.

### Option A: The Recommended Way (Two Terminals)

The frontend and orchestrator can be started together with Turborepo:

**One command — starts Frontend, Orchestrator AND the Inference Engine simultaneously:**

```bash
# From the project root (Matcha-AI-DTU/)
npx turbo run dev
```

This starts all three services at once:

- **Web Frontend** at `http://localhost:3000` (or `3001` if 3000 is busy)
- **Orchestrator API** at `http://localhost:4000/api/v1`
- **Python Inference Engine** at `http://localhost:8000`

This works because `services/inference/package.json` now has a `dev` script that calls `./venv/bin/python -m uvicorn ...` directly via the venv path — **no manual shell activation needed**.

You'll see the Inference engine download YOLO model weights on the first run (~22 MB for `yolov8s-pose.pt` and ~6 MB for `yolov8n.pt`) — this is normal and only happens once.

> **Windows users**: The `dev` script uses `./venv/bin/python` (macOS/Linux path). On Windows, either run the inference manually in a separate terminal or use `npm run dev:win` from inside the `services/inference` folder.

### Option B: Manual Per-Service Startup

If you need to debug a specific service, open 3 separate terminals:

**Terminal 1 — Frontend only:**

```bash
cd apps/web
npm run dev
```

**Terminal 2 — Orchestrator only:**

```bash
cd services/orchestrator
npm run dev
```

**Terminal 3 — Inference Engine (macOS/Linux):**

```bash
cd services/inference
./venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 — Inference Engine (Windows):**

```powershell
cd services/inference
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Step 7: Verify Everything is Working

Once all services are running, confirm they're healthy:

### Check the Orchestrator

```bash
curl http://localhost:4000/api/v1/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "orchestrator",
  "uptime": 42,
  "timestamp": "2026-02-22T18:00:00.000Z"
}
```

### Check the Inference Engine

```bash
curl http://localhost:8000/
```

Expected response: some JSON with API info or a `{"message": "Matcha Inference API"}` type response.

### Check the Frontend

Open your browser and go to: **[http://localhost:3000](http://localhost:3000)**

You should see the Matcha AI web interface. If port 3000 was already in use, check the `npx turbo run dev` terminal output — it will say something like `⚠ Port 3000 is in use, trying 3001 instead`.

---

## ❓ Troubleshooting Common Errors

### Error: `Cannot find module '@matcha/env'` or `@matcha/shared`

**What it means:** The shared packages haven't been built yet, or the build failed.

**Fix:**

```bash
# From the project root
npm install
npx turbo run build
```

If the build still fails, check the specific package. For example, for `@matcha/env`:

```bash
cd packages/env
npm install
npx tsc
```

---

### Error: `Module not found: @t3-oss/env-core` (TypeScript build error)

**What it means:** The `tsconfig.json` in `packages/env` is using an incompatible `moduleResolution` setting.

**Fix:** Ensure `packages/env/tsconfig.json` has:

```json
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "target": "ES2021",
    "outDir": "./dist",
    "declaration": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  }
}
```

---

### Error: `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` (Python)

**What it means:** You're running Python 3.9, but the code uses the `X | Y` union type syntax which is only valid in Python 3.10+.

**Fix:** This has already been patched in `app/core/heatmap.py` and `app/core/analysis.py` to use `Optional[X]` from the `typing` module. If you see this error in a different file, replace any `SomeType | None` with `Optional[SomeType]` and add `from typing import Optional` at the top of the file.

---

### Error: Docker Desktop is not running

**Problem:** `docker-compose up -d --build` fails with `Cannot connect to the Docker daemon`, or Docker commands hang without listing containers.

**Cause:** Docker Desktop or the Docker daemon is not running yet, so Compose cannot create PostgreSQL, Redis, or MinIO containers.

**Fix:** Open Docker Desktop and wait until it reports that the engine is running. Then retry `docker-compose up -d --build` from the project root and confirm the services are healthy with `docker ps`.

---

### Error: Port `5433` or `6380` is already in use

**Problem:** Docker Compose reports that it cannot bind `0.0.0.0:5433` or `0.0.0.0:6380` when starting `matcha_postgres` or `matcha_redis`.

**Cause:** A local PostgreSQL, Redis, or earlier Matcha container is already using the host port mapped in `docker-compose.yml`.

**Fix:** Stop the process using the port, or change the left side of the port mapping in `docker-compose.yml` and mirror the same host port in your environment values. For example, on macOS/Linux run `lsof -ti:5433 | xargs kill -9` for the conflicting PostgreSQL process, then start the stack again with `docker-compose up -d`.

---

### Error: Container keeps restarting because of volume permissions

**Problem:** `docker ps` shows `Restarting`, or `docker logs matcha_postgres`, `docker logs matcha_redis`, or `docker logs matcha_minio` includes permission errors for `/var/lib/postgresql/data`, `/data`, or a mounted volume.

**Cause:** Docker is reusing a named volume created by another user, an older Docker Desktop state, or a previous failed setup with incompatible permissions.

**Fix:** If you do not need the local development data, reset the containers and volumes with `docker-compose down -v`, then run `docker-compose up -d --build` again. If you need to keep data, inspect the failing container with `docker logs <container-name>` and fix ownership or permissions before restarting it.

---

### Error: `Connection refused` on port 5433 (Database)

**Problem:** The Orchestrator or Prisma cannot connect to PostgreSQL at `localhost:5433`.

**Cause:** Docker is not running, the PostgreSQL container is still starting, or the local `.env` database URL does not match the Compose port mapping.

**Fix:**

1. Open Docker Desktop and make sure it's running.
2. Run `docker-compose up -d` from the project root.
3. Wait ~10 seconds, then run `docker ps` to confirm `matcha_postgres` status shows `Up`.
4. Check that your Orchestrator `DATABASE_URL` points to `localhost:5433` when running services directly on your host machine.

---

### Error: Cannot connect to PostgreSQL from the Orchestrator

**Problem:** The Orchestrator starts, but API requests or Prisma commands fail with database connection errors.

**Cause:** The database container is unavailable, the `DATABASE_URL` points at the wrong host or port, or the password/database name differs from the values in `docker-compose.yml`.

**Fix:** Confirm PostgreSQL is running with `docker ps` and `docker logs matcha_postgres`. For local development, the URL should use the Compose credentials and host port, for example `postgresql://matcha_user:matcha_password@localhost:5433/matcha_db`. Restart the Orchestrator after editing `.env` so it reloads the connection string.

---

### Error: `CORS` errors in the browser

**What it means:** The frontend is making requests from a URL that the Orchestrator doesn't recognize.

**Fix:** Check `CORS_ORIGIN` in `services/orchestrator/.env`. It must match the exact URL your frontend is running on (including the port). If Next.js is on port `3001`, set:

```env
CORS_ORIGIN=http://localhost:3001
```

---

### Error: `Prisma` says database schema not found

**What it means:** The database migrations haven't been applied to your PostgreSQL container.

**Fix:**

```bash
cd services/orchestrator
npx prisma migrate deploy
```

If that doesn't work, try:

```bash
npx prisma db push
```

---

### Error: `Port XXXX is already in use`

**What it means:** Another process is already using port 3000, 4000, or 8000.

**Fix:** Kill the process using that port:

```bash
# macOS/Linux — find and kill the process on port 4000:
lsof -ti:4000 | xargs kill -9

# or use kill-port (install with: npm i -g kill-port):
npx kill-port 4000
```

---

### YOLO model weights not downloading

**What it means:** The Inference engine needs internet access on first run to download ~22MB of YOLO weights.

**Fix:** Ensure you have an internet connection when you first start the Inference engine. The weights are saved locally after the first download at `services/inference/yolov8s-pose.pt` and `services/inference/yolov8n.pt`.

---

## 🏁 Quick Reference: Full Startup Checklist

Every time you want to run the project after a fresh terminal session:

- [ ] **Docker Desktop** is open and running
- [ ] Run `docker-compose up -d` from project root (or verify containers already running with `docker ps`)
- [ ] Run `npx turbo run dev` from project root — **starts Frontend, Orchestrator, and Inference Engine all at once**
- [ ] Open browser at `http://localhost:3000`

---

> [!TIP] **First time setup takes the longest.** Installing Python packages (especially PyTorch) and building Node.js packages can take 10-15 minutes total. Every subsequent startup is much faster — usually under 30 seconds.

> [!NOTE] **Gemini API Key**: Many features (AI commentary, match summaries, event detection via Vision AI) require a Gemini API key. You can get one for free at https://aistudio.google.com/app/apikey — the free tier is generous enough for development. Without it, the system falls back to a simpler motion-based highlight detection.

> [!IMPORTANT] **Windows users**: The default `npx turbo run dev` command starts the Inference engine using `./venv/bin/python` (a macOS/Linux path). On Windows, run the Inference engine separately using `cd services/inference && .\venv\Scripts\activate && python -m uvicorn main:app --port 8000 --reload`, or use `npm run dev:win` from inside `services/inference`.
