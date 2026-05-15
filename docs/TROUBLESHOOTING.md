# Troubleshooting Matcha-AI-DTU

If you run into issues launching the application or processing a video, refer to the most common resolutions below.

---

## Docker Infrastructure Issues

### Problem: PostgreSQL port `5433` is already in use

**Problem**: `docker compose up -d` fails with an error similar to `Bind for 0.0.0.0:5433 failed: port is already allocated`.

**Cause**: A local PostgreSQL instance or another container is already listening on port `5433`, so Docker cannot publish the Matcha-AI-DTU database container on that host port.

**Fix**:

1. Find the process using the port:
   - Windows: `netstat -ano | findstr :5433`
   - macOS/Linux: `lsof -i :5433`
2. Stop the conflicting process, or change the PostgreSQL host port in `docker-compose.yml`.
3. If you change the port mapping, mirror the same value in `services/orchestrator/.env` so the orchestrator connects to the right database endpoint.

### Problem: Redis port `6380` is already in use

**Problem**: Docker fails to start Redis and reports that port `6380` is already allocated.

**Cause**: Another Redis instance, background service, or previous Matcha-AI-DTU container is still bound to the host port.

**Fix**:

1. Check what is using the port:
   - Windows: `netstat -ano | findstr :6380`
   - macOS/Linux: `lsof -i :6380`
2. Stop the conflicting process, or run `docker compose down` to remove stale project containers.
3. If the port must stay occupied, update the Redis port mapping in `docker-compose.yml` and keep any matching environment variables in sync.

### Problem: A container keeps restarting because mounted volumes are not writable

**Problem**: `docker compose ps` shows a service repeatedly restarting, and `docker compose logs <service>` includes permission errors for mounted directories or generated files.

**Cause**: Docker cannot write to a bind-mounted project directory, commonly after switching between Windows, WSL, Docker Desktop, or a different user account.

**Fix**:

1. Stop the stack with `docker compose down`.
2. Ensure the project directory is writable by your current user. On macOS/Linux, run `chmod -R u+rw .` from the repository root if needed.
3. Remove stale containers and recreate them with `docker compose up -d --force-recreate`.
4. Re-check the failing service with `docker compose logs <service>`.

### Problem: The orchestrator cannot connect to PostgreSQL

**Problem**: The orchestrator starts but logs database connection errors such as `ECONNREFUSED`, authentication failures, or timeouts when it tries to reach PostgreSQL.

**Cause**: The database container is not healthy yet, the host/port in `services/orchestrator/.env` does not match `docker-compose.yml`, or the orchestrator is using a host-only address from inside Docker.

**Fix**:

1. Confirm PostgreSQL is running with `docker compose ps postgres` and inspect logs with `docker compose logs postgres`.
2. Compare `DATABASE_URL` in `services/orchestrator/.env` with the PostgreSQL service name, username, password, and published port in `docker-compose.yml`.
3. When the orchestrator runs on the host, use the published host port such as `localhost:5433`. When it runs inside Docker, use the Compose service name and internal port, such as `postgres:5432`.
4. Restart the orchestrator after changing environment variables.

### Problem: Docker Desktop is not running

**Problem**: Docker commands fail with messages like `Cannot connect to the Docker daemon`, `docker daemon is not running`, or `error during connect`.

**Cause**: Docker Desktop or the Docker daemon has not started, or your terminal is connected to a context where Docker is unavailable.

**Fix**:

1. Start Docker Desktop and wait until it reports that Docker is running.
2. Run `docker info` to confirm the daemon is reachable from the same terminal.
3. If you use WSL, ensure Docker Desktop WSL integration is enabled for the distribution where you cloned the repository.
4. Re-run `docker compose up -d` after the daemon is available.

---

## Python / Inference Service Issues

### Problem: Torch/Cuda uses excessive VRAM

**Symptoms**: A video fails midway through analysis with `RuntimeError: CUDA out of memory`. **Cause**: High-resolution video frames (e.g., 4K) are overflowing GPU VRAM limits. **Resolution**:

1. Open `services/inference/app/core/analysis.py`.
2. Locate the frame reading logic and ensure `cv2.resize()` scales frames down to a maximum of 720p or 1080p before passing them into the YOLO pipeline.

### Problem: `piper-tts` fails to install on Windows

**Symptoms**: `pip install piper-tts` fails with obscure C++ compiler errors. **Resolution**: `piper-tts` has been removed from `requirements.txt`. The TTS system now uses **Kokoro-82M** (via `huggingface-hub`) and **edge-tts** as fallbacks. Both install without C++ compilation. Just run `pip install -r requirements.txt`.

### Problem: Progress sits at 0% and does not update

**Symptoms**: Video uploads successfully via Next.js, but the UI is stuck at "Beginning Analysis (0%)". **Cause**: The Inference service isn't reachable, or the Orchestrator WebSocket URL is incorrect inside Python. **Resolution**:

1. Verify `services/inference` is actively running on Port `8000` via Uvicorn.
2. Check `services/inference/.env` — ensure `ORCHESTRATOR_URL=http://localhost:4000` is set correctly. It **must** include the `http://` prefix.
3. Check the inference terminal for callback error messages like `Failed to send completion`.

### Problem: Kokoro-82M TTS fails / analysis uses fallback voice

**Symptoms**: Inference logs show `[TTS Tier-2] edge-tts generated` instead of `[TTS Tier-1] Kokoro-82M`. **Cause**: `HF_TOKEN` is not set, or the HuggingFace API is rate-limiting the anonymous user. **Resolution**:

1. Create a free HuggingFace account at [huggingface.co](https://huggingface.co).
2. Generate a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Add to `services/inference/.env`: `HF_TOKEN=hf_your_token_here`
4. Also add to `services/orchestrator/.env`: `HF_TOKEN=hf_your_token_here`
5. Restart the inference service.

> `edge-tts` (Tier 2) still produces high-quality British neural voice output. The system continues to work without `HF_TOKEN` — just at lower quality.

### Problem: Heatmap not appearing on the Analytics tab

**Symptoms**: The Analytics tab shows "Heatmap not generated yet" even after analysis completes. **Cause A**: YOLO did not detect any players in the video (e.g., very dark footage, wrong sport). **Cause B**: The `generate_heatmap()` call failed silently. **Resolution**:

1. Check the inference service terminal for log lines containing `Heatmap saved →` (success) or `Heatmap generation failed:` (error message).
2. If failed, re-analyze the match — click "Re-analyze" in the match detail page header.
3. Ensure the `/uploads/` directory is writable: `New-Item -ItemType Directory -Force -Path uploads`.

### Problem: Analytics tab shows 0 km/h ball speed

**Symptoms**: Ball speed is 0.0 KM/H on the Analytics tab. **Cause**: YOLO detected no ball (`sports ball` class) in the video, so no consecutive ball positions were available to compute speed. **Resolution**:

1. Ensure your video contains close-up footage where the ball is visible and reasonably sized.
2. Lower `MIN_CONF["sports ball"]` in `analysis.py` from `0.30` to `0.20` to increase sensitivity.
3. Re-analyze the match.

---

## Node.js & Prisma Issues

### Problem: Prisma Client reports missing tables

**Resolution**: Run the unified migration command from the root.

```bash
npx turbo run db:migrate
```

### Problem: Prisma `generate` fails with EPERM (file locked)

**Resolution**:

1. Stop all services (`Ctrl+C` in the turbo terminal).
2. Run `npx turbo run generate`.
3. Restart.

### Problem: `heatmapUrl` / `topSpeedKmh` fields don't exist (TypeScript type error)

**Symptoms**: IDE shows red underlines on `heatmapUrl` in `matches.service.ts`. **Cause**: The Prisma client types are stale — the migration ran but `generate` hasn't completed while the process was running. **Resolution**: Follow the EPERM fix above to stop the service, regenerate, and restart. The types will resolve automatically.

### Problem: `npm install` throws ERESOLVE conflicts

**Symptoms**: Older packages conflict with Next.js 15 or React 19. **Cause**: Strict peer dependency checking in npm v10+. **Resolution**: We heavily rely on latest features. Ensure you run:

```bash
npm install --legacy-peer-deps
```

If errors persist inside `apps/web`.

---

## Frontend Issues

### Problem: Analytics tab shows empty state for all metrics

**Symptoms**: Ball speed "unavailable", no heatmap, no team colors — even after a completed analysis. **Cause**: Match was analyzed before Phase 5 analytics were added (no `heatmapUrl` / `topSpeedKmh` in DB). **Resolution**: Click **Re-analyze** on the match detail page. This re-runs the full 5-phase pipeline and generates all analytics data.

### Problem: Mobile layout overflow / filter tabs not scrolling

**Symptoms**: Status filter tabs in the Match Dashboard overflow on small screens. **Resolution**: The `hide-scrollbar` utility class in `globals.css` should be applied to the tab container. Verify `match-dashboard.tsx` has `className="... hide-scrollbar"` on the `<div>` wrapping the filter tabs.

---

Still stuck? Please check our `.github/ISSUE_TEMPLATE` and file a detailed bug report!
