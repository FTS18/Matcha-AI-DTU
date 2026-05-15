# Troubleshooting Matcha-AI-DTU

If you run into issues launching the application or processing a video, refer to the most common resolutions below.

---

## Docker Infrastructure Issues

### Problem: `docker compose up -d` fails or ports are bound

**Symptoms**: Docker cannot bind to `5433` (Postgres) or `6380` (Redis). **Cause**: Another application or local database service is occupying the port. **Resolution**:

1. Run `netstat -ano | findstr :5433` (Windows) to identify the PID.
2. Terminate the blocking process via Task Manager or `taskkill /PID <id> /F`.
3. Alternatively, explicitly define different port mappings in the `docker-compose.yml` and mirror them in `services/orchestrator/.env`.

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
