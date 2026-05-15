# Frequently Asked Questions

This document answers the most common questions asked by contributors and users of Matcha-AI-DTU. Read this before opening an issue — your question is probably already answered here.

---

## Setup and Installation

### Do I need a GPU to run this project?

No. A GPU is completely optional. The entire stack runs on a standard CPU laptop or desktop. The difference is speed: on a CPU, analyzing a 90-minute match video may take 20-40 minutes. With an NVIDIA GPU (CUDA 12.4), the same analysis typically completes in 5-8 minutes. For development and testing purposes, CPU is perfectly fine — use short test videos (30 seconds to 5 minutes) to keep iteration fast.

---

### Can I run this without a Gemini API key?

Partially. Without a Gemini API key set in `services/inference/.env`, the following features will not work:

- AI-generated event commentary (the text description for each detected event)
- The overall match summary narrative

Everything else still works: YOLO tracking, event detection, the highlight reel video (with silent or edge-tts audio), the heatmap, ball speed estimation, and team color detection. You can get a free Gemini API key with a generous daily quota from https://aistudio.google.com/app/apikey — it takes about 2 minutes.

---

### Can I run this without a HuggingFace token?

Yes. The HuggingFace token (`HF_TOKEN`) is only used for the Kokoro-82M TTS model (Tier 1). Without it, the system automatically falls back to Microsoft edge-tts (Tier 2), which requires no API key and always works. The highlight reel audio quality will be slightly lower (Microsoft voice instead of Kokoro), but all functionality is intact.

---

### Why does the inference engine fail to start on Windows when I run `npx turbo run dev`?

The `dev` script in `services/inference/package.json` calls `./venv/bin/python`, which is the Unix path to the Python executable inside the virtual environment. On Windows, this path does not exist. The Windows equivalent is `.\venv\Scripts\python.exe`.

The solution is to run the inference engine manually in a separate terminal:

```powershell
cd services/inference
.\venv\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Leave this terminal open and run `npx turbo run dev` from the project root in a different terminal. The Turbo dev command will start the frontend and orchestrator, and the manually started Python process handles the inference service.

---

### The `docker-compose up -d` command fails with "Cannot connect to the Docker daemon." What do I do?

Docker Desktop is not running. On Windows, open the Docker Desktop application from your Start menu and wait for it to fully start (the Docker icon in the system tray should stop animating). Then run `docker-compose up -d` again. Docker Desktop must be open and running every time you work on this project.

---

### `npm install` shows many "Unsupported engine" warnings. Is something broken?

No. These warnings come from some deep transitive dependencies (packages that your packages depend on) that were written for older npm versions. They are warnings, not errors. `npm install` still completes successfully and the project runs correctly. You can safely ignore these messages.

---

### I ran `npx prisma migrate dev` and it asked to reset my database. Should I say yes?

Yes, if this is your local development database. The reset deletes all data in the database and re-applies migrations from scratch to get everything in sync. Since your local database only has test data (no real user data), resetting is the correct action. Never run `migrate dev` against a production database.

---

### The build fails with "Cannot find module '@matcha/env'" or similar. What went wrong?

The shared TypeScript packages have not been built yet, or a previous build failed. Run from the project root:

```bash
npm install
npx turbo run build
```

The `build` command compiles all shared packages (`@matcha/env`, `@matcha/shared`, `@matcha/ui`, etc.) into JavaScript so the apps can import them. If the build still fails, check the specific package output for the error — it usually points to a missing environment variable or a TypeScript type error.

---

### Installing Python requirements takes forever and then fails. What should I do?

This usually happens for one of these reasons:

1. **PyTorch download is slow**: PyTorch is a large package (~800MB for CPU, ~2GB for CUDA). This is normal on the first install. Be patient — it can take 10-20 minutes on a slow connection.

2. **Disk space**: Make sure you have at least 5GB of free disk space before installing.

3. **Python version**: Make sure you are using Python 3.9 or higher. Run `python --version` to check. The project is not compatible with Python 3.7 or 3.8.

4. **Not inside the virtual environment**: Make sure you see `(venv)` at the start of your terminal prompt before running `pip install`. If not, run `.\venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux) first.

---

## Using the Application

### What is the minimum video length for analysis?

There is no enforced minimum, but videos shorter than about 30 seconds may produce zero detected events, because the motion-peak detection algorithm needs enough frames to identify meaningful action. For best results, use at least 5 minutes of footage. Full match videos (90 minutes) work correctly.

### What video formats are supported?

The inference engine uses OpenCV to read videos, which supports most common formats: `.mp4`, `.mov`, `.avi`, `.mkv`. The frontend validates for `.mp4`, `.mov`, and `.avi` before uploading. For best compatibility, use H.264-encoded MP4 files.

---

### Why does the analysis get stuck at 0%?

The most common cause is that the frontend's WebSocket is connected but the Inference engine is not calling back to the Orchestrator. Check these in order:

1. Is the Python inference engine running? Look for `Uvicorn running on http://0.0.0.0:8000` in your terminal.
2. Is `ORCHESTRATOR_URL` set correctly in `services/inference/.env`? It must be `http://localhost:4000`.
3. Is the Orchestrator running? Check for `Nest application successfully started` in the terminal.
4. Check the Python terminal for red error messages. A missing `GEMINI_API_KEY` or a video format issue will cause the pipeline to crash silently before sending progress callbacks.

---

### Why is the heatmap not showing in the Analytics tab?

The heatmap is only generated if YOLO detected at least a few players during Phase 1. If the video has very low resolution, poor lighting, or is not a football match, YOLO may fail to detect players. Check the Python inference terminal for a line that says either `Heatmap saved to uploads/heatmap_<id>.png` or `Heatmap generation failed: <error>`. If generation failed, the error message will indicate the cause.

---

### The highlight reel audio is silent. Why?

Either:

1. Both Kokoro-82M and edge-tts failed, causing the system to fall back to the silent audio fallback. Check the Python terminal for TTS error messages.
2. Your `HF_TOKEN` is set but invalid. Try removing it from `.env` to force Tier 2 (edge-tts), which does not need a token.
3. `edge-tts` is not installed. Run `pip install edge-tts` inside your activated venv.

---

## Contributing

### I want to contribute but I have never done open source before. Where do I start?

Read `FIRST_CONTRIBUTION.md` before anything else. It explains forking, branching, and pull requests in plain English without assuming prior experience. Then look at `ROADMAP.md` and find a task labeled `[Beginner]` that interests you.

---

### How do I know which issues are suitable for beginners?

Look for issues with the `good first issue` label on the GitHub Issues tab. All beginner-level roadmap items have corresponding issues labeled this way. You can also filter by `documentation` label — documentation improvements are an excellent first contribution with a very low technical barrier.

---

### Can I work on a feature that is not in the ROADMAP?

Yes, but you must open a Feature Request issue first and get approval from a maintainer before starting work. This prevents you from spending time on something that turns out to be out of scope or already in progress. If a maintainer approves your feature request, they will add it to the roadmap and you can proceed.

---

### How long does it take to get a PR reviewed?

Maintainers aim to review all open PRs within 48-72 hours. If your PR has not received a review after 3 days, leave a comment on the PR to ping the reviewers. Do not open a duplicate PR.

---

### I made a mistake in my PR (wrong branch, forgot to add something, etc.). What do I do?

Just push more commits to the same branch. GitHub automatically updates the PR with the new commits. You do not need to close and re-open the PR. For small fixes, a simple `git add . && git commit -m "fix: address review comments" && git push` is all you need.

---

### My PR is showing conflicts with the base branch. How do I fix it?

```bash
git checkout dev
git pull upstream dev
git checkout your-feature-branch
git rebase dev
# Resolve any conflicts in the files Git marks
git add .
git rebase --continue
git push --force-with-lease origin your-feature-branch
```

The `--force-with-lease` flag is safer than `--force` because it will fail if someone else has pushed to your branch in the meantime.

---

### Do I need to run tests before opening a PR?

The project does not yet have an automated test suite (writing tests is itself a roadmap item). However, you must manually test your changes before opening a PR:

- Run the full stack locally.
- Upload a test video and verify the analysis completes without errors.
- Confirm your changes work end-to-end.
- Check that you have not broken any other feature (other tabs, other pages).
- Fill out the "How Has This Been Tested?" section of the PR template honestly.
