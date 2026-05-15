---
description: how to run the full matcha-ai stack
---

# Running Matcha AI

## Single Command (Recommended)

Run everything from the repo root:

// turbo

```bash
docker-compose up -d && mkdir -p uploads && npx turbo run dev
```

This starts:

- **Docker** → PostgreSQL, Redis, MinIO
- **Inference service** → Python/uvicorn on :8000 (via `services/inference` `dev` script)
- **Orchestrator** → NestJS on :4000
- **Web frontend** → Next.js on :3000

> **First time only** — you must create the Python venv first:
>
> ```bash
> cd services/inference && python3 -m venv venv && venv/bin/pip install -r requirements.txt && cd ../..
> ```

## Service URLs

| Service          | URL                   |
| ---------------- | --------------------- |
| Web Frontend     | http://localhost:3000 |
| Orchestrator API | http://localhost:4000 |
| Inference API    | http://localhost:8000 |
| MinIO Console    | http://localhost:9001 |

## Troubleshooting

- **Prisma types missing**: `cd services/orchestrator && npx prisma generate`
- **Inference fails to start**: Make sure `services/inference/venv/` exists (see First Time setup above)
- **Stuck at 0%**: Inference service likely not running — check terminal output for port :8000 errors
