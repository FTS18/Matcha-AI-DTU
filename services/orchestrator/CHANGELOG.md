# orchestrator

## 1.0.1

### Patch Changes

- dcb2f44: fix: hardened infrastructure security and code quality
  - Sanitized logging in `goal_detection.py`, `analysis.py`, `downloader.py`, `vision_engine.py`, and `transformer.py` to prevent sensitive data leakage.
  - Cleaned up unused imports and dead code in `analysis.py`, `vision_engine.py`, and `transformer.py` to reduce CodeQL alerts.
  - Replaced empty `except: pass` blocks with safe debug logging for better observability.
  - Standardized error handling across the inference service.
  - Stabilized `orchestrator/Dockerfile` by removing the problematic `npm@latest` upgrade and ensuring `npm ci` works with the pruned lockfile.
