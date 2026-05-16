# Engineering Setup and Development Guide

## Prerequisites
*   Node.js 18+ (npm 10+)
*   Python 3.10+
*   Turbo CLI (`npm install -g turbo`)
*   FFmpeg (for video processing)

## Monorepo Structure
Matcha AI uses a workspace-based monorepo managed by Turbo.

```text
.
â”œâ”€â”€ apps/
â”‚   â”œâ”€â”€ web/          # Next.js Dashboard
â”‚   â””â”€â”€ mobile/       # Expo/React Native App
â”œâ”€â”€ services/
â”‚   â”œâ”€â”€ inference/    # Python AI Engine
â”‚   â””â”€â”€ orchestrator/ # NestJS API Gateway
â””â”€â”€ packages/         # Shared configs and UI components
```

## Initial Setup

### 1. JavaScript Dependencies
From the root directory:
```bash
npm install --legacy-peer-deps
```
*Note: The `--legacy-peer-deps` flag is required to maintain compatibility between ESLint 8.x and Next.js 14.x configurations.*

### 2. Python Environment
Navigate to `services/inference`:
```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Environment Variables
Copy `.env.example` to `.env` in the following locations and populate the required keys:
*   `./.env`
*   `./services/inference/.env`
*   `./services/orchestrator/.env`

## Execution

### Development Mode
To start all services in parallel:
```bash
npm run dev
```

### Automated Testing
To verify the AI engine tracking logic:
```bash
cd services/inference
pytest test_goal_detection.py
```

### Linting
To ensure code quality across all workspaces:
```bash
npm run lint
```
