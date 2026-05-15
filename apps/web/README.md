# Matcha-AI-DTU Frontend (`apps/web`)

This is the frontend application for the Matcha-AI-DTU platform, built with modern web technologies including **Next.js 15**, **React 19**, and **Tailwind CSS**.

## Features

- **Video Upload Portal**: Utilizes `react-dropzone` for robust and intuitive drag-and-drop video file uploading.
- **Real-Time Dashboards**: Establishes continuous connection to the Orchestrator utilizing `socket.io-client` for real-time video processing updates and analysis percentages.
- **Client-Side AI Integration**: Features TensorFlow.js (`@tensorflow/tfjs`, `@tensorflow-models/coco-ssd`) and FFmpeg WASM interfaces for client-side preprocessing.
- **Sleek UI Framework**: Tailored with extensive Tailwind CSS styling, animated icons from `lucide-react`, and centralized logical theming mapped to exact CSS hex variables.
- **On-the-fly PDF Match Reports**: Uses `@react-pdf/renderer` to seamlessly generate visually rich, multi-page Match Report PDFs locally in the browser.
- **Unified Component Library**: Fully consumes **`@matcha/ui`** for high-fidelity elements including `VideoPlayer`, `MatchReportPDF`, and `ScoreBadge`.
- **Fault-Tolerant API**: Leverages the resilient `ApiClient` from **`@matcha/shared`**, featuring automatic exponential backoff and retries via `fetchWithRetry`.
- **Standardized Real-time logic**: Utilizes the shared **`useMatchSocket`** hook for consistent, de-duplicated event monitoring.
- **Zod Data Contracts**: All API interactions and forms are validated against unified schemas from **`@matcha/contracts`**.
- **Design System**: Styles are bound to global tokens in **`@matcha/theme`**, ensuring perfect brand consistency.
- **Boot-time Env Safety**: Uses **`@matcha/env`** and T3-Env to ensure all required API keys and public URLs are validated before the React application boots.

## Tech Stack

- **Framework**: Next.js 15 (App Router)
- **UI System**: `@matcha/ui`, `@matcha/theme`, `framer-motion`, `lucide-react`
- **Logic & Types**: `@matcha/shared`, `@matcha/contracts`, `socket.io-client`
- **Validation**: `@matcha/env`
- **Media**: `@react-pdf/renderer` (PDF generation), `FFmpeg` (Video)

## Getting Started

From the root of the repository or from `apps/web`:

### Installation

Dependencies are gracefully handled by turbo in the root workspace, but you can install locally too:

```bash
npm install
```

### Running Locally

```bash
npm run dev
```

The application will launch on `http://localhost:3000`. Ensure the Orchestrator API (`http://localhost:4000`) is running concurrently for full functionality, as the frontend heavily relies on it for API requests and WebSocket events.

## Directory Context

- `app/`: Next.js App Router endpoints, pages, and layouts (e.g. `page.tsx`, `globals.css`).
- Components include modern Hooks-based React designs managing websocket states, highlighting parsed game events, and rendering generated commentary audio on top of analyzed sports videos.
