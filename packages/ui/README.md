# `@matcha/ui`

A high-fidelity React component library built with **Tailwind CSS** and **Framer Motion**. This package is the source of truth for all visual elements across the Matcha AI platform.

## Key Components

### 1. `VideoPlayer`

The flagship component for match analysis.

- **AI Overlays**: Renders real-time bounding boxes and tracklets via Canvas.
- **Event Highlights**: Synchronized event toasts and seeker-bar markers.
- **Motion Polish**: Smooth transitions and interaction states powered by `framer-motion`.

### 2. `MatchReportPDF`

A robust PDF generation engine utilizing `@react-pdf/renderer`.

- **Dynamic Data**: Generates multi-page reports with event breakdowns, heatmaps, and analytics.
- **Themed Styling**: Bound to `BRAND_COLORS` from `@matcha/theme`.

### 3. `ScoreBadge` & `CopyButton`

Atomic primitives with standardized styling and feedback.

## Shared Hooks

### `useMatchSocket`

A specialized hook for real-time match monitoring.

- **De-duplication**: Handles WebSocket connection pooling.
- **Typed Events**: Provides reactive state for match status, analysis progress, and event streaming.

## Usage

Ensure your tailwind config includes the `packages/ui` source:

```ts
import { tailwindConfig } from "@matcha/theme";
export default {
  ...tailwindConfig,
  content: [...tailwindConfig.content, "../../packages/ui/src/**/*.{ts,tsx}"],
};
```
