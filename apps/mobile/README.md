# Matcha AI Mobile

Expo (React Native) app for Matcha AI DTU.

## Stack

- **Expo SDK 52** + Expo Router (file-based navigation)
- **TypeScript** throughout
- **socket.io-client** for real-time match events
- **expo-document-picker** + **expo-file-system** for video upload
- **expo-av** for video playback
- **Universal Types & Logic**: Integrates seamlessly with the `@matcha/shared` workspace package, inheriting `MatchDetail`, `formatTime()`, and unified `@matcha/shared` WebSocket constants.

## Folder layout

```
apps/mobile/
├── app/ # Expo Router screens (file = route)
│ ├── _layout.tsx # Root stack navigator
│ ├── index.tsx # Dashboard (match list)
│ ├── upload.tsx # Video upload
│ └── matches/
│ └── [id].tsx # Match detail
├── components/ # Shared UI components
│ ├── MatchCard.tsx
│ ├── StatusChip.tsx
│ └── IntensityChart.tsx
├── hooks/ # Custom React hooks
│ ├── useMatches.ts # Match list + WebSocket
│ ├── useMatchDetail.ts # Single match fetch
│ └── useVideoUpload.ts # File pick + upload with progress
├── services/
│ └── orchestrator.ts # HTTP API client
├── constants/
│ └── api.ts # Base URLs + status colors
└── assets/ # Icons, splash screen
```

## Dev

```bash
cd apps/mobile
npm install
npx expo start --android
```

> **Note:** `API_BASE` in `constants/api.ts` uses `10.0.2.2:4000` which maps to `localhost` from the Android emulator. For a physical device, replace with your machine's local IP (e.g. `192.168.x.x:4000`).
