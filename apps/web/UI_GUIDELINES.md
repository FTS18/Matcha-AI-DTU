# UI/UX Guidelines & Architecture

Welcome to the frontend documentation for Matcha-AI-DTU (`apps/web`). This document outlines our implementation patterns for component creation, styling hooks, and state management.

## Theme & Styling Strategy (Tailwind CSS v4)

We aggressively use **Tailwind v4** utilizing an aesthetically minimal, "premium flat" UI paradigm as updated from our brutalist roots.

### Core Utilities

- **`clsx` & `tailwind-merge` (`twMerge`)**: ALWAYS use these when creating reusable components that accept `className` props to resolve tailwind conflicts gracefully.

```tsx
// Example generic component pattern
import { cn } from "@/lib/utils"; // Assume this wraps clsx(twMerge(...))

export const Card = ({ className, children }) => {
  return (
    <div className={cn("bg-neutral-900 rounded-xl shadow-lg border border-neutral-800", className)}>{children}</div>
  );
};
```

### Color Palette Constraints

- Adhere strictly to the defined CSS variables in `app/globals.css`.
- Rely entirely on dark/neutral hues (neutral-800 to neutral-950) for backgrounds.
- Highlighting colors should use our predefined primary accent variables.

## State & Real-Time Management (WebSockets)

Matcha-AI is inherently real-time. Video processing takes time, and the user expects live updates.

### Subscribing to Orchestrator Events

We use `socket.io-client` localized to React hooks.

**Standard Pattern for implementing Socket Listeners:**

1. Connect via `useEffect` mounted strictly once.
2. Bind to the event name emitted by the Orchestrator (`analysisProgress`, `analysisComplete`, `eventDetected`).
3. Ensure you gracefully `socket.off()` or `socket.disconnect()` on component unmount to prevent severe memory leaks in the Next.js App Router client rendering lifecycle.

```tsx
useEffect(() => {
  // connect to ws://localhost:4000
  const socket = io(process.env.NEXT_PUBLIC_ORCHESTRATOR_URL);

  socket.on("analysisProgress", (data) => {
    setProgress(data.percentage);
  });

  return () => {
    socket.disconnect(); // CRITICAL!
  };
}, []);
```

## ️ Handling Video Uploads & Preview

Video drops are managed via `react-dropzone`. When previewing video tags `<video>`, always ensure `preload="metadata"` and provide explicit error fallbacks. For very large files, avoid loading the entire blob into React state; prefer generating temporary `URL.createObjectURL(file)` bindings instead.

## Icons

Use `lucide-react` for all iconography. Import specific icons natively to maintain slim bundle sizes rather than blanket imports.
