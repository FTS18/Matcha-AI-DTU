# `@matcha/theme`

The design system and styling configuration for the Matcha AI platform.

## Brand Tokens

We use a centralized `BRAND_COLORS` object to ensure consistency across Web, Canvas, and PDF layers.

| Token        | Hex       | Role                                       |
| :----------- | :-------- | :----------------------------------------- |
| `primary`    | `#bef264` | Matcha Lime (High-visibility action color) |
| `background` | `#07080F` | Deep dark UI backdrop                      |
| `card`       | `#111218` | Elevated element background                |
| `border`     | `#1f2028` | Subtle separator color                     |
| `success`    | `#34d399` | Positive event state                       |

## Tailwind Integration

This package exports a base `tailwindConfig` that standardizes:

- **Fonts**: Barlow (Display), Inter (San-serif), DM Mono (Monospace).
- **Colors**: Mapped to CSS variables for dynamic theme support.
- **Radii**: Standardized rounded corners (`xl`, `lg`, `md`).

## Usage

In your app's `tailwind.config.ts`:

```ts
import type { Config } from "tailwindcss";
import { tailwindConfig } from "@matcha/theme";

const config: Config = {
  ...tailwindConfig,
  content: ["./app/**/*.{ts,tsx}"],
};

export default config;
```
