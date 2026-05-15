# `@matcha/shared`

The universal logic layer for the Matcha AI platform. This package contains shared types, the API client, and logical utilities used by Web, Mobile, and Orchestrator.

## Features

### 1. Unified Types

All TypeScript interfaces and enums are exported from here, ensuring zero type-drift between the frontend and backend. Most types are inferred from `@matcha/contracts`.

### 2. The Resilient `ApiClient`

A robust abstraction over `fetch` featuring:

- **Exponential Backoff**: Automatic retries for failed network requests via `fetchWithRetry`.
- **Automatic Auth**: Seamlessly injects Bearer tokens when available.
- **Typed Responses**: Guaranteed return types for every endpoint.

### 3. Registry & Utils

- **`EVENT_REGISTRY`**: Metadata about SoccerNet event types (icons, labels, colors).
- **Formatters**: Consistent match-time formatting and relative timestamp logic.
- **`useMatchSocket`**: Although the hook lives in `@matcha/ui`, it consumes the registries and types defined here.

## Usage

```ts
import { apiClient } from "@matcha/shared";
import { MatchStatus } from "@matcha/shared/types";

const matches = await apiClient.matches.list();
```
