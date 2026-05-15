# `@matcha/contracts`

The single source of truth for API schemas and validation logic.

## Features

This package exports raw **Zod** schemas used by:

1. **Orchestrator**: For request body validation and response serialization.
2. **Web App**: For form validation and typed API consumption.
3. **Shared Types**: The `@matcha/shared` types are inferred directly from these schemas.

## Core Contracts

### `MatchContract`

Validates full match objects, including video URLs, status enums, and nested analytics (events, heatmaps).

### `MatchEventSchema`

Structural validation for the 11 SoccerNet event types (GOAL, SAVE, etc.), ensuring positional and temporal coordinates are present.

### `AnalyzeMatchSchema`

Request schema for initiating analysis, ensuring valid team names and jersey color selections.

## Usage

```ts
import { AnalyzeMatchSchema } from "@matcha/contracts";
import { z } from "zod";

type AnalyzeRequest = z.infer<typeof AnalyzeMatchSchema>;
```
