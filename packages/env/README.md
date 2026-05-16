# `@matcha/env`

Strict, boot-time environment variable validation powered by **T3-Env** and **Zod**.

## Why?

This package prevents the "silent failure" of services when environment variables are missing or misconfigured. If a required key is missing, the service will **crash early** with a detailed diagnostic message.

## Environment Scopes

The package validates separate scopes:
- **Server**: Database URLs, Secret Keys, Internal Service URLs.
- **Client**: Public API URLs (prefixed with `NEXT_PUBLIC_`).

## Usage

In your entry point (`main.ts` or `app/layout.tsx`):

```ts
import { env } from "@matcha/env";

console.log(env.DATABASE_URL); // Guaranteed to be a valid string
```
