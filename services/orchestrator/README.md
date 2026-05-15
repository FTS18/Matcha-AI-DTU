# Matcha-AI-DTU Orchestrator (`services/orchestrator`)

The Orchestrator acts as the central hub of the Matcha-AI-DTU monorepo. Built aggressively leveraging the **NestJS** framework for its modularity and robust API routing, this service primarily manages communication between the web frontend and the Python AI Inference engine.

## The Central Link

- **Coordinates the Flow**: The primary gateway `http://localhost:4000`. Receives uploaded sports footage, delegates video analysis to the Inference Engine, and then pushes structured progress updates directly to the frontend.
- **WebSocket Gateway**: Integrates extensively with `socket.io`. When Inference reports back processed statuses or identified events, the Orchestrator instantly emits these updates to Next.js clients avoiding active polling.
- **Data Persistence**: Offloads all database logic, migrations, and schema management to the centralized **`@matcha/database`** package.
- **Protocol Safety**: Uses **`@matcha/contracts`** for end-to-end request/response validation, ensuring the API cannot drift from the frontend.
- **Auth & Security**: JWT-based authentication linking analyzed matches to users, powered by NestJS AuthModule and strict environment validation via **`@matcha/env`**.
- **Standardized Utilities**: Re-uses registries and formatting logic from **`@matcha/shared`**.

## Tech Stack

- **Framework**: NestJS (`@nestjs/core`)
- **Real-time**: Socket.io (`@nestjs/websockets`)
- **Infrastructure**: Redis (Cache), PostgreSQL (via `@matcha/database`)
- **Validation**: `@matcha/contracts`, `@matcha/env`

## Setup & Execution

### 1. Unified Setup

It is highly recommended to manage the Orchestrator via the root monorepo commands.
Please refer to the **[Root SETUP.md](../../SETUP.md)** for detailed infrastructure and environment variable instructions.

### 2. Manual Commands

If running in isolation:

```bash
# Generate database types
npx turbo run generate

# Run migrations
npx turbo run db:migrate (filter accordingly)

# Start in dev mode
npm run start:dev
```

### Service Launch

Install dependencies if not already run at the workspace root:

```bash
npm install
```

Run in development/watch mode:

```bash
npm run start:dev
```

```bash
# Debug Mode
npm run start:debug
```

The server binds natively to port **`4000`**. You can hit it from `apps/web` components directly.

## Service Design

The `services/orchestrator` operates via distinct Controllers, Services, and Gateways.
It essentially wraps around `app.module.ts` separating logic based on:

1. Video Intake Routes (`HTTP POST`)
2. Real-time Status (`WebSocket events`)
3. Inference Return Hooks (`HTTP Callbacks`)
4. Media Hosting (`Uploads` Static Directory Delivery)
