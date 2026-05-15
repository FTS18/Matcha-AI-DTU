# `@matcha/database`

The centralized data layer for the Matcha AI platform, containing the Prisma schema and generated client.

## Commands

Manage the database from the monorepo root:

```bash
# Generate the Prisma Client
npx turbo run generate

# Create and apply a new migration
npx turbo run db:migrate -- --name your_migration_name

# Explore data via Prisma Studio
cd packages/database && npx prisma studio
```

## Schema Design

The schema is optimized for sports analytics, with dedicated relations for:

- **Matches**: Core metadata, video assets, and analysis status.
- **Events**: Temporal markers, event types, and detection confidence.
- **Teams**: Jersey colors and team identifiers.
- **Users**: Authentication and ownership.

## Usage

```ts
import { prisma } from "@matcha/database";

const match = await prisma.match.findUnique({ ... });
```
