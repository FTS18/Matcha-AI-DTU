# Stage 1: Pruner - Minimize the context for the build
FROM node:22-bookworm-slim AS pruner
WORKDIR /app
RUN npm install -g turbo
COPY . .
RUN turbo prune orchestrator --docker

# Stage 2: Builder - Install dependencies and build
FROM node:22-bookworm-slim AS builder
RUN apt-get update && apt-get install -y openssl
WORKDIR /app

# First copy the pruned package.json and lockfile to cache dependencies
COPY --from=pruner /app/out/json/ .
COPY --from=pruner /app/out/package-lock.json ./
RUN npm install --ignore-scripts

# Copy the full source code (only for pruned packages)
COPY --from=pruner /app/out/full/ .

# Generate Prisma client and build
RUN npx prisma generate --schema=packages/database/prisma/schema.prisma
RUN npm run build --filter=orchestrator

# Prune devDependencies to reduce final image size
RUN npm prune --production

# Stage 3: Runner - Optimized production image
FROM node:22-bookworm-slim AS runner
RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Set production environment
ENV NODE_ENV=production

# Copy only the necessary files for execution from the pruned builder stage
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/packages ./packages
COPY --from=builder /app/services/orchestrator/dist ./services/orchestrator/dist
COPY --from=builder /app/services/orchestrator/package.json ./services/orchestrator/package.json
COPY services/orchestrator/docker-entrypoint.sh /usr/local/bin/

# Fix permissions and setup directories
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    mkdir -p uploads && \
    chown -R root:root /app

EXPOSE 4000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "dist/main.js"]
