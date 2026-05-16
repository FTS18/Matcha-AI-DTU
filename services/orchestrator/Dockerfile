# Stage 1: Pruner - Minimize the context for the build
FROM node:22-alpine AS pruner
RUN apk add --no-cache libc6-compat
WORKDIR /app
RUN --mount=type=cache,target=/root/.npm npm install -g turbo
COPY . .
RUN turbo prune orchestrator --docker

# Stage 2: Builder - Install dependencies and build
FROM node:22-alpine AS builder
RUN apk add --no-cache openssl
WORKDIR /app

# First copy the pruned package.json and lockfile to cache dependencies
COPY --from=pruner /app/out/json/ .
COPY --from=pruner /app/out/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

# Copy the full source code (only for pruned packages)
COPY --from=pruner /app/out/full/ .

# Generate Prisma client and build
RUN npx prisma generate --schema=packages/database/prisma/schema.prisma
RUN npm run build --filter=orchestrator

# Stage 3: Runner - Optimized production image
FROM node:22-alpine AS runner
RUN apk add --no-cache openssl
WORKDIR /app

# Set production environment
ENV NODE_ENV=production

# Copy the pruned package.json and lockfile again for production-only install
COPY --from=pruner /app/out/json/ .
COPY --from=pruner /app/out/package-lock.json ./

# Install ONLY production dependencies
RUN --mount=type=cache,target=/root/.npm npm ci --omit=dev

# Copy only the necessary files for execution
COPY --from=builder /app/services/orchestrator/dist ./services/orchestrator/dist
# Copy relevant package code for shared packages
COPY --from=builder /app/packages ./packages
COPY --from=builder /app/services/orchestrator/package.json ./services/orchestrator/package.json
COPY services/orchestrator/docker-entrypoint.sh /usr/local/bin/

# Fix permissions
RUN chmod +x /usr/local/bin/docker-entrypoint.sh && \
    mkdir -p uploads && chown -R node:node /app

USER node
WORKDIR /app/services/orchestrator
EXPOSE 4000

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "dist/main.js"]
