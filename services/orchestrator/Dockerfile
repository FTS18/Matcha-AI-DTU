FROM node:22-alpine

WORKDIR /app

# Install system dependencies for Prisma
RUN apk add --no-cache openssl

# Copy root workspace files
COPY package.json package-lock.json turbo.json ./
COPY packages/ ./packages/
COPY services/orchestrator/package.json ./services/orchestrator/

# Install dependencies (only for orchestrator and packages)
RUN npm install -w services/orchestrator -w packages/database -w packages/shared -w packages/contracts

# Copy orchestrator source
COPY services/orchestrator ./services/orchestrator

# Generate Prisma client
RUN npm run postinstall -w services/orchestrator

# Build orchestrator
RUN npm run build -w services/orchestrator

WORKDIR /app/services/orchestrator

EXPOSE 4000

CMD ["npm", "run", "start:prod"]
