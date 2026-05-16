.PHONY: dev build lint test clean db-up db-down db-migrate logs help

# --- Development ---

dev: ## Start all services in development mode
	npm run dev

build: ## Build all packages and services
	npm run build

lint: ## Lint the entire monorepo
	npm run lint

format: ## Format the entire monorepo
	npm run format

# --- Docker & Infrastructure ---

up: ## Start infrastructure (DB, Redis, MinIO)
	docker compose up -d postgres redis minio

down: ## Stop all containers
	docker compose down

restart: ## Restart all containers
	docker compose restart

logs: ## Tail logs for all containers
	docker compose logs -f

# --- Database ---

db-migrate: ## Run database migrations locally
	npx turbo run db:migrate

db-studio: ## Open Prisma Studio
	npx turbo run db:studio

# --- Maintenance ---

clean: ## Remove node_modules and build artifacts
	find . -name "node_modules" -type d -prune -exec rm -rf '{}' +
	find . -name ".next" -type d -prune -exec rm -rf '{}' +
	find . -name "dist" -type d -prune -exec rm -rf '{}' +

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
