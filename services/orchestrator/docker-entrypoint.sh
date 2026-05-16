#!/bin/sh
set -e

echo "Running database migrations..."
npx prisma migrate deploy --schema=/app/packages/database/prisma/schema.prisma

echo "Starting application..."
exec "$@"
