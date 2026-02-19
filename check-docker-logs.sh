#!/bin/bash
# Quick script to check Docker container logs

echo "=== Docker Container Status ==="
docker compose ps

echo ""
echo "=== API Container Logs (last 50 lines) ==="
docker compose logs --tail=50 api

echo ""
echo "=== Postgres Container Logs (last 20 lines) ==="
docker compose logs --tail=20 postgres

echo ""
echo "=== To follow logs in real-time, run: ==="
echo "  docker compose logs -f api"
echo ""
echo "=== To check if migrations are needed: ==="
echo "  docker compose exec api alembic current"
echo ""
echo "=== To run migrations: ==="
echo "  docker compose exec api alembic upgrade head"
