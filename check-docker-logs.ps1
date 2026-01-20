# PowerShell script to check Docker container logs

Write-Host "=== Docker Container Status ===" -ForegroundColor Cyan
docker compose ps

Write-Host "`n=== API Container Logs (last 50 lines) ===" -ForegroundColor Cyan
docker compose logs --tail=50 api

Write-Host "`n=== Postgres Container Logs (last 20 lines) ===" -ForegroundColor Cyan
docker compose logs --tail=20 postgres

Write-Host "`n=== To follow logs in real-time, run: ===" -ForegroundColor Yellow
Write-Host "  docker compose logs -f api"
Write-Host "`n=== To check if migrations are needed: ===" -ForegroundColor Yellow
Write-Host "  docker compose exec api alembic current"
Write-Host "`n=== To run migrations: ===" -ForegroundColor Yellow
Write-Host "  docker compose exec api alembic upgrade head"
