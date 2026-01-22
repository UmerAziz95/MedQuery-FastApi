# Docker Desktop Troubleshooting

## Issue
Docker Desktop API returning 500 errors. This is a Docker Desktop issue, not a code issue.

## Quick Fixes

### Option 1: Restart Docker Desktop
1. Right-click Docker Desktop icon in system tray
2. Click "Restart"
3. Wait for Docker Desktop to fully start
4. Try commands again

### Option 2: Restart Docker Service
```powershell
# Stop Docker Desktop
Stop-Process -Name "Docker Desktop" -Force

# Wait a few seconds
Start-Sleep -Seconds 5

# Start Docker Desktop again (or restart from Start Menu)
```

### Option 3: Reset Docker Desktop
1. Open Docker Desktop
2. Settings → Troubleshoot → Reset to factory defaults
3. Restart Docker Desktop

## Manual Container Management (If Docker Compose Fails)

If Docker Compose continues to fail, you can manage containers manually:

```powershell
# Stop containers manually
docker stop medquery-api pgvector-postgres

# Remove containers
docker rm medquery-api pgvector-postgres

# Rebuild image
docker build -t medquery-fastapi-api .

# Start containers manually
docker run -d --name pgvector-postgres -p 5433:5432 -e POSTGRES_DB=medquery -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -v pgvector_data:/var/lib/postgresql/data pgvector/pgvector:0.6.2-pg16

docker run -d --name medquery-api --env-file .env -p 8000:8000 --link pgvector-postgres:postgres --memory="4g" medquery-fastapi-api
```

## After Docker Desktop is Fixed

Once Docker Desktop is working again:

```powershell
# Rebuild and start
docker compose down
docker compose build api
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f api
```
