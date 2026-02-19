# Quick Start After Docker Desktop Restart

## Step 1: Wait for Docker Desktop
Wait 30-60 seconds for Docker Desktop to fully start (check system tray icon).

## Step 2: Rebuild and Start
```powershell
cd D:\python\MedQuery-FastApi
docker compose down
docker compose build api
docker compose up -d
```

## Step 3: Check Status
```powershell
docker compose ps
docker compose logs api --tail=20
```

## Step 4: Test Upload
1. Open http://localhost:8000/rag-builder
2. Upload a small PDF
3. Monitor logs: `docker compose logs -f api`
4. Check for:
   - "Using PyMuPDF (fitz) for memory-efficient PDF processing"
   - Memory warnings (if any)
   - "Document successfully indexed"

## All Fixes Applied ✅

1. ✅ PyMuPDF (memory-efficient PDF library)
2. ✅ Memory monitoring (psutil)
3. ✅ Embeddings in batches of 25
4. ✅ Database writes in batches of 50
5. ✅ Aggressive garbage collection
6. ✅ 4GB container memory limit
7. ✅ Better error handling

The system should now process documents without crashing!
