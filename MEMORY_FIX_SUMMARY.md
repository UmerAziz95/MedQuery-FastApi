# Memory Crash Fix Summary

## Problem
Container crashes with exit code 137 (OOM kill) even on small PDF files (KB size) after ~90 seconds.

## Root Causes Identified
1. **PDF Processing**: Even PyMuPDF can accumulate memory during processing
2. **Embedding Generation**: OpenAI API responses accumulate in memory
3. **Database Writes**: Vector embeddings (384 dimensions × 4 bytes = 6KB per chunk) accumulate
4. **Memory Leaks**: Objects not being garbage collected between operations

## Solutions Implemented

### 1. **Memory Monitoring** (`psutil`)
   - Added `psutil` for real-time memory monitoring
   - Check memory before/after each major operation
   - Early warning if memory exceeds 3.5GB

### 2. **Aggressive Garbage Collection**
   - `gc.collect()` after each page processed
   - `gc.collect()` after each embedding batch
   - `gc.collect()` after each database batch
   - Explicit `del` statements to free memory immediately

### 3. **Smaller Batch Sizes**
   - Embeddings: Process in batches of 25 (was all at once)
   - Database writes: Batches of 50 (was 100-200)
   - Immediate commit after each batch to free memory

### 4. **Memory-Efficient PDF Processing**
   - Using PyMuPDF (fitz) instead of pypdf
   - Process pages one at a time
   - Close PDF document immediately after processing
   - Delete page objects immediately after text extraction

### 5. **Container Memory Limits**
   - Increased to 4GB maximum
   - 1GB reserved
   - Memory optimization environment variables

## Code Changes

### `app/services/rag_ingest_service.py`
- Added `psutil` for memory monitoring
- Added `_check_memory()` and `_check_memory_safe()` methods
- Embeddings processed in batches of 25
- Database writes in batches of 50 with immediate commits
- Aggressive `gc.collect()` calls throughout

### `requirements.txt`
- Added `psutil==5.9.8` for memory monitoring

### `docker-compose.yml`
- Memory limit: 4GB
- Memory reservation: 1GB

## Expected Results
- **No more crashes**: Memory monitored and managed at every step
- **Lower memory usage**: Smaller batches + aggressive GC
- **Stable processing**: Early warnings prevent OOM kills

## If Still Crashing
If crashes persist, consider:
1. **Increase container memory** to 6-8GB
2. **Use external worker process** (Celery, RQ, etc.)
3. **Process PDFs outside container** (pre-extract text)
4. **Use streaming embeddings** (process and save incrementally)

## Testing
1. Upload a small PDF (< 100KB)
2. Monitor logs: `docker compose logs -f api`
3. Check for memory warnings in logs
4. Verify chunks are created successfully
