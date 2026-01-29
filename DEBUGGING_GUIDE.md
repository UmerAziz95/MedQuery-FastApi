# Step-by-Step Debugging Guide

## Overview

The application now has comprehensive step-by-step debugging with detailed logging at each stage of document ingestion. This helps identify exactly where and why crashes occur.

## Enhanced Logging Features

### 1. **Step-by-Step Progress Tracking**

Each step is clearly marked in logs:
```
[STEP 0] Initialization and validation...
[STEP 1] Extracting all text from document...
[STEP 2] Creating chunks from text...
[STEP 3] Generating embeddings for X chunks in batch...
[STEP 4] Inserting X chunks in single batch transaction...
```

### 2. **Memory Monitoring at Each Step**

Memory usage is logged before and after each major operation:
```
[STEP 1] ✅ Text extraction completed: 50000 chars, 4 pages in 2.1s. Memory: 1.2GB
[STEP 2] ✅ Chunking completed: 234 chunks in 0.05s. Memory: 1.3GB
[STEP 3] ✅ Embedding generation completed: 234 embeddings in 4.8s. Memory: 1.8GB
[STEP 4] ✅ Database insert completed: 234 chunks in 1.2s. Memory: 1.5GB
```

### 3. **Database Status Updates**

Document status is updated in the database at each step:
- "Initializing document processing..."
- "Step 1/4: Extracting text from document..."
- "Step 2/4: Creating chunks from X pages..."
- "Step 3/4: Generating embeddings for X chunks..."
- "Step 4/4: Storing X chunks in database..."

### 4. **Detailed Crash Logs**

When a crash occurs, a detailed log file is created in `logs/crashes/` with:
- System information (memory, CPU, process details)
- Exception details (type, message, full stack trace)
- Operation context (which step failed, progress made)
- Memory statistics at crash time
- Garbage collection info

## How to Debug Crashes

### Step 1: Check Application Logs

Look for step markers in Docker logs:
```bash
docker compose logs -f api | grep "\[STEP"
```

You'll see which step completed and which step failed:
```
[STEP 1] ✅ Text extraction completed...
[STEP 2] ✅ Chunking completed...
[STEP 3] ❌ Embedding generation failed: ...
```

### Step 2: Check Crash Logs

Crash logs are automatically created in `logs/crashes/`:
```bash
# List crash logs
ls logs/crashes/crash_*.log

# View most recent crash
cat logs/crashes/crash_*.log | tail -100
```

### Step 3: Analyze Crash Patterns

Use the crash analyzer script:
```bash
python scripts/analyze_crashes.py
```

This will show:
- Most common exception types
- Which steps fail most often
- Memory usage patterns
- Recommendations for fixes

### Step 4: Check Document Status

Query the database to see document status:
```sql
SELECT id, filename, status, meta_json 
FROM documents 
WHERE status = 'processing' OR status = 'failed'
ORDER BY created_at DESC;
```

The `meta_json` field contains the last status message showing which step was running.

## Common Failure Points

### Step 1: Text Extraction
**Symptoms:**
- Log shows `[STEP 1] ❌ Text extraction failed`
- Crash log shows `operation: document_ingestion_step1_extraction`

**Possible Causes:**
- PDF file is corrupted
- PDF is too complex (scanned images, encrypted)
- Memory exhausted during PDF opening
- PyMuPDF library issue

**Solutions:**
- Try converting PDF to text file first
- Use a simpler PDF
- Check memory usage before extraction
- Verify PDF is not encrypted

### Step 2: Chunking
**Symptoms:**
- Log shows `[STEP 2] ❌ Chunking failed`
- Crash log shows `operation: document_ingestion_step2_chunking`

**Possible Causes:**
- Text is too large (memory issue)
- Invalid chunk_words/overlap_words settings
- String processing error

**Solutions:**
- Reduce chunk_words setting
- Check text length in crash log
- Verify chunk_words > overlap_words

### Step 3: Embedding Generation
**Symptoms:**
- Log shows `[STEP 3] ❌ Embedding generation failed`
- Crash log shows `operation: document_ingestion_step3_embedding`

**Possible Causes:**
- OpenAI API timeout
- API rate limit exceeded
- Invalid API key
- Network connectivity issues
- Too many chunks (API limit)

**Solutions:**
- Check OpenAI API status
- Verify API key is valid
- Check network connectivity
- Reduce chunk count (split document)
- Check API rate limits

### Step 4: Database Insert
**Symptoms:**
- Log shows `[STEP 4] ❌ Database insert failed`
- Crash log shows `operation: document_ingestion_step4_database_insert`

**Possible Causes:**
- Database connection lost
- pgvector extension not installed
- Database disk space full
- Transaction timeout
- Vector dimension mismatch
- Too many chunks in single transaction

**Solutions:**
- Check database connection
- Verify pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector;`
- Check database disk space
- Increase transaction timeout
- Verify vector dimensions match
- Split into smaller batches

## Memory Issues

### Identifying Memory Problems

Check crash logs for:
```
memory_rss_mb: 3500.5
memory_percent: 87.5
system_memory_percent: 95.0
```

If memory > 2GB, it's a memory issue.

### Memory Issue Patterns

1. **Memory increases during extraction:**
   - PDF is loading entire document into memory
   - Solution: Use smaller PDFs or split them

2. **Memory spikes during embedding:**
   - All chunks + embeddings in memory at once
   - Solution: Process in smaller batches

3. **Memory exhausted during insert:**
   - Too many chunk objects in memory
   - Solution: Insert in smaller batches

## Debugging Workflow

1. **Upload a document**
2. **Watch logs in real-time:**
   ```bash
   docker compose logs -f api
   ```
3. **Identify which step fails** (look for `[STEP X] ❌`)
4. **Check crash log** in `logs/crashes/`
5. **Review crash log details:**
   - Exception type and message
   - Memory usage at crash
   - Which step failed
   - Progress made (chunks, pages)
6. **Apply fix** based on failure point
7. **Re-test** with same or different document

## Example Debug Session

```
# Terminal 1: Watch logs
docker compose logs -f api

# Terminal 2: Upload document
# (via web UI or API)

# Logs show:
[STEP 0] Initialization complete. Memory: 0.8GB
[STEP 1] ✅ Text extraction completed: 50000 chars, 4 pages in 2.1s. Memory: 1.2GB
[STEP 2] ✅ Chunking completed: 234 chunks in 0.05s. Memory: 1.3GB
[STEP 3] Generating embeddings for 234 chunks in batch...
[STEP 3] ❌ Embedding generation failed: TimeoutError: Request timed out

# Check crash log:
cat logs/crashes/crash_20260122_143052_12345.log

# Analysis:
# - Step 3 failed (embedding generation)
# - Memory was fine (1.3GB)
# - Error: TimeoutError
# - Solution: Check OpenAI API connectivity or increase timeout
```

## Crash Log Structure

Each crash log contains:

```
================================================================================
CRASH REPORT: crash_20260122_143052_12345
================================================================================

SYSTEM INFORMATION
--------------------------------------------------------------------------------
timestamp: 2026-01-22T14:30:52.123456
memory_rss_mb: 3500.5
memory_percent: 87.5
system_memory_percent: 95.0
...

EXCEPTION DETAILS
--------------------------------------------------------------------------------
Type: MemoryError
Message: Memory threshold (2.0GB) exceeded before embedding generation
...

OPERATION CONTEXT
--------------------------------------------------------------------------------
operation: document_ingestion_step3_embedding
document_id: ea989841-791e-4680-8437-e79e3d3915cf
chunk_count: 234
page_count: 4
...
```

## Quick Reference

| Step | What It Does | Common Failures | Solutions |
|------|--------------|-----------------|-----------|
| 0 | Initialization | File not found, invalid config | Check file path, verify config |
| 1 | Text Extraction | PDF corruption, memory | Convert PDF, split file |
| 2 | Chunking | Memory, invalid settings | Reduce chunk_words |
| 3 | Embedding | API timeout, rate limit | Check API, reduce chunks |
| 4 | Database Insert | Connection, disk space | Check DB, verify pgvector |

## Next Steps After Identifying Issue

1. **Read the crash log** - It has all the details
2. **Check the step that failed** - Review code for that step
3. **Apply the recommended solution** - Based on failure type
4. **Test with a smaller document** - To verify fix
5. **Monitor memory** - Use `memory_pressure.log` for trends

## Summary

The enhanced debugging system provides:
- ✅ Clear step-by-step progress tracking
- ✅ Memory monitoring at each step
- ✅ Detailed crash logs with full context
- ✅ Database status updates
- ✅ Crash analysis tools

This makes it easy to identify exactly where and why the application is crashing!
