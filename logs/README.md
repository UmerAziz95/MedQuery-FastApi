# Logs Directory

This directory contains crash logs, progress logs, and **comprehensive ingestion logs** for document upload and processing. With Docker volume **`./logs:/app/logs`**, **`CRASH_LOG_DIR=/app/logs/crashes`**, and **`INGESTION_LOG_DIR=/app/logs`**, the API writes logs here. If no files appear, ensure the API container is running and the volume is mounted.

## Directory Structure

```
logs/
├── ingestion.log      # Step-by-step ingestion log (every step from upload to indexed)
├── chat.log           # Step-by-step chat log (query → embed → RAG → prompt → GPT → response)
├── crashes/           # ingest_progress.log (OOM), crash_*.log, memory_pressure.log
│   ├── ingest_progress.log   # Last step before OOM (exit 137)
│   ├── crash_YYYYMMDD_HHMMSS_PID.log
│   └── memory_pressure.log
└── README.md          # This file
```

## Chat Log (`chat.log`)

**One JSON line per step.** Every step of the RAG chat flow is logged:

- **CHAT_REQUEST_RECEIVED** – Request hit the API (business_client_id, workspace_id, query)
- **CHAT_BUSINESS_FOUND** / **CHAT_WORKSPACE_FOUND** / **CHAT_CONFIG_FOUND** – Lookups OK
- **CHAT_QUERY_RECEIVED** – Query in system (query, query_word_count)
- **CHAT_EMBEDDING_START** / **CHAT_EMBEDDING_DONE** – Query embedded (vector_dim, vector_preview)
- **CHAT_RETRIEVAL_START** / **CHAT_RETRIEVAL_DONE** – RAG search (chunk_count, chunk_ids, chunks_matched)
- **CHAT_PROMPT_BUILT** – Final prompt for GPT (prompt_preview, lengths)
- **CHAT_OPENAI_CALL** – Call to ChatGPT API (model, temperature, max_tokens)
- **CHAT_OPENAI_RESPONSE** – GPT response (answer_preview, token counts)
- **CHAT_SAVED** – Request/response saved to DB
- **CHAT_ERROR** – Failure (step, error, error_type) so you can see exactly where it failed

## Ingestion Log (`ingestion.log`)

**One JSON line per step.** Every step of document upload and processing is logged with no steps skipped:

- **INGEST_START** – Upload started (document_id, filename, file_type, storage_path)
- **FILE_VERIFIED** – File found on disk (file_path, file_size_bytes, file_size_kb)
- **FILE_SIZE_VALIDATED** – File size within limit (max_file_size_mb)
- **INIT_COMPLETE** – Initialization done (chunk_words, overlap_words, memory_gb, memory_percent)
- **PDF_INGEST_START** – PDF detected; page-by-page flow started
- **PDF_PAGE_BY_PAGE_START** – PDF page loop started
- **PDF_OPEN** – PDF opened; page count (page_count, use_subprocess)
- **PAGE_EXTRACT_END** – Page text extracted (page, text_length, text_preview)
- **PAGE_SKIP_EMPTY** / **PAGE_SKIP_NO_CHUNKS** – Page skipped
- **CHUNKS_CREATED** – Page chunked (page, num_chunks, chunk_words, overlap_words, chunk_sizes_sample, first_chunk_previews)
- **EMBED_BATCH_START** – Embedding batch started (page, batch_size, batch_index, embedding_model)
- **EMBED_BATCH_END** – Embedding batch done (embedding_count)
- **COMMIT_END** – Batch committed to DB (page, batch_size, total_chunks)
- **PAGE_COMPLETE** – Page fully processed (page, total_chunks)
- **PDF_INGEST_END** / **PDF_INGEST_END_EMPTY** – PDF flow finished (total_pages, total_chunks, duration_seconds)
- **INGEST_END** – Ingestion finished (status=indexed|empty, total_chunks, duration_seconds)

For non-PDF: **STEP1_EXTRACT_END** (text_length, text_preview), **STEP2_CHUNKS_END**, **STEP3_EMBED_START** / **STEP3_EMBED_END**, **STEP4_INSERT_END**, **INGEST_END**.

Each line includes **memory_rss_mb** and **memory_percent** (when psutil is available). Text previews are truncated (e.g. 500 chars) for safety.

## Crash Log Files

### Crash Reports (`crash_*.log`)

Each crash report contains:

1. **System Information**
   - Timestamp
   - Python version
   - Platform information
   - Process ID
   - Memory usage (RSS, VMS, percentage)
   - CPU usage
   - Thread count
   - System-wide memory statistics

2. **Exception Details**
   - Exception type and message
   - Full stack trace
   - Exception arguments

3. **Garbage Collection Info**
   - GC counts and statistics
   - GC thresholds

4. **Operation Context**
   - What operation was being performed
   - Document/file information (if applicable)
   - Request details (if applicable)
   - Progress information (pages processed, chunks processed, etc.)

5. **Additional Information**
   - Environment variables (sensitive data excluded)
   - Custom context data

### Memory Pressure Log (`memory_pressure.log`)

Continuous log of memory pressure events showing:
- Timestamp
- Current memory usage
- Memory thresholds
- System and process memory percentages
- Operation context

## When Crash Logs Are Created

Crash logs are automatically created when:

1. **Memory Errors** - Out of memory (OOM) errors
2. **System Errors** - System-level failures
3. **Unhandled Exceptions** - Critical unhandled exceptions
4. **Signal Termination** - Application terminated by signals (SIGTERM, SIGINT)
5. **Document Processing Failures** - Critical failures during document ingestion

## How to Use Crash Logs

1. **Identify the Issue**
   - Check the exception type and message
   - Review the stack trace to find where the error occurred

2. **Check Memory Usage**
   - Look at memory statistics at the time of crash
   - Compare with memory thresholds
   - Check if system memory was exhausted

3. **Review Context**
   - See what operation was being performed
   - Check document/file information
   - Review progress information to see how much was processed

4. **Analyze Patterns**
   - Look for recurring issues
   - Check memory_pressure.log for memory trends
   - Identify common failure points

## Example: Analyzing a Memory Crash

```
1. Open the crash log file
2. Check "SYSTEM INFORMATION" section:
   - memory_rss_mb: 3500 (exceeded 2GB threshold)
   - system_memory_percent: 95% (system memory exhausted)
3. Check "OPERATION CONTEXT" section:
   - pages_processed: 15
   - chunks_processed: 234
   - This shows the crash happened during PDF processing
4. Check "EXCEPTION DETAILS":
   - MemoryError: Memory threshold exceeded
   - Stack trace shows it happened in _iter_pages_ultra_safe
```

## Log Retention

- Crash logs are kept indefinitely (manual cleanup recommended)
- Memory pressure logs are appended continuously
- Consider archiving old logs periodically

## Security Note

Crash logs may contain:
- File paths
- Document names
- System information
- Environment variables (sensitive ones are excluded)

Do not share crash logs publicly without reviewing them first.
