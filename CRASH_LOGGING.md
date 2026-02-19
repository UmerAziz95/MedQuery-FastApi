# Crash Logging System

## Overview

The application now includes a comprehensive crash logging system that captures detailed information about crashes, memory issues, and errors to help debug application failures.

## Features

### 1. **Automatic Crash Detection**
- Detects and logs all critical crashes (MemoryError, SystemError, etc.)
- Captures unhandled exceptions with full context
- Logs signal-based terminations (SIGTERM, SIGINT)

### 2. **Detailed Crash Reports**
Each crash log includes:
- **System Information**: Memory usage, CPU, process details, system resources
- **Exception Details**: Full stack trace, exception type, message, arguments
- **Garbage Collection Info**: GC statistics and thresholds
- **Operation Context**: What was happening when the crash occurred
- **Additional Info**: Environment variables, custom context data

### 3. **Memory Pressure Monitoring**
- Continuous logging of memory pressure events
- Tracks memory usage against thresholds
- Helps identify memory leaks and trends

### 4. **Integration Points**
The crash logger is integrated into:
- Main application exception handlers
- Document ingestion service
- Background task processing
- Signal handlers

## Log File Locations

### Crash Reports
- **Location**: `logs/crashes/crash_YYYYMMDD_HHMMSS_PID.log`
- **Format**: Detailed text reports with all crash information
- **Naming**: `crash_20260122_143052_12345.log` (timestamp + process ID)

### Memory Pressure Log
- **Location**: `logs/crashes/memory_pressure.log`
- **Format**: Continuous append log of memory events
- **Content**: Timestamp, memory usage, thresholds, context

## Example Crash Log Structure

```
================================================================================
CRASH REPORT: crash_20260122_143052_12345
================================================================================

SYSTEM INFORMATION
--------------------------------------------------------------------------------
timestamp: 2026-01-22T14:30:52.123456
python_version: 3.11.0
platform: linux
executable: /usr/local/bin/python
process_id: 12345
memory_rss_mb: 3500.5
memory_vms_mb: 4200.3
memory_percent: 87.5
cpu_percent: 45.2
num_threads: 8
system_memory_total_gb: 4.0
system_memory_available_gb: 0.2
system_memory_used_gb: 3.8
system_memory_percent: 95.0

EXCEPTION DETAILS
--------------------------------------------------------------------------------
Type: MemoryError
Message: Memory threshold exceeded. Stopping processing to prevent crash.
Args: ('Memory threshold (2.0GB) exceeded. Stopping processing to prevent crash.',)

FULL TRACEBACK:
Traceback (most recent call last):
  File "/app/app/services/rag_ingest_service.py", line 145, in ingest_document
    await process_small_batch(pending_chunks)
  ...
MemoryError: Memory threshold exceeded

GARBAGE COLLECTION INFO
--------------------------------------------------------------------------------
gc_counts: (123, 45, 12)
gc_stats: [...]
gc_threshold: (700, 10, 10)

OPERATION CONTEXT
--------------------------------------------------------------------------------
operation: document_ingestion
document_id: ea989841-791e-4680-8437-e79e3d3915cf
document_filename: file-sample_150kB.pdf
file_type: pdf
pages_processed: 15
chunks_processed: 234
embedding_batches: 24

ADDITIONAL INFORMATION
--------------------------------------------------------------------------------
chunk_words: 400
overlap_words: 40
file_path: /app/storage/file-sample_150kB.pdf
file_size_bytes: 142786
================================================================================
End of crash report
================================================================================
```

## How to Use Crash Logs for Debugging

### 1. **Identify the Crash Type**
Look at the exception type:
- `MemoryError` → Out of memory issue
- `SystemError` → System-level failure
- `KeyboardInterrupt` → Manual termination
- Other exceptions → Application logic errors

### 2. **Check Memory Statistics**
Review memory usage at crash time:
- **memory_rss_mb**: Actual memory used by process
- **memory_percent**: Percentage of available memory
- **system_memory_percent**: System-wide memory usage
- Compare with thresholds (2GB for process, 4GB for container)

### 3. **Review Operation Context**
Check what was happening:
- **operation**: What operation was running
- **pages_processed**: How many pages were processed
- **chunks_processed**: How many chunks were created
- **document_filename**: Which file caused the issue

### 4. **Analyze the Stack Trace**
Find where the error occurred:
- Look for the first line in your application code
- Check the call chain leading to the error
- Identify the specific function/method that failed

### 5. **Check Memory Pressure Log**
Review `memory_pressure.log` for trends:
- See if memory was gradually increasing
- Identify when memory pressure started
- Check if it's a pattern or one-time event

## Common Crash Scenarios

### Memory Error During Document Processing
**Symptoms**: MemoryError with high memory_rss_mb
**Solution**: 
- File is too large → Split into smaller files
- Batch size too large → Already optimized to 10 chunks
- Memory leak → Check for objects not being deleted

### System Memory Exhausted
**Symptoms**: system_memory_percent > 90%
**Solution**:
- Increase container memory limit
- Reduce concurrent document processing
- Check for memory leaks in other processes

### Processing Timeout
**Symptoms**: TimeoutError in background task
**Solution**:
- File is too complex → Simplify or split
- Network issues → Check embedding API connectivity
- System overload → Reduce load

## Log Retention and Cleanup

- Crash logs are kept indefinitely
- Memory pressure log is appended continuously
- Recommended: Archive logs older than 30 days
- Manual cleanup: Delete old `crash_*.log` files

## Integration in Code

The crash logger is automatically used in:

1. **Main Application** (`app/main.py`)
   - Global exception handler
   - Signal handlers

2. **Document Ingestion** (`app/services/rag_ingest_service.py`)
   - Memory error handling
   - Processing errors
   - Memory pressure monitoring

3. **Background Tasks** (`app/api/routes/documents.py`)
   - Background processing errors
   - Memory errors in async tasks

## Manual Crash Logging

You can also manually log crashes in your code:

```python
from app.core.crash_logger import crash_logger

try:
    # Your code here
    pass
except Exception as e:
    crash_logger.log_crash(
        e, type(e), e.__traceback__,
        context={"operation": "my_operation", "custom_data": "value"},
        additional_info={"extra": "info"}
    )
```

## Troubleshooting

### Logs Not Being Created
- Check `logs/crashes/` directory exists and is writable
- Verify application has write permissions
- Check application logs for crash logger errors

### Missing Information in Logs
- Ensure `psutil` is installed for memory information
- Check that exceptions have tracebacks
- Verify context is being passed correctly

### Log Files Too Large
- Logs are text-based and typically < 100KB per crash
- If files are very large, check for excessive context data
- Consider archiving old logs

## Best Practices

1. **Regular Review**: Check crash logs regularly to identify patterns
2. **Monitor Memory Pressure**: Watch `memory_pressure.log` for trends
3. **Archive Old Logs**: Keep logs for analysis but archive old ones
4. **Document Solutions**: When you fix an issue, note it in the log or documentation
5. **Share Relevant Logs**: When reporting bugs, include relevant crash logs (after removing sensitive data)

## Security Considerations

Crash logs may contain:
- File paths and names
- System information
- Environment variables (sensitive ones are excluded)
- Document metadata

**Do not**:
- Commit crash logs to version control (already in .gitignore)
- Share crash logs publicly without review
- Store logs in publicly accessible locations

## Summary

The crash logging system provides comprehensive debugging information to help identify and fix application crashes. All critical errors are automatically logged with full context, making it easier to diagnose and resolve issues.

For questions or issues with the crash logging system, check the logs themselves or review the `app/core/crash_logger.py` source code.
