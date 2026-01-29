#!/usr/bin/env python3
"""
Crash Log Analyzer

Analyzes crash logs to identify patterns and root causes.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

CRASH_LOG_DIR = Path("logs/crashes")


def parse_crash_log(log_file: Path) -> Optional[Dict]:
    """Parse a crash log file and extract key information."""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        crash_info = {
            "file": str(log_file),
            "crash_id": None,
            "timestamp": None,
            "exception_type": None,
            "exception_message": None,
            "memory_rss_mb": None,
            "memory_percent": None,
            "system_memory_percent": None,
            "operation": None,
            "document_id": None,
            "step_failed": None,
            "chunks_created": None,
            "pages_processed": None,
        }
        
        # Extract crash ID
        crash_id_match = re.search(r"CRASH REPORT: (crash_\S+)", content)
        if crash_id_match:
            crash_info["crash_id"] = crash_id_match.group(1)
        
        # Extract exception type
        exc_type_match = re.search(r"Type: (\w+)", content)
        if exc_type_match:
            crash_info["exception_type"] = exc_type_match.group(1)
        
        # Extract exception message
        exc_msg_match = re.search(r"Message: (.+)", content)
        if exc_msg_match:
            crash_info["exception_message"] = exc_msg_match.group(1)
        
        # Extract memory info
        mem_rss_match = re.search(r"memory_rss_mb: ([\d.]+)", content)
        if mem_rss_match:
            crash_info["memory_rss_mb"] = float(mem_rss_match.group(1))
        
        mem_percent_match = re.search(r"memory_percent: ([\d.]+)", content)
        if mem_percent_match:
            crash_info["memory_percent"] = float(mem_percent_match.group(1))
        
        sys_mem_match = re.search(r"system_memory_percent: ([\d.]+)", content)
        if sys_mem_match:
            crash_info["system_memory_percent"] = float(sys_mem_match.group(1))
        
        # Extract operation context
        op_match = re.search(r"operation: (\S+)", content)
        if op_match:
            crash_info["operation"] = op_match.group(1)
            # Determine which step failed
            if "step1" in crash_info["operation"]:
                crash_info["step_failed"] = "Step 1: Text Extraction"
            elif "step2" in crash_info["operation"]:
                crash_info["step_failed"] = "Step 2: Chunking"
            elif "step3" in crash_info["operation"]:
                crash_info["step_failed"] = "Step 3: Embedding Generation"
            elif "step4" in crash_info["operation"]:
                crash_info["step_failed"] = "Step 4: Database Insert"
        
        # Extract document info
        doc_id_match = re.search(r"document_id: ([a-f0-9-]+)", content)
        if doc_id_match:
            crash_info["document_id"] = doc_id_match.group(1)
        
        # Extract progress info
        chunks_match = re.search(r"chunk_count: (\d+)", content)
        if chunks_match:
            crash_info["chunks_created"] = int(chunks_match.group(1))
        
        pages_match = re.search(r"page_count: (\d+)", content)
        if pages_match:
            crash_info["pages_processed"] = int(pages_match.group(1))
        
        return crash_info
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return None


def analyze_crashes() -> None:
    """Analyze all crash logs and provide summary."""
    if not CRASH_LOG_DIR.exists():
        print(f"Crash log directory not found: {CRASH_LOG_DIR}")
        return
    
    crash_files = list(CRASH_LOG_DIR.glob("crash_*.log"))
    
    if not crash_files:
        print("No crash logs found.")
        return
    
    print(f"\n{'='*80}")
    print(f"CRASH LOG ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Found {len(crash_files)} crash log(s)\n")
    
    crashes = []
    for crash_file in sorted(crash_files, reverse=True):  # Most recent first
        crash_info = parse_crash_log(crash_file)
        if crash_info:
            crashes.append(crash_info)
    
    if not crashes:
        print("No valid crash logs found.")
        return
    
    # Summary statistics
    print("SUMMARY STATISTICS")
    print("-" * 80)
    
    exception_types = {}
    steps_failed = {}
    memory_issues = 0
    
    for crash in crashes:
        # Count exception types
        exc_type = crash.get("exception_type", "Unknown")
        exception_types[exc_type] = exception_types.get(exc_type, 0) + 1
        
        # Count steps failed
        step = crash.get("step_failed", "Unknown")
        steps_failed[step] = steps_failed.get(step, 0) + 1
        
        # Count memory issues
        if crash.get("memory_rss_mb", 0) > 2000:  # > 2GB
            memory_issues += 1
    
    print(f"\nException Types:")
    for exc_type, count in sorted(exception_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {exc_type}: {count}")
    
    print(f"\nSteps Failed:")
    for step, count in sorted(steps_failed.items(), key=lambda x: x[1], reverse=True):
        print(f"  {step}: {count}")
    
    print(f"\nMemory Issues (>2GB): {memory_issues}/{len(crashes)}")
    
    # Recent crashes
    print(f"\n{'='*80}")
    print(f"RECENT CRASHES (Last 5)")
    print(f"{'='*80}\n")
    
    for crash in crashes[:5]:
        print(f"Crash ID: {crash.get('crash_id', 'Unknown')}")
        print(f"  Exception: {crash.get('exception_type', 'Unknown')}")
        print(f"  Message: {crash.get('exception_message', 'N/A')[:100]}")
        print(f"  Step Failed: {crash.get('step_failed', 'Unknown')}")
        print(f"  Memory: {crash.get('memory_rss_mb', 'N/A')}MB ({crash.get('memory_percent', 'N/A')}%)")
        print(f"  System Memory: {crash.get('system_memory_percent', 'N/A')}%")
        if crash.get('chunks_created'):
            print(f"  Progress: {crash.get('chunks_created')} chunks, {crash.get('pages_processed', 'N/A')} pages")
        print(f"  File: {crash.get('file')}")
        print()
    
    # Recommendations
    print(f"{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if memory_issues > 0:
        print("⚠️  Memory Issues Detected:")
        print("  - Reduce chunk_words setting")
        print("  - Split large documents into smaller files")
        print("  - Increase container memory limit")
        print()
    
    if steps_failed.get("Step 1: Text Extraction", 0) > 0:
        print("⚠️  Text Extraction Failures:")
        print("  - PDF may be corrupted or too complex")
        print("  - Try converting PDF to text file first")
        print()
    
    if steps_failed.get("Step 3: Embedding Generation", 0) > 0:
        print("⚠️  Embedding Generation Failures:")
        print("  - Check OpenAI API connectivity")
        print("  - Verify API key is valid")
        print("  - Check API rate limits")
        print()
    
    if steps_failed.get("Step 4: Database Insert", 0) > 0:
        print("⚠️  Database Insert Failures:")
        print("  - Check database connection")
        print("  - Verify pgvector extension is installed")
        print("  - Check database disk space")
        print()


if __name__ == "__main__":
    analyze_crashes()
