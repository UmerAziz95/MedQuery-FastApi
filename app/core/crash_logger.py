"""
Crash and Error Logger Module

This module provides comprehensive crash logging with detailed system information,
memory usage, stack traces, and context to help debug application failures.
"""
import gc
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

# Directory for crash logs (use /app/logs/crashes in Docker so volume mount works)
CRASH_LOG_DIR = Path(os.environ.get("CRASH_LOG_DIR", "logs/crashes"))
try:
    CRASH_LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Write one line at import so a file exists even if process is OOM-killed before lifespan
    _progress_log = CRASH_LOG_DIR / "ingest_progress.log"
    with open(_progress_log, "a", encoding="utf-8") as _f:
        _f.write(f"[{datetime.utcnow().isoformat()}] step=module_loaded pid={os.getpid()}\n")
        _f.flush()
        try:
            os.fsync(_f.fileno())
        except Exception:
            pass
except Exception as _e:
    # Don't fail import; logging not configured yet
    import warnings
    warnings.warn(f"Crash log dir not ready: {_e}", UserWarning, stacklevel=0)


class CrashLogger:
    """Comprehensive crash and error logging system."""
    
    def __init__(self, log_dir: Path = CRASH_LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        info = {
            "timestamp": datetime.utcnow().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "executable": sys.executable,
        }
        
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()
                
                info.update({
                    "process_id": os.getpid(),
                    "memory_rss_mb": mem_info.rss / (1024 * 1024),
                    "memory_vms_mb": mem_info.vms / (1024 * 1024),
                    "memory_percent": process.memory_percent(),
                    "cpu_percent": process.cpu_percent(interval=0.1),
                    "num_threads": process.num_threads(),
                    "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None,
                })
                
                # System-wide memory
                sys_mem = psutil.virtual_memory()
                info.update({
                    "system_memory_total_gb": sys_mem.total / (1024 ** 3),
                    "system_memory_available_gb": sys_mem.available / (1024 ** 3),
                    "system_memory_used_gb": sys_mem.used / (1024 ** 3),
                    "system_memory_percent": sys_mem.percent,
                })
            except Exception as e:
                info["system_info_error"] = str(e)
        
        return info
    
    def get_gc_info(self) -> Dict[str, Any]:
        """Get garbage collection statistics."""
        try:
            counts = gc.get_count()
            stats = gc.get_stats()
            return {
                "gc_counts": counts,
                "gc_stats": stats,
                "gc_threshold": gc.get_threshold(),
            }
        except Exception as e:
            return {"gc_info_error": str(e)}
    
    def get_exception_details(self, exc: Exception, exc_type: type, exc_traceback) -> Dict[str, Any]:
        """Extract detailed exception information."""
        return {
            "exception_type": exc_type.__name__,
            "exception_message": str(exc),
            "exception_args": str(exc.args) if hasattr(exc, 'args') else None,
            "traceback": "".join(traceback.format_exception(exc_type, exc, exc_traceback)),
            "traceback_lines": traceback.format_exception(exc_type, exc, exc_traceback),
        }
    
    def get_recent_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get recent operation context."""
        if context is None:
            context = {}
        
        return {
            "context": context,
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if not any(sensitive in k.upper() for sensitive in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN'])
            },
        }
    
    def log_crash(
        self,
        exc: Exception,
        exc_type: type,
        exc_traceback,
        context: Optional[Dict[str, Any]] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log a crash with comprehensive details.
        
        Returns the path to the crash log file.
        """
        timestamp = datetime.utcnow()
        crash_id = f"crash_{timestamp.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        log_file = self.log_dir / f"{crash_id}.log"
        
        crash_report = {
            "crash_id": crash_id,
            "severity": "CRITICAL",
            "system_info": self.get_system_info(),
            "exception": self.get_exception_details(exc, exc_type, exc_traceback),
            "gc_info": self.get_gc_info(),
            "context": self.get_recent_context(context),
        }
        
        if additional_info:
            crash_report["additional_info"] = additional_info
        
        # Write to file
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"CRASH REPORT: {crash_id}\n")
                f.write("=" * 80 + "\n\n")
                
                # System Info
                f.write("SYSTEM INFORMATION\n")
                f.write("-" * 80 + "\n")
                for key, value in crash_report["system_info"].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Exception Details
                f.write("EXCEPTION DETAILS\n")
                f.write("-" * 80 + "\n")
                exc_info = crash_report["exception"]
                f.write(f"Type: {exc_info['exception_type']}\n")
                f.write(f"Message: {exc_info['exception_message']}\n")
                if exc_info['exception_args']:
                    f.write(f"Args: {exc_info['exception_args']}\n")
                f.write("\n")
                f.write("FULL TRACEBACK:\n")
                f.write(exc_info['traceback'])
                f.write("\n\n")
                
                # GC Info
                f.write("GARBAGE COLLECTION INFO\n")
                f.write("-" * 80 + "\n")
                for key, value in crash_report["gc_info"].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Context
                if crash_report["context"].get("context"):
                    f.write("OPERATION CONTEXT\n")
                    f.write("-" * 80 + "\n")
                    for key, value in crash_report["context"]["context"].items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                # Additional Info
                if additional_info:
                    f.write("ADDITIONAL INFORMATION\n")
                    f.write("-" * 80 + "\n")
                    for key, value in additional_info.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write(f"End of crash report\n")
                f.write("=" * 80 + "\n")
        except Exception as write_error:
            logger.error(f"Failed to write crash log: {write_error}")
        
        # Also log to standard logger
        logger.critical(
            f"CRASH DETECTED: {crash_id}\n"
            f"Exception: {exc_type.__name__}: {exc}\n"
            f"Memory: {crash_report['system_info'].get('memory_rss_mb', 'N/A')}MB RSS\n"
            f"Log file: {log_file}"
        )
        
        return str(log_file)
    
    def log_error(
        self,
        exc: Exception,
        exc_type: type,
        exc_traceback,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "ERROR",
    ) -> Optional[str]:
        """
        Log a non-critical error with details.
        
        Returns the path to the error log file if severity is high enough.
        """
        if severity in ["CRITICAL", "FATAL"]:
            return self.log_crash(exc, exc_type, exc_traceback, context)
        
        # For non-critical errors, just log to standard logger with details
        system_info = self.get_system_info()
        exc_details = self.get_exception_details(exc, exc_type, exc_traceback)
        
        logger.error(
            f"ERROR: {exc_type.__name__}: {exc}\n"
            f"Memory: {system_info.get('memory_rss_mb', 'N/A')}MB RSS "
            f"({system_info.get('memory_percent', 'N/A')}%)\n"
            f"Context: {context}\n"
            f"Traceback: {exc_details['traceback']}"
        )
        
        return None
    
    def log_memory_pressure(
        self,
        current_memory_mb: float,
        threshold_mb: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log memory pressure warnings."""
        system_info = self.get_system_info()
        
        logger.warning(
            f"MEMORY PRESSURE: {current_memory_mb:.1f}MB used (threshold: {threshold_mb:.1f}MB)\n"
            f"System memory: {system_info.get('system_memory_percent', 'N/A')}% used\n"
            f"Process memory: {system_info.get('memory_percent', 'N/A')}%\n"
            f"Context: {context}"
        )
        
        # Write to separate memory log
        memory_log = self.log_dir / "memory_pressure.log"
        try:
            with open(memory_log, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.utcnow().isoformat()}] ")
                f.write(f"Memory: {current_memory_mb:.1f}MB / {threshold_mb:.1f}MB, ")
                f.write(f"System: {system_info.get('system_memory_percent', 'N/A')}%, ")
                f.write(f"Process: {system_info.get('memory_percent', 'N/A')}%\n")
                if context:
                    f.write(f"  Context: {context}\n")
        except Exception as e:
            logger.error(f"Failed to write memory log: {e}")

    def write_progress(self, step: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Write a progress line to ingest_progress.log and flush.
        Use this before/after heavy operations so that when OOM kills the process (exit 137),
        the last written step is visible in logs/crashes on the host.
        """
        progress_log = self.log_dir / "ingest_progress.log"
        try:
            info = self.get_system_info()
            mem_mb = info.get("memory_rss_mb")
            mem_pct = info.get("memory_percent")
            line = (
                f"[{datetime.utcnow().isoformat()}] step={step} "
                f"memory_rss_mb={mem_mb} memory_percent={mem_pct}"
            )
            if context:
                line += " " + " ".join(f"{k}={v}" for k, v in context.items())
            line += "\n"
            with open(progress_log, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                try:
                    os.fsync(f.fileno())  # Force write to disk (visible on host volume)
                except Exception:
                    pass
            logger.info(f"Progress: {step} (memory: {mem_mb}MB)")
        except Exception as e:
            logger.warning(f"Failed to write progress log: {e}")


# Global instance
crash_logger = CrashLogger()
