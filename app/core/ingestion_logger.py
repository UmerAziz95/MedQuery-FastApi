"""
Comprehensive ingestion logger: every step of document upload and processing.
Writes to logs/ingestion.log (or INGESTION_LOG_DIR env) with timestamp, step, message, details, and memory.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Same parent as crashes so ./logs/ingestion.log on host when volume is ./logs:/app/logs
LOG_DIR = Path(os.environ.get("INGESTION_LOG_DIR", os.environ.get("LOG_DIR", "logs")))
INGESTION_LOG_FILE = LOG_DIR / "ingestion.log"
TEXT_PREVIEW_MAX = 500  # max chars of extracted text to log
CHUNK_PREVIEW_MAX = 200  # max chars per chunk preview


def _memory_info() -> dict:
    if not HAS_PSUTIL:
        return {"memory_rss_mb": None, "memory_percent": None}
    try:
        p = psutil.Process()
        rss_mb = p.memory_info().rss / (1024 * 1024)
        pct = p.memory_percent()
        return {"memory_rss_mb": round(rss_mb, 2), "memory_percent": round(pct, 2)}
    except Exception:
        return {"memory_rss_mb": None, "memory_percent": None}


def _safe_value(v: Any) -> Any:
    """Make value JSON-serializable and safe for logs."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:2000]  # limit length
    if isinstance(v, (list, tuple)):
        return [_safe_value(x) for x in v[:50]]  # limit list size
    if isinstance(v, dict):
        return {str(k): _safe_value(x) for k, x in list(v.items())[:30]}
    return str(v)[:500]


def log_ingestion(step: str, message: str, **details: Any) -> None:
    """
    Write one line to logs/ingestion.log: timestamp, step, message, details (JSON), memory.
    Flush + fsync so nothing is lost on crash/OOM.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        mem = _memory_info()
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": step,
            "message": message,
            "memory_rss_mb": mem.get("memory_rss_mb"),
            "memory_percent": mem.get("memory_percent"),
        }
        for k, v in details.items():
            if v is not None and k not in payload:
                payload[k] = _safe_value(v)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with open(INGESTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception as e:
        # Do not break ingestion if logging fails
        try:
            with open(INGESTION_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "step": "ingestion_log_error",
                        "message": str(e),
                        "failed_step": step,
                    }, ensure_ascii=False) + "\n"
                )
                f.flush()
        except Exception:
            pass


def text_preview(text: str, max_len: int = TEXT_PREVIEW_MAX) -> str:
    """Return truncated text safe for logging."""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip().replace("\n", " ").replace("\r", " ")
    return t[:max_len] + ("..." if len(t) > max_len else "")
