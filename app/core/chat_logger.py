"""
Chat flow logger: every step of query → embed → RAG search → prompt → GPT → response.
Writes to logs/chat.log with timestamp, step, message, and details.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path(os.environ.get("CHAT_LOG_DIR", os.environ.get("LOG_DIR", "logs")))
CHAT_LOG_FILE = LOG_DIR / "chat.log"
VECTOR_PREVIEW_LEN = 8
PROMPT_PREVIEW_MAX = 800
ANSWER_PREVIEW_MAX = 500
# Max length for API request/response bodies in logs (for debugging)
API_BODY_LOG_MAX = 4000


def _sanitize_str(s: str, max_len: int = API_BODY_LOG_MAX) -> str:
    """Truncate and replace control chars so JSON log line never breaks."""
    if not s or not isinstance(s, str):
        return ""
    s = s[:max_len]
    return "".join(c if ord(c) >= 32 or c in "\n\r\t" else "\uFFFD" for c in s)


def _safe_value(v: Any) -> Any:
    """Make value JSON-serializable and safe for logs."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return _sanitize_str(v)
    if isinstance(v, (list, tuple)):
        if v and isinstance(v[0], (int, float)):
            return [round(x, 6) if isinstance(x, float) else x for x in v[:VECTOR_PREVIEW_LEN]]
        return [_safe_value(x) for x in v[:50]]
    if isinstance(v, dict):
        return {str(k): _safe_value(x) for k, x in list(v.items())[:30]}
    return str(v)[:500]


def log_chat(step: str, message: str, **details: Any) -> None:
    """
    Write one line to logs/chat.log: timestamp, step, message, details (JSON).
    Flush + fsync so nothing is lost on crash.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "step": step,
            "message": message,
        }
        for k, v in details.items():
            if v is not None and k not in payload:
                payload[k] = _safe_value(v)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    except Exception as e:
        try:
            with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "step": "chat_log_error",
                        "message": str(e),
                        "failed_step": step,
                    }, ensure_ascii=False) + "\n"
                )
                f.flush()
        except Exception:
            pass
