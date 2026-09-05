"""
Structured logging + timing/VRAM instrumentation for the VDocRAG demo.

Design rationale (see handoff doc Section 6): plain stdlib `logging` writing
JSONL, not OpenTelemetry. This is a single-process Colab notebook with no
second service to correlate traces against, so OTel's actual value
proposition doesn't apply here. JSONL gives the same debugging value and is
directly `pandas.read_json(lines=True)`-able, which is what the page-cap
formula (limits.py) and the VRAM-sharing decision both need: real numbers,
not guesses.

Usage:
    from vdocrag_app.telemetry import setup_logging, log_call

    setup_logging("session_2026_09_04")

    class VDocRetriever:
        @log_call("retriever")
        def encode_document(self, image):
            ...
"""
from __future__ import annotations

import functools
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import torch
    # A broken/CUDA-less torch install can raise things other than
    # ImportError at import time (e.g. OSError from a CUDA .so lookup) --
    # catching broadly here means this module degrades to "no VRAM stats"
    # rather than crashing every module that imports it, in any environment
    # where torch is present but not fully usable.
    _TORCH_AVAILABLE = torch.cuda.is_available() if hasattr(torch, "cuda") else False
    _TORCH_IMPORTED = True
except Exception:
    _TORCH_AVAILABLE = False
    _TORCH_IMPORTED = False


def _default_log_dir() -> Path:
    """Prefer Drive persistence in Colab; fall back to local ./logs elsewhere
    (e.g. when running ingest.py standalone outside Colab, as in local tests)."""
    drive_path = Path("/content/drive/MyDrive/vdocrag-project/logs")
    if drive_path.parent.parent.exists():  # /content/drive/MyDrive exists -> Drive is mounted
        return drive_path
    return Path("./logs")


class JsonlHandler(logging.Handler):
    """Writes one JSON object per line: timestamp, level, message, plus any
    structured fields passed via `extra={"extra_fields": {...}}`."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            entry.update(record.extra_fields)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def setup_logging(session_name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Call once per session/notebook run. Idempotent-ish: clears existing
    handlers first so re-running a Colab cell doesn't duplicate log lines."""
    log_dir = log_dir or _default_log_dir()
    logger = logging.getLogger("vdocrag")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    jsonl_path = log_dir / f"{session_name}.jsonl"
    logger.addHandler(JsonlHandler(jsonl_path))
    logger.info(f"Logging initialized: {jsonl_path}")
    return logger


logger = logging.getLogger("vdocrag")


def _vram_gb() -> float:
    if _TORCH_AVAILABLE:
        return round(torch.cuda.memory_allocated() / 1e9, 3)
    return 0.0


def _vram_peak_gb() -> float:
    if _TORCH_AVAILABLE:
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return 0.0


def log_call(component: str):
    """Decorator: logs timing + VRAM delta + peak for any function. VRAM
    fields are 0.0 on CPU-only steps (e.g. ingest.py) rather than erroring —
    this module is imported by both GPU and non-GPU code paths."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _TORCH_AVAILABLE:
                torch.cuda.reset_peak_memory_stats()
            mem_before = _vram_gb()
            t0 = time.time()
            status, err = "ok", None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                status, err = "error", repr(e)
                raise
            finally:
                duration_ms = (time.time() - t0) * 1000
                fields = {
                    "component": component,
                    "function": fn.__name__,
                    "duration_ms": round(duration_ms, 1),
                    "vram_before_gb": mem_before,
                    "vram_after_gb": _vram_gb(),
                    "vram_peak_gb": _vram_peak_gb(),
                    "status": status,
                }
                if err:
                    fields["error"] = err
                logger.info(
                    f"{component}.{fn.__name__} {status} ({duration_ms:.0f}ms)",
                    extra={"extra_fields": fields},
                )

        return wrapper

    return decorator
