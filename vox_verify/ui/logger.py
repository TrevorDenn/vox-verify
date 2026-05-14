"""
Thread-safe JSON event logger for Vox-Verify Local.

Features:
- Append-mode writes (no full-file load)
- Auto-rotation when file exceeds 10 MB
- Thread-safe via threading.Lock
- Methods: log_event, get_recent, export
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List


_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class DetectionLogger:
    """Append-only, thread-safe JSON-Lines logger with auto-rotation."""

    def __init__(
        self,
        log_dir: str | Path = _DEFAULT_LOG_DIR,
        filename: str = "detections.json",
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._filename = filename
        self._lock = threading.Lock()
        self._path: Path = self._log_dir / filename
        # Ensure the file exists (as an empty JSONL file)
        if not self._path.exists():
            self._path.touch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(self, event: dict) -> None:
        """Append one detection event.  Rotates the file if needed."""
        # Enforce required keys with sensible defaults
        entry = {
            "timestamp": event.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            "bonafide_score": float(event.get("bonafide_score", 0.0)),
            "spoof_score": float(event.get("spoof_score", 0.0)),
            "is_spoof": bool(event.get("is_spoof", False)),
            "confidence": float(event.get("confidence", 0.0)),
            "noise_level_db": float(event.get("noise_level_db", 0.0)),
            "model_latency_ms": float(event.get("model_latency_ms", 0.0)),
        }
        line = json.dumps(entry, separators=(",", ":")) + "\n"

        with self._lock:
            self._maybe_rotate()
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def get_recent(self, n: int = 100) -> List[dict]:
        """Return the *n* most-recent events without loading the entire file."""
        if n <= 0:
            return []
        with self._lock:
            if not self._path.exists():
                return []
            lines = self._tail_lines(self._path, n)

        events: List[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def export(self, dest_path: str | Path) -> Path:
        """Copy the current log file to *dest_path* and return the resolved path."""
        dest = Path(dest_path)
        with self._lock:
            shutil.copy2(self._path, dest)
        return dest.resolve()

    @property
    def current_path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_rotate(self) -> None:
        """Rotate the log file if it exceeds _MAX_FILE_BYTES.

        Must be called while self._lock is held.
        """
        if not self._path.exists():
            self._path.touch()
            return
        if self._path.stat().st_size < _MAX_FILE_BYTES:
            return

        stem = self._path.stem
        suffix = self._path.suffix
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        rotated = self._log_dir / f"{stem}.{timestamp}{suffix}"
        self._path.rename(rotated)
        self._path = self._log_dir / self._filename
        self._path.touch()

    @staticmethod
    def _tail_lines(path: Path, n: int) -> List[str]:
        """Memory-efficient tail: read last *n* lines from file."""
        # Use a small deque-like approach with a binary read
        with path.open("rb") as fh:
            # Seek from end, grab a chunk, scan for newlines
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            if end == 0:
                return []

            chunk_size = min(8192, end)
            buffer = b""
            pos = end
            lines_found: list[bytes] = []

            while pos > 0 and len(lines_found) <= n:
                pos = max(0, pos - chunk_size)
                fh.seek(pos)
                chunk = fh.read(min(chunk_size, end - pos))
                buffer = chunk + buffer
                parts = buffer.split(b"\n")
                # Last element is an incomplete line (keep in buffer)
                buffer = parts[0]
                lines_found = parts[1:] + lines_found

            # Include any remaining buffer content
            if buffer:
                lines_found = [buffer] + lines_found

            last_n = lines_found[-n:] if len(lines_found) >= n else lines_found
            return [line.decode("utf-8", errors="replace") for line in last_n]


# Module-level singleton for convenience
_default_logger: DetectionLogger | None = None
_singleton_lock = threading.Lock()


def get_logger(
    log_dir: str | Path = _DEFAULT_LOG_DIR,
    filename: str = "detections.json",
) -> DetectionLogger:
    """Return (or create) the module-level singleton logger."""
    global _default_logger
    with _singleton_lock:
        if _default_logger is None:
            _default_logger = DetectionLogger(log_dir=log_dir, filename=filename)
    return _default_logger
