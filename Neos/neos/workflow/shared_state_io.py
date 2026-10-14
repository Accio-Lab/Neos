
from __future__ import annotations

import errno
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

logger = logging.getLogger(__name__)
TRANSIENT_ERRNOS = {errno.ESTALE, errno.EINTR, errno.EAGAIN, errno.ETIMEDOUT}


def _retry(operation, path: Path):
    attempts = max(1, min(60, int(os.getenv("NEOS_STATE_IO_ATTEMPTS", "12"))))
    for attempt in range(attempts):
        try:
            return operation()
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            retryable = not isinstance(exc, OSError) or exc.errno in TRANSIENT_ERRNOS
            if not retryable or attempt + 1 == attempts:
                raise
            if attempt in {0, 3, 7}:
                logger.warning("shared state retry path=%s attempt=%s/%s error=%s",
                               path, attempt + 1, attempts, exc)
            time.sleep(min(0.1 * 2 ** min(attempt, 4), 1.0))


def read_json(path: Path, default=None):
    path = Path(path)
    try:
        return _retry(lambda: json.loads(path.read_text(encoding="utf-8")), path)
    except FileNotFoundError:
        return default


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False) + "\n"

    def publish():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        try:
            with temp.open("w", encoding="utf-8") as out:
                out.write(payload)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                
                pass

    _retry(publish, path)
