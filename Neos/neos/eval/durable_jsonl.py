from __future__ import annotations
from neos.identifiers import new_id, path_namespace
import errno
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable
RETRYABLE_PUBLISH_ERRNOS = frozenset({errno.EAGAIN, errno.EBUSY, errno.EIO, errno.ENOSPC, errno.ESTALE, errno.ETIMEDOUT})

class DurablePublishError(RuntimeError):
    pass

def trajectory_row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get('bench') or ''), str(row.get('sid') or ''), int(row.get('evaluation_repeat') or 1))

def _atomic_local_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=448)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 384)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def _find_last_newline(handle, size: int, chunk_size: int=1024 * 1024) -> int:
    cursor = size
    suffix = b''
    while cursor > 0:
        take = min(chunk_size, cursor)
        cursor -= take
        handle.seek(cursor)
        block = handle.read(take)
        combined = block + suffix
        index = combined.rfind(b'\n')
        if index >= 0:
            return cursor + index
        suffix = combined
    return -1

def repair_truncated_jsonl_tail(path: str | Path, *, quarantine_root: str | Path | None=None) -> dict[str, Any]:
    pass
    target = Path(path).expanduser().resolve()
    report: dict[str, Any] = {'path': str(target), 'repaired': False, 'discarded_bytes': 0, 'quarantine_path': None}
    if not target.is_file() or target.stat().st_size == 0:
        return report
    with target.open('r+b', buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            size = os.fstat(handle.fileno()).st_size
            handle.seek(size - 1)
            if handle.read(1) == b'\n':
                return report
            last_newline = _find_last_newline(handle, size)
            tail_start = last_newline + 1
            handle.seek(tail_start)
            tail = handle.read(size - tail_start)
            try:
                json.loads(tail)
            except (ValueError, UnicodeDecodeError):
                root = Path(quarantine_root or os.getenv('NEOS_EVAL_CORRUPT_TAIL_ROOT', '/tmp/neos-eval-corrupt-tails')).expanduser().resolve()
                digest = new_id()
                quarantine = root / f'{digest}.bin'
                _atomic_local_write(quarantine, tail)
                os.ftruncate(handle.fileno(), tail_start)
                os.fsync(handle.fileno())
                report.update({'repaired': True, 'discarded_bytes': len(tail), 'quarantine_path': str(quarantine)})
                return report
            handle.seek(0, os.SEEK_END)
            handle.write(b'\n')
            os.fsync(handle.fileno())
            report['appended_missing_newline'] = True
            return report
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

def _safe_append_batch(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 384)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        start = os.lseek(descriptor, 0, os.SEEK_END)
        if start:
            with target.open('rb') as reader:
                reader.seek(start - 1)
                if reader.read(1) != b'\n':
                    raise DurablePublishError(f'refusing to append after an unterminated JSONL tail: {target}')
        view = memoryview(payload)
        written = 0
        try:
            while written < len(view):
                written += os.write(descriptor, view[written:])
            os.fsync(descriptor)
        except BaseException:
            try:
                os.ftruncate(descriptor, start)
                os.fsync(descriptor)
            except OSError:
                pass
            raise
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

class DurableJSONLPublisher:
    pass

    def __init__(self, target: str | Path, *, existing_rows: Iterable[dict[str, Any]]=(), local_root: str | Path | None=None, before_publish: Callable[[dict[str, Any]], Any] | None=None, batch_rows: int | None=None, key_fn: Callable[[dict[str, Any]], Any] | None=None) -> None:
        self.target = Path(target).expanduser().resolve()
        base = Path(local_root or os.getenv('NEOS_EVAL_SPOOL_ROOT', '/tmp/neos-eval-spool')).expanduser().resolve()
        token = path_namespace(self.target)
        self.root = base / token
        self.pending_root = self.root / 'pending'
        self.pending_root.mkdir(parents=True, exist_ok=True, mode=448)
        self.before_publish = before_publish
        self.key_fn = key_fn or trajectory_row_key
        self._row_paths = {}
        self._pending_path_lock = threading.Lock()
        for path in self.pending_root.glob('*.json'):
            try:
                row = json.loads(path.read_text(encoding='utf-8'))
                self._row_paths.setdefault(self.key_fn(row), path)
            except (OSError, ValueError, KeyError):
                continue
        self.batch_rows = max(1, int(batch_rows or os.getenv('NEOS_EVAL_PUBLISH_BATCH_ROWS', '16')))
        self.max_spool_bytes = max(1, int(os.getenv('NEOS_EVAL_SPOOL_MAX_BYTES', str(1024 ** 4))))
        self._known_keys = {self.key_fn(row) for row in existing_rows}
        self._known_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self._last_error: str | None = None
        self._fatal_error: str | None = None
        self._published_rows = 0
        self._publish_attempts = 0
        self._publish_errors = 0
        self._local_rows = 0
        self._local_bytes = sum((path.stat().st_size for path in self.pending_root.glob('*.json') if path.is_file()))
        self._thread = threading.Thread(target=self._run, name=f'eval-jsonl-publisher-{self.target.name}', daemon=True)
        self._thread.start()
        self._wake.set()

    def _path_for_key(self, key):
        with self._pending_path_lock:
            if key not in self._row_paths:
                self._row_paths[key] = self.pending_root / (new_id() + '.json')
            return self._row_paths[key]

    def enqueue(self, row: dict[str, Any]) -> dict[str, Any]:
        key = self.key_fn(row)
        with self._known_lock:
            if key in self._known_keys:
                return {'already_published': True, 'local_write_ms': 0.0}
        payload = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
        started = time.perf_counter()
        with self._stats_lock:
            projected = self._local_bytes + len(payload)
            if projected > self.max_spool_bytes:
                raise DurablePublishError(f'local evaluation spool would exceed {self.max_spool_bytes} bytes: {self.root}')
        path = self._path_for_key(key)
        prior_size = path.stat().st_size if path.is_file() else 0
        _atomic_local_write(path, payload)
        with self._stats_lock:
            self._local_rows += int(prior_size == 0)
            self._local_bytes += len(payload) - prior_size
        self._wake.set()
        return {'already_published': False, 'local_write_ms': (time.perf_counter() - started) * 1000, 'serialized_bytes': len(payload), 'spool_path': str(path)}

    def _pending_paths(self) -> list[Path]:
        return sorted((path for path in self.pending_root.glob('*.json') if path.is_file()))

    def _discard_pending_path(self, path: Path) -> None:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return
        path.unlink()
        with self._stats_lock:
            self._local_bytes = max(0, self._local_bytes - size)

    def _publish_paths(self, paths: list[Path]) -> None:
        rows: list[dict[str, Any]] = []
        publish_paths: list[Path] = []
        keys: list[Any] = []
        for path in paths:
            row = json.loads(path.read_bytes())
            key = self.key_fn(row)
            with self._known_lock:
                known = key in self._known_keys
            if known:
                self._discard_pending_path(path)
                continue
            rows.append(row)
            publish_paths.append(path)
            keys.append(key)
        if not rows:
            return
        if self.before_publish is not None:
            for row in rows:
                self.before_publish(row)
        payload = b''.join(((json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8') for row in rows))
        with self._stats_lock:
            self._publish_attempts += 1
        _safe_append_batch(self.target, payload)
        with self._known_lock:
            self._known_keys.update(keys)
        for path in publish_paths:
            self._discard_pending_path(path)
        with self._stats_lock:
            self._published_rows += len(rows)
            self._last_error = None

    def _run(self) -> None:
        backoff = 0.25
        while True:
            paths = self._pending_paths()
            if not paths:
                if self._stop.is_set():
                    return
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            try:
                self._publish_paths(paths[:self.batch_rows])
                backoff = 0.25
            except BaseException as exc:
                with self._stats_lock:
                    self._publish_errors += 1
                    self._last_error = f'{type(exc).__name__}: {exc}'
                    if not (isinstance(exc, OSError) and exc.errno in RETRYABLE_PUBLISH_ERRNOS) and (not isinstance(exc, DurablePublishError)):
                        self._fatal_error = self._last_error
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def pending_count(self) -> int:
        return len(self._pending_paths())

    def reset_known_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        pass
        if self.pending_count():
            raise DurablePublishError('cannot reset published keys while local rows are pending')
        with self._known_lock:
            self._known_keys = {self.key_fn(row) for row in rows}

    def flush(self, timeout: float | None=None) -> None:
        timeout = float(timeout if timeout is not None else os.getenv('NEOS_EVAL_PUBLISH_FLUSH_TIMEOUT_SECONDS', '1800'))
        deadline = time.monotonic() + max(0.0, timeout)
        self._wake.set()
        while self.pending_count():
            with self._stats_lock:
                fatal = self._fatal_error
                last = self._last_error
            if fatal:
                raise DurablePublishError(f'fatal evaluation publisher error; local spool retained at {self.root}: {fatal}')
            if time.monotonic() >= deadline:
                raise DurablePublishError(f'timed out publishing evaluation rows; local spool retained at {self.root}; last_error={last}')
            self._wake.set()
            time.sleep(0.1)

    def close(self, timeout: float | None=None) -> None:
        self.flush(timeout)
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=10.0)

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {'schema_version': 'neos-eval-local-spool-publisher', 'target': str(self.target), 'local_root': str(self.root), 'pending_rows': self.pending_count(), 'local_rows_enqueued': self._local_rows, 'local_bytes_pending': self._local_bytes, 'published_rows': self._published_rows, 'publish_attempts': self._publish_attempts, 'publish_errors': self._publish_errors, 'last_error': self._last_error}
