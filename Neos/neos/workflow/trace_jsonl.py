
import collections
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def trace_record_paths(root):
    root = Path(root)
    single_file = root / 'records.jsonl'
    if single_file.is_file(): yield single_file
    yield from sorted((root / 'record_shards').glob('*.jsonl'))


class TraceReadAudit:
    pass
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counts = collections.Counter()
        self.bytes_file = (self.root / 'bad_records.bin').open('wb')
        self.index_file = (self.root / 'reader_errors.jsonl').open('w', encoding='utf-8')

    def bad(self, path, offset, line, raw, kind, error):
        self.counts[kind] += 1
        location = self.bytes_file.tell()
        self.bytes_file.write(raw)
        self.index_file.write(json.dumps(dict(source=str(path), source_offset=offset, line=line,
            bytes=len(raw), reason=kind, error=str(error)[:400], quarantine_offset=location), ensure_ascii=True)+'\n')

    def close(self):
        self.bytes_file.close(); self.index_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def jsonl_rows(path, *, audit=None):
    pass
    path = Path(path)
    with path.open('rb') as handle:
        size = path.stat().st_size
        line = 0; errors = 0
        if audit: audit.counts['files'] += 1
        while handle.tell() < size:
            offset = handle.tell()
            raw = handle.readline(size-offset)
            if not raw: break
            line += 1
            if audit: audit.counts['lines'] += 1
            if not raw.strip(): continue
            try:
                value = json.loads(raw.decode('utf-8'))
                if not isinstance(value, dict): raise TypeError('trace row is not a JSON object')
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                kind = ('incomplete_tail' if handle.tell() >= size and not raw.endswith(b'\n') else
                        'invalid_utf8' if isinstance(exc, UnicodeDecodeError) else
                        'invalid_json' if isinstance(exc, json.JSONDecodeError) else 'non_object')
                errors += 1
                if audit: audit.bad(path, offset, line, raw, kind, exc)
                continue
            if audit: audit.counts['valid_rows'] += 1
            yield value
        if errors: logger.warning('Excluded %s malformed trace rows from %s; original file retained', errors, path)
