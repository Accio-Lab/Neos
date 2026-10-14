from __future__ import annotations
from neos.identifiers import new_id, path_namespace
import atexit
import base64
import copy
import json
import os
import socket
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from neos.eval.durable_jsonl import DurableJSONLPublisher
TRACE_SCHEMA_VERSION = 'neos-mei-trace'
ASSET_URI_PREFIX = 'mei-asset://images/'

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def _atomic_local_write(path: Path, payload: bytes) -> None:
    if path.is_file():
        return
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
        temporary_path.unlink(missing_ok=True)

def _atomic_shared_write_once(path: Path, payload: bytes) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)

def _data_url(value: str) -> tuple[str, bytes] | None:
    if not value.startswith('data:image/') or ';base64,' not in value:
        return None
    header, payload = value.split(',', 1)
    mime_type = header[5:].split(';', 1)[0].lower()
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception:
        return None
    return (mime_type, raw)

def _suffix_for_mime(mime_type: str) -> str:
    return {'image/jpeg': '.jpg', 'image/png': '.png', 'image/webp': '.webp', 'image/gif': '.gif'}.get(mime_type, '.bin')

class MEITraceStore:
    pass

    def __init__(self, shared_root: str | Path, *, local_root: str | Path | None=None) -> None:
        self.shared_root = Path(shared_root).expanduser().resolve()
        token = path_namespace(self.shared_root)
        local_base = Path(local_root or os.getenv('NEOS_MEI_TRACE_LOCAL_ROOT', '/tmp/neos-mei-traces')).expanduser().resolve()
        self.local_root = local_base / token
        self.local_asset_root = self.local_root / 'assets'
        self.local_asset_root.mkdir(parents=True, exist_ok=True, mode=448)
        self._closed = False
        self._close_lock = threading.Lock()
        layout = os.getenv('NEOS_MEI_TRACE_LAYOUT', 'single_file')
        if layout not in {'single_file', 'writer_shards'}:
            raise ValueError(f'unknown mei trace layout: {layout}')
        self.layout = layout
        if layout == 'writer_shards':
            owner = f'{socket.gethostname()}-pid{os.getpid()}-{uuid.uuid4().hex[:12]}'
            target = self.shared_root / 'record_shards' / f'{owner}.jsonl'
            spool = self.local_root / 'row-spool' / owner
            _atomic_shared_write_once(self.shared_root / 'trace_layout.json', _canonical({'schema': 'neos-trace-layout', 'records': 'record_shards/*.jsonl', 'single_file_records': 'records.jsonl', 'writer_scope': 'one-publisher-process'}))
        else:
            target = self.shared_root / 'records.jsonl'
            spool = self.local_root / 'row-spool'
        self.publisher = DurableJSONLPublisher(target, local_root=spool, before_publish=self._publish_assets, batch_rows=int(os.getenv('NEOS_MEI_TRACE_BATCH_ROWS', '32')), key_fn=lambda row: row['trace_id'])

    def _asset_path(self, digest: str, mime_type: str, *, local: bool) -> Path:
        root = self.local_asset_root if local else self.shared_root / 'assets'
        return root / digest[:2] / digest[2:4] / f'{digest}{_suffix_for_mime(mime_type)}'

    def _externalize_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        assets: dict[str, dict[str, Any]] = {}

        def visit(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): visit(item) for key, item in value.items()}
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, str):
                decoded = _data_url(value)
                if decoded is None:
                    return value
                mime_type, raw = decoded
                digest = new_id()
                local_path = self._asset_path(digest, mime_type, local=True)
                _atomic_local_write(local_path, raw)
                assets[digest] = {'asset_id': digest, 'mime_type': mime_type, 'bytes': len(raw), 'uri': ASSET_URI_PREFIX + digest, 'local_path': str(local_path)}
                return ASSET_URI_PREFIX + digest
            return value
        normalized = visit(copy.deepcopy(messages))
        return (normalized, [assets[key] for key in sorted(assets)])

    def _publish_assets(self, row: dict[str, Any]) -> None:
        for asset in row.get('assets') or []:
            digest = str(asset['asset_id'])
            mime_type = str(asset['mime_type'])
            local_path = Path(str(asset.get('local_path') or ''))
            shared_path = self._asset_path(digest, mime_type, local=False)
            if not shared_path.is_file():
                if not local_path.is_file():
                    raise FileNotFoundError(f'MEI trace asset is unavailable: {digest}')
                payload = local_path.read_bytes()
                _atomic_shared_write_once(shared_path, payload)
            if local_path.is_file():
                local_path.unlink()
            asset.pop('local_path', None)
            asset['shared_path'] = str(shared_path)

    def record(self, *, namespace: str, prompt_version: str, model: str, fallback_models: list[str] | tuple[str, ...], messages: list[dict[str, Any]], max_completion_tokens: int, raw_output: str, parsed_output: dict[str, Any] | None, timing: dict[str, Any] | None, descriptor: dict[str, Any]) -> dict[str, Any]:
        wire_messages, assets = self._externalize_messages(messages)
        input_payload = {'messages': wire_messages, 'max_completion_tokens': int(max_completion_tokens), 'descriptor': copy.deepcopy(descriptor)}
        example_id = new_id()
        trace_id = uuid.uuid4().hex
        timing = copy.deepcopy(timing or {})
        record = {'schema_version': TRACE_SCHEMA_VERSION, 'trace_id': trace_id, 'example_id': example_id, 'created_at_utc': _utc_now(), 'namespace': str(namespace), 'prompt_version': str(prompt_version), 'source_run_id': os.getenv('NEOS_MEI_TRACE_RUN_ID', os.getenv('NEOS_RUN_ID', '')), 'configured_model': str(model), 'configured_fallback_models': [str(value) for value in fallback_models], 'actual_model': timing.get('actual_model'), 'fallback_used': timing.get('fallback_used'), 'compute_called': timing.get('compute_called'), 'gateway_call': timing.get('gateway_call'), 'input': input_payload, 'assets': assets, 'output': {'raw_text': str(raw_output), 'parsed_json': copy.deepcopy(parsed_output), 'parsed': parsed_output is not None}, 'selection': {'main_trajectory_correct': None, 'owner': 'downstream_trace_id_join'}}
        publish = self.publisher.enqueue(record)
        return {'trace_id': trace_id, 'example_id': example_id, 'asset_count': len(assets), 'spool_path': publish.get('spool_path')}

    def stats(self) -> dict[str, Any]:
        return {'schema_version': TRACE_SCHEMA_VERSION, 'shared_root': str(self.shared_root), 'layout': self.layout, 'local_root': str(self.local_root), 'publisher': self.publisher.stats()}

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self.publisher.close(float(os.getenv('NEOS_MEI_TRACE_FLUSH_TIMEOUT_SECONDS', '900')))
_STORE_LOCK = threading.Lock()
_STORES: dict[str, MEITraceStore] = {}

def get_mei_trace_store() -> MEITraceStore | None:
    configured = os.getenv('NEOS_MEI_TRACE_ROOT', '').strip()
    if not configured:
        return None
    root = str(Path(configured).expanduser().resolve())
    with _STORE_LOCK:
        store = _STORES.get(root)
        if store is None:
            store = MEITraceStore(root)
            _STORES[root] = store
        return store

def close_mei_trace_stores() -> None:
    with _STORE_LOCK:
        stores = list(_STORES.values())
    for store in stores:
        try:
            store.close()
        except Exception:
            pass
atexit.register(close_mei_trace_stores)
