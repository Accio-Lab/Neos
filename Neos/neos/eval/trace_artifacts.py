from __future__ import annotations
from neos.identifiers import new_id, path_namespace, file_key
import io
import os
import tempfile
import threading
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_RESULTS_ROOT = PROJECT_ROOT / 'eval_results'
DEFAULT_ARTIFACT_ROOT = DEFAULT_EVAL_RESULTS_ROOT / '.artifacts'
ARTIFACT_URI_PREFIX = 'artifact://images/'

def default_artifact_root() -> Path:
    configured = os.environ.get('NEOS_EVAL_ARTIFACT_ROOT')
    return Path(configured).expanduser().resolve() if configured else DEFAULT_ARTIFACT_ROOT

def _valid_artifact_id(value):
    return bool(value) and value not in {'.', '..'} and (Path(value).name == value)

class EvaluationArtifactStore:
    pass
    version = 'neos-eval-artifacts-local-spool-multicodec'

    def __init__(self, root: str | Path | None=None, *, local_root: str | Path | None=None, defer_shared: bool | None=None):
        self.root = Path(root).expanduser().resolve() if root is not None else default_artifact_root()
        self.defer_shared = str(os.getenv('NEOS_EVAL_DEFER_SHARED_ARTIFACTS', '0')).lower() in {'1', 'true', 'yes', 'on'} if defer_shared is None else bool(defer_shared)
        configured_local = local_root or os.getenv('NEOS_EVAL_LOCAL_ARTIFACT_ROOT')
        if configured_local is None:
            instance = os.getenv('NEOS_RUN_ID', 'default')
            token = path_namespace(self.root) / file_key(instance)
            configured_local = Path('/tmp/neos-eval-artifacts') / token
        self.local_root = Path(configured_local).expanduser().resolve()
        if self.defer_shared:
            self.local_root.mkdir(parents=True, exist_ok=True, mode=448)
        self.publish_workers = max(1, int(os.getenv('NEOS_EVAL_ARTIFACT_PUBLISH_WORKERS', '8')))
        self.encoding = str(os.getenv('NEOS_EVAL_ARTIFACT_ENCODING', 'png')).strip().lower()
        if self.encoding not in {'png', 'jpeg'}:
            raise ValueError('NEOS_EVAL_ARTIFACT_ENCODING must be png or jpeg')
        self.jpeg_quality = min(100, max(1, int(os.getenv('NEOS_EVAL_ARTIFACT_JPEG_QUALITY', '90'))))

    def manifest(self) -> dict[str, Any]:
        return {'version': self.version, 'root': str(self.root), 'layout': 'id[0:2]/id[2:4]/id.(png|jpg)', 'format': 'lossless PNG' if self.encoding == 'png' else f'JPEG quality={self.jpeg_quality}', 'preferred_encoding': self.encoding, 'content_addressed': False, 'deferred_shared_publish': self.defer_shared, 'local_spool_root': str(self.local_root) if self.defer_shared else None, 'shared_publish_workers': self.publish_workers}

    def path_for_id(self, artifact_id: str) -> Path:
        digest = str(artifact_id or '').lower()
        if not _valid_artifact_id(digest):
            raise ValueError('invalid artifact artifact_id')
        parent = self.root / digest[:2] / digest[2:4]
        for suffix in ('.png', '.jpg'):
            candidate = parent / f'{digest}{suffix}'
            if candidate.is_file():
                return candidate
        suffix = '.jpg' if self.encoding == 'jpeg' else '.png'
        return parent / f'{digest}{suffix}'

    def resolve_uri(self, uri: str) -> Path:
        if not str(uri).startswith(ARTIFACT_URI_PREFIX):
            raise ValueError('unsupported artifact URI')
        digest = str(uri)[len(ARTIFACT_URI_PREFIX):]
        shared = self.path_for_id(digest)
        if shared.is_file():
            return shared
        local = self.local_path_for_id(digest)
        return local if local.is_file() else shared

    def local_path_for_id(self, artifact_id: str) -> Path:
        digest = str(artifact_id or '').lower()
        if not _valid_artifact_id(digest):
            raise ValueError('invalid artifact artifact_id')
        parent = self.local_root / digest[:2] / digest[2:4]
        for suffix in ('.png', '.jpg'):
            candidate = parent / f'{digest}{suffix}'
            if candidate.is_file():
                return candidate
        suffix = '.jpg' if self.encoding == 'jpeg' else '.png'
        return parent / f'{digest}{suffix}'

    def _encoded_bytes(self, image: Image.Image) -> tuple[bytes, tuple[int, int], str]:
        rgb = image.convert('RGB')
        buffer = io.BytesIO()
        if self.encoding == 'jpeg':
            rgb.save(buffer, format='JPEG', quality=self.jpeg_quality, subsampling=0, optimize=True)
            mime_type = 'image/jpeg'
        else:
            rgb.save(buffer, format='PNG', optimize=False)
            mime_type = 'image/png'
        return (buffer.getvalue(), rgb.size, mime_type)

    @staticmethod
    def _atomic_write_once(path: Path, payload: bytes) -> None:
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.stem}.{os.getpid()}.{threading.get_ident()}.', suffix='.tmp')
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

    def save_image(self, image: Image.Image, *, image_ref: str, kind: str, source_url: str | None=None) -> dict[str, Any]:
        payload, (width, height), mime_type = self._encoded_bytes(image)
        digest = new_id()
        shared_path = self.path_for_id(digest)
        if not shared_path.is_file():
            path = self.local_path_for_id(digest) if self.defer_shared else shared_path
            self._atomic_write_once(path, payload)
        result: dict[str, Any] = {'ref': str(image_ref), 'kind': str(kind), 'uri': ARTIFACT_URI_PREFIX + digest, 'artifact_id': digest, 'mime_type': mime_type, 'width': width, 'height': height, 'bytes': len(payload)}
        if isinstance(source_url, str) and source_url.startswith(('http://', 'https://')):
            result['source_url'] = source_url
        return result

    def save_path(self, path: str | Path, *, image_ref: str, kind: str, source_url: str | None=None) -> dict[str, Any]:
        with Image.open(path) as opened:
            image = opened.convert('RGB')
        return self.save_image(image, image_ref=image_ref, kind=kind, source_url=source_url)

    @staticmethod
    def _artifact_ids(value: Any) -> set[str]:
        found: set[str] = set()

        def visit(item: Any) -> None:
            if isinstance(item, dict):
                uri = str(item.get('uri') or '')
                digest = str(item.get('artifact_id') or '').lower()
                if uri.startswith(ARTIFACT_URI_PREFIX):
                    uri_digest = uri[len(ARTIFACT_URI_PREFIX):].lower()
                    if _valid_artifact_id(uri_digest):
                        found.add(uri_digest)
                elif _valid_artifact_id(digest) and item.get('mime_type'):
                    found.add(digest)
                for child in item.values():
                    visit(child)
            elif isinstance(item, list | tuple):
                for child in item:
                    visit(child)
        visit(value)
        return found

    def publish_artifact(self, artifact_id: str) -> None:
        shared = self.path_for_id(artifact_id)
        if shared.is_file():
            local = self.local_path_for_id(artifact_id)
            if local.is_file():
                local.unlink()
            return
        local = self.local_path_for_id(artifact_id)
        if not local.is_file():
            raise FileNotFoundError(f'deferred evaluation artifact is unavailable: {artifact_id}')
        payload = local.read_bytes()
        self._atomic_write_once(shared, payload)
        if local.is_file():
            local.unlink()

    def publish_row_artifacts(self, row: dict[str, Any]) -> dict[str, Any]:
        pass
        if not self.defer_shared:
            return {'referenced': 0, 'published_or_present': 0}
        digests = sorted(self._artifact_ids(row))
        if not digests:
            return {'referenced': 0, 'published_or_present': 0}
        with ThreadPoolExecutor(max_workers=min(self.publish_workers, len(digests))) as executor:
            list(executor.map(self.publish_artifact, digests))
        return {'referenced': len(digests), 'published_or_present': len(digests)}

    def preflight(self):
        return self.manifest()

def describe_unpersisted_image(image: Image.Image, *, image_ref: str, kind: str) -> dict[str, Any]:
    pass
    width, height = image.size
    return {'ref': str(image_ref), 'kind': str(kind), 'persisted': False, 'width': int(width), 'height': int(height)}

def describe_dataset_image(image: Image.Image, path: str | Path, *, image_ref: str, source_url: str | None=None) -> dict[str, Any]:
    pass
    resolved = Path(path).resolve()
    width, height = image.size
    result: dict[str, Any] = {'ref': str(image_ref), 'kind': 'input', 'storage': 'benchmark_dataset', 'local_path': str(resolved), 'filename': resolved.name, 'width': int(width), 'height': int(height), 'bytes': resolved.stat().st_size}
    if isinstance(source_url, str) and source_url.startswith(('http://', 'https://')):
        result['source_url'] = source_url
    return result
