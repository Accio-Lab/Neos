
from __future__ import annotations
import io
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse
from PIL import Image
from neos.eval.gateway import FatalLLMGatewayError, RetryExhaustedLLMGatewayError
from neos.workflow.mei import NeosMEI
from neos.workflow.schemas import CROP_IMAGE_MAX_PER_IMAGE, CROP_IMAGE_MAX_TOTAL, DEFAULT_NEOS_SCHEMA_VERSION, FETCH_IMAGE_MAX_URLS, IMAGE_SEARCH_MAX_QUERIES, NEOS_SCHEMA_VERSIONS, NEOS_TOOL_NAMES, REVERSE_IMAGE_SEARCH_MAX_INPUTS, WEB_VISIT_MAX_PAGES, validate_tool_arguments
from neos.workflow.visual_results import VISUAL_RESULT_SCHEMA_VERSION, output_refs, render_visual_result
from neos.tools.online_client import OnlineSearchClient
from neos.tools.runtime import ToolExecution, downscale, resolve_image_ref
logger = logging.getLogger(__name__)

def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)

def _timing_values(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(record.get(key) or 0.0) for record in records]
    return {f'{key}_sum': round(sum(values), 3), f'{key}_max': round(max(values, default=0.0), 3)}
DEFAULTS = {'text_search_max_queries': 0, 'text_search_top_k': 10, 'text_search_snippet_chars': 500, 'web_visit_max_pages': WEB_VISIT_MAX_PAGES, 'web_visit_page_chars': 20000, 'web_visit_max_image_refs': 15, 'image_search_max_queries': IMAGE_SEARCH_MAX_QUERIES, 'image_search_top_k': 5, 'image_candidate_pool_multiplier': 3, 'image_candidate_download_attempts': 3, 'web_visit_image_download_attempts': 1, 'reverse_max_images': REVERSE_IMAGE_SEARCH_MAX_INPUTS, 'reverse_top_k': 5, 'fetch_max_images': FETCH_IMAGE_MAX_URLS, 'fetch_max_side': 2048, 'crop_max_per_image': CROP_IMAGE_MAX_PER_IMAGE, 'crop_max_total': CROP_IMAGE_MAX_TOTAL, 'context_image_max_side': 1536, 'context_image_max_pixels': 786432, 'context_image_jpeg_quality': 85, 'context_image_max_aspect_ratio': 128, 'use_mei_vlm': True, 'tool_schema_version': DEFAULT_NEOS_SCHEMA_VERSION, 'reuse_loaded_images': False}

def _http_url(value: Any) -> str:
    text = str(value or '').strip()
    parsed = urlparse(text)
    return text if parsed.scheme in {'http', 'https'} and parsed.netloc else ''

def _string_list(value: Any, *, limit: int, allow_none: bool=False) -> list[str]:
    if value is None and allow_none:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('[') and stripped.endswith(']') or (stripped.startswith('{') and stripped.endswith('}')):
            raise ValueError('protocol normalization missed a complete JSON-container-looking string')
        if not stripped:
            raise ValueError('expected a non-empty string or list of strings')
        values = [stripped]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError('expected a non-empty string or list of strings')
    result: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError('list items must be non-empty strings')
        result.append(item.strip())
        if limit > 0 and len(result) >= limit:
            break
    return result

def _type_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _type_signature(item) for key, item in value.items()}
    if isinstance(value, list):
        item_types = sorted({json.dumps(_type_signature(item), sort_keys=True) for item in value})
        return {'type': 'list', 'item_types': [json.loads(item) for item in item_types]}
    if value is None:
        return 'null'
    return type(value).__name__

def _result_image_urls(item: dict[str, Any]) -> list[str]:
    pass
    urls: list[str] = []
    for value in (item.get('original'), item.get('imageUrl'), item.get('image_url'), item.get('image'), item.get('thumbnailUrl'), item.get('thumbnail_url'), item.get('thumbnail')):
        url = _http_url(value)
        if url and url not in urls:
            urls.append(url)
    return urls

def _result_source_url(item: dict[str, Any]) -> str:
    return _http_url(item.get('link') or item.get('sourceUrl') or item.get('source_url') or item.get('url'))

def _jpeg_bytes(image: Image.Image, max_side: int=1536) -> bytes:
    buffer = io.BytesIO()
    downscale(image, max_side).save(buffer, format='JPEG', quality=90)
    return buffer.getvalue()

@dataclass(frozen=True)
class _FetchRequest:
    position: int
    url: str
    caption: str

@dataclass(frozen=True)
class _CropRequest:
    position: int
    image_token: str
    box: tuple[float, float, float, float]

@dataclass(frozen=True)
class _ImagePrefetch:
    image: Image.Image | None
    attempts: int
    resolved_url: str | None = None
    urls_tried: int = 1

class NeosRuntime:
    pass

    def __init__(self, client: OnlineSearchClient | None=None, mei: NeosMEI | bool | None=None, config: dict[str, Any] | None=None):
        self.client = client or OnlineSearchClient()
        self.config = {**DEFAULTS, **(config or {})}
        reuse_env = os.getenv('NEOS_REUSE_LOADED_IMAGES')
        if reuse_env is not None:
            if reuse_env.lower() not in {'0', '1', 'false', 'true', 'off', 'on'}:
                raise ValueError('NEOS_REUSE_LOADED_IMAGES must be a boolean')
            self.config['reuse_loaded_images'] = reuse_env.lower() in {'1', 'true', 'on'}
        if self.config['tool_schema_version'] not in NEOS_SCHEMA_VERSIONS:
            raise ValueError(f"unknown Neos tool_schema_version: {self.config['tool_schema_version']!r}")
        schema_limits = {'web_visit_max_pages': WEB_VISIT_MAX_PAGES, 'image_search_max_queries': IMAGE_SEARCH_MAX_QUERIES, 'reverse_max_images': REVERSE_IMAGE_SEARCH_MAX_INPUTS, 'fetch_max_images': FETCH_IMAGE_MAX_URLS, 'crop_max_per_image': CROP_IMAGE_MAX_PER_IMAGE, 'crop_max_total': CROP_IMAGE_MAX_TOTAL}
        mismatches = {key: (self.config.get(key), expected) for key, expected in schema_limits.items() if int(self.config.get(key) or 0) != expected}
        if mismatches:
            raise ValueError(f'Neos runtime limits must exactly match the advertised tool schemas: {mismatches}')
        self._mei = mei
        self._mei_lock = threading.Lock()
        self.text_search_query_workers = int(os.getenv('NEOS_TEXT_SEARCH_QUERY_WORKERS', '384'))
        if self.text_search_query_workers <= 0:
            raise ValueError('NEOS_TEXT_SEARCH_QUERY_WORKERS must be positive')
        self._text_search_executor = ThreadPoolExecutor(max_workers=self.text_search_query_workers, thread_name_prefix='neos-text-query')

    def _mei_instance(self) -> NeosMEI | None:
        if self._mei is False or not self.config['use_mei_vlm']:
            return None
        if os.getenv('NEOS_COEV_ROOT'):
            from neos.workflow.coevolution import bound_mei_model
            model = bound_mei_model()
            with self._mei_lock:
                if self._mei is None or self._mei.model != model:
                    self._mei = NeosMEI(model=model, fallback_models=[])
                return self._mei
        if self._mei is None:
            with self._mei_lock:
                if self._mei is None:
                    self._mei = NeosMEI()
        return self._mei

    def _prefetch_one_image(self, url: str, attempts: int) -> _ImagePrefetch:
        pass
        fetch_with_retries = getattr(self.client, 'fetch_image_with_retries', None)
        if callable(fetch_with_retries):
            result = fetch_with_retries(url, attempts=max(1, int(attempts)))
            return _ImagePrefetch(image=getattr(result, 'image', None), attempts=int(getattr(result, 'attempts', attempts) or attempts), resolved_url=url if getattr(result, 'image', None) is not None else None)
        image = None
        used = 0
        for used in range(1, max(1, int(attempts)) + 1):
            image = self.client.fetch_image(url)
            if image is not None:
                break
        return _ImagePrefetch(image=image, attempts=used, resolved_url=url if image is not None else None)

    @staticmethod
    def _candidate_image_urls(candidate: dict[str, Any]) -> list[str]:
        values = [candidate.get('image_url')]
        fallbacks = candidate.get('_fallback_image_urls') or []
        if isinstance(fallbacks, (list, tuple)):
            values.extend(fallbacks)
        urls: list[str] = []
        for value in values:
            url = _http_url(value)
            if url and url not in urls:
                urls.append(url)
        return urls

    def _prefetch_one_candidate(self, candidate: dict[str, Any], attempts: int) -> _ImagePrefetch:
        total_attempts = 0
        urls_tried = 0
        for url in self._candidate_image_urls(candidate):
            urls_tried += 1
            result = self._prefetch_one_image(url, attempts)
            total_attempts += result.attempts
            if result.image is not None:
                return _ImagePrefetch(image=result.image, attempts=total_attempts, resolved_url=url, urls_tried=urls_tried)
        return _ImagePrefetch(image=None, attempts=total_attempts, resolved_url=None, urls_tried=urls_tried)

    def _prefetch_image_candidates(self, candidates: list[dict[str, Any]], *, attempts: int) -> tuple[list[_ImagePrefetch], dict[str, Any]]:
        pass
        started = time.perf_counter()
        if candidates:
            with ThreadPoolExecutor(max_workers=min(16, len(candidates))) as pool:
                fetched = list(pool.map(lambda candidate: self._prefetch_one_candidate(candidate, attempts), candidates))
        else:
            fetched = []
        return (fetched, {'image_prefetch_ms': _elapsed_ms(started), 'image_prefetch_urls': sum((record.urls_tried for record in fetched)), 'image_prefetch_successes': sum((record.image is not None for record in fetched)), 'image_prefetch_failures': sum((record.image is None for record in fetched)), 'image_prefetch_attempts': sum((record.attempts for record in fetched)), 'image_prefetch_fallback_successes': sum((record.image is not None and record.urls_tried > 1 for record in fetched))})

    def _validate_ranked_image_candidates(self, candidates: list[dict[str, Any]], *, limit: int) -> tuple[list[dict[str, Any]], list[Image.Image], dict[str, Any]]:
        pass
        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            urls = self._candidate_image_urls(candidate)
            if not urls or urls[0] in seen:
                continue
            seen.add(urls[0])
            normalized = dict(candidate)
            normalized['image_url'] = urls[0]
            normalized['_fallback_image_urls'] = urls[1:]
            deduplicated.append(normalized)
        selected: list[dict[str, Any]] = []
        images: list[Image.Image] = []
        stats: dict[str, Any] = {'image_prefetch_ms': 0.0, 'image_prefetch_urls': 0, 'image_prefetch_successes': 0, 'image_prefetch_failures': 0, 'image_prefetch_attempts': 0, 'image_prefetch_fallback_successes': 0}
        cursor = 0
        while cursor < len(deduplicated) and len(selected) < int(limit):
            needed = int(limit) - len(selected)
            batch = deduplicated[cursor:cursor + needed]
            cursor += len(batch)
            fetched, batch_stats = self._prefetch_image_candidates(batch, attempts=int(self.config['image_candidate_download_attempts']))
            for key in stats:
                stats[key] += batch_stats[key]
            for candidate, record in zip(batch, fetched):
                if len(selected) >= int(limit):
                    break
                if record.image is None or not record.resolved_url:
                    continue
                resolved = dict(candidate)
                resolved['image_url'] = record.resolved_url
                resolved.pop('_fallback_image_urls', None)
                selected.append(resolved)
                images.append(record.image)
        stats['image_prefetch_ms'] = round(float(stats['image_prefetch_ms']), 3)
        stats['ranked_candidate_pool'] = len(deduplicated)
        stats['ranked_candidates_considered'] = cursor
        stats['ranked_candidates_returned'] = len(selected)
        return (selected, images, stats)

    def text_search(self, parameters: dict[str, Any]) -> ToolExecution:
        queries = _string_list(parameters.get('query'), limit=int(self.config['text_search_max_queries']))
        if not queries:
            return ToolExecution('Error: query must be a non-empty string or list.')

        def work(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
            worker_started = time.perf_counter()
            backend_started = time.perf_counter()
            try:
                results = self.client.search(query, num=int(self.config['text_search_top_k']))
                result = {'query': query, 'results': results, 'error': None}
            except Exception as exc:
                result = {'query': query, 'results': [], 'error': f'{type(exc).__name__}: {exc}'}
            backend_ms = _elapsed_ms(backend_started)
            return (result, {'search_backend_ms': backend_ms, 'worker_wall_ms': _elapsed_ms(worker_started)})
        parallel_started = time.perf_counter()
        futures = [self._text_search_executor.submit(work, query) for query in queries]
        completed = [future.result() for future in futures]
        parallel_wall_ms = _elapsed_ms(parallel_started)
        batches = [item[0] for item in completed]
        operation_timings = [item[1] for item in completed]
        render_started = time.perf_counter()
        rendered = []
        snippet_chars = int(self.config['text_search_snippet_chars'])
        for batch in batches:
            if batch['error']:
                rendered.append({'query': batch['query'], 'error': batch['error']})
                continue
            records = []
            for item in batch['results'][:int(self.config['text_search_top_k'])]:
                records.append({'title': str(item.get('title') or 'Untitled'), 'url': _result_source_url(item), 'snippet': str(item.get('snippet') or '')[:snippet_chars]})
            rendered.append({'query': batch['query'], 'results': records})
        render_ms = _elapsed_ms(render_started)
        return ToolExecution(json.dumps(rendered, ensure_ascii=False, indent=2), metadata={'n_queries': len(queries), 'pixels_returned': 0, 'timing': {'parallel_wall_ms': parallel_wall_ms, 'shared_query_pool_workers': self.text_search_query_workers, 'render_ms': render_ms, **_timing_values(operation_timings, 'search_backend_ms'), 'operations': operation_timings}})

    def web_visit(self, parameters: dict[str, Any]) -> ToolExecution:
        urls = []
        for raw_url in _string_list(parameters.get('url'), limit=int(self.config['web_visit_max_pages'])):
            url = _http_url(raw_url)
            if url:
                urls.append(url)
        goal = str(parameters.get('goal') or '').strip()
        if not urls or not goal:
            return ToolExecution('Error: url must contain complete http(s) URL(s) and goal must be non-empty.')

        def work(url: str) -> tuple[dict[str, Any], dict[str, Any]]:
            worker_started = time.perf_counter()
            fetch_started = time.perf_counter()
            document = self.client.fetch_page_document(url, max_chars=int(self.config['web_visit_page_chars']), max_images=int(self.config['web_visit_max_image_refs']))
            fetch_page_ms = _elapsed_ms(fetch_started)
            page_text = str(document.get('text') or '')
            refs = list(document.get('images') or [])
            if not page_text:
                return ({'url': url, 'error': 'page empty or blocked', 'image_refs': refs}, {'fetch_page_ms': fetch_page_ms, 'mei_total_ms': 0.0, 'mei_model_ms': 0.0, 'worker_wall_ms': _elapsed_ms(worker_started)})
            mei_result = None
            mei_error = None
            mei_timing = None
            mei = None
            try:
                mei = self._mei_instance()
                if mei is not None:
                    mei_result = mei.summarize_web(url=url, goal=goal, page_text=page_text, image_refs=refs)
                    mei_timing = mei.take_last_timing()
            except (FatalLLMGatewayError, RetryExhaustedLLMGatewayError):
                raise
            except Exception as exc:
                mei_error = f'{type(exc).__name__}: {exc}'
                if mei is not None:
                    mei_timing = mei.take_last_timing()
            if mei_result is None:
                mei_result = {'rational': 'mei VLM unavailable; raw page excerpt returned', 'evidence': page_text, 'summary': page_text[:2000], 'relevant_images': refs}
            result = {'url': url, 'analysis': mei_result, 'image_refs': refs}
            if mei_error:
                result['mei_error'] = mei_error
            mei_timing = mei_timing or {}
            return (result, {'fetch_page_ms': fetch_page_ms, 'mei_total_ms': float(mei_timing.get('total_ms') or 0.0), 'mei_model_ms': float(mei_timing.get('mei_model_ms') or 0.0), 'mei_timing': mei_timing, 'worker_wall_ms': _elapsed_ms(worker_started)})
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(urls)) as executor:
            from neos.workflow.coevolution import parallel_map
            completed = parallel_map(executor, work, urls)
        parallel_wall_ms = _elapsed_ms(parallel_started)
        results = [item[0] for item in completed]
        operation_timings = [item[1] for item in completed]
        return ToolExecution(json.dumps(results, ensure_ascii=False, indent=2), metadata={'n_pages': len(urls), 'n_fetched': sum(('error' not in result for result in results)), 'pixels_returned': 0, 'timing': {'parallel_wall_ms': parallel_wall_ms, **_timing_values(operation_timings, 'fetch_page_ms'), **_timing_values(operation_timings, 'mei_total_ms'), **_timing_values(operation_timings, 'mei_model_ms'), 'operations': operation_timings}})

    def image_search(self, parameters: dict[str, Any]) -> ToolExecution:
        queries = _string_list(parameters.get('query'), limit=int(self.config['image_search_max_queries']))
        goal = str(parameters.get('goal') or '').strip()
        if not queries or not goal:
            return ToolExecution('Error: query and goal must both be non-empty.')

        def work(query: str) -> tuple[dict[str, Any], dict[str, Any]]:
            worker_started = time.perf_counter()
            search_started = time.perf_counter()
            top_k = int(self.config['image_search_top_k'])
            pool_size = max(10, top_k * int(self.config['image_candidate_pool_multiplier']))
            try:
                raw_results = self.client.image_search(query, num=pool_size)
            except Exception as exc:
                return ({'query': query, 'error': f'{type(exc).__name__}: {exc}'}, {'image_search_backend_ms': _elapsed_ms(search_started), 'candidate_fetch_ms': 0.0, 'mei_total_ms': 0.0, 'mei_model_ms': 0.0, 'worker_wall_ms': _elapsed_ms(worker_started)})
            image_search_backend_ms = _elapsed_ms(search_started)
            ranked_candidates = []
            for raw in raw_results:
                image_urls = _result_image_urls(raw)
                if not image_urls:
                    continue
                ranked_candidates.append({'title': str(raw.get('title') or 'Untitled'), 'source': str(raw.get('source') or ''), 'source_url': _result_source_url(raw), 'image_url': image_urls[0], '_fallback_image_urls': image_urls[1:]})
                if len(ranked_candidates) >= pool_size:
                    break
            candidates, temporary_images, candidate_prefetch = self._validate_ranked_image_candidates(ranked_candidates, limit=top_k)
            mei_result = None
            mei_error = None
            mei_timing = None
            mei = None
            candidate_fetch_ms = float(candidate_prefetch.get('image_prefetch_ms') or 0.0)
            try:
                mei = self._mei_instance()
                if mei is not None and candidates:
                    mei_result = mei.summarize_image_candidates(query=query, goal=goal, candidates=candidates, images=temporary_images)
                    mei_timing = mei.take_last_timing()
            except (FatalLLMGatewayError, RetryExhaustedLLMGatewayError):
                raise
            except Exception as exc:
                mei_error = f'{type(exc).__name__}: {exc}'
                if mei is not None:
                    mei_timing = mei.take_last_timing()
            result = {'query': query, 'goal': goal, 'candidates': candidates, 'visual_ranking': mei_result or {'rational': 'mei VLM unavailable; candidates are metadata-only'}}
            if mei_error:
                result['mei_error'] = mei_error
            mei_timing = mei_timing or {}
            return (result, {'image_search_backend_ms': image_search_backend_ms, 'candidate_fetch_ms': candidate_fetch_ms, 'mei_total_ms': float(mei_timing.get('total_ms') or 0.0), 'mei_model_ms': float(mei_timing.get('mei_model_ms') or 0.0), 'mei_timing': mei_timing, **candidate_prefetch, 'worker_wall_ms': _elapsed_ms(worker_started)})
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(queries)) as executor:
            from neos.workflow.coevolution import parallel_map
            completed = parallel_map(executor, work, queries)
        parallel_wall_ms = _elapsed_ms(parallel_started)
        results = [item[0] for item in completed]
        operation_timings = [item[1] for item in completed]
        return ToolExecution(json.dumps(results, ensure_ascii=False, indent=2), metadata={'n_queries': len(queries), 'n_candidates': sum((len(item.get('candidates') or []) for item in results)), 'pixels_returned': 0, 'timing': {'parallel_wall_ms': parallel_wall_ms, **_timing_values(operation_timings, 'image_search_backend_ms'), **_timing_values(operation_timings, 'candidate_fetch_ms'), **_timing_values(operation_timings, 'mei_total_ms'), **_timing_values(operation_timings, 'mei_model_ms'), 'image_prefetch_failures': sum((int(item.get('image_prefetch_failures') or 0) for item in operation_timings)), 'image_prefetch_fallback_successes': sum((int(item.get('image_prefetch_fallback_successes') or 0) for item in operation_timings)), 'ranked_candidates_considered': sum((int(item.get('ranked_candidates_considered') or 0) for item in operation_timings)), 'ranked_candidates_returned': sum((int(item.get('ranked_candidates_returned') or 0) for item in operation_timings)), 'operations': operation_timings}})

    def reverse_image_search(self, parameters: dict[str, Any], registry: list[Image.Image], source_urls: list[str | None] | None=None) -> ToolExecution:
        goal = str(parameters.get('goal') or '').strip()
        token_values = _string_list(parameters.get('image_token'), limit=int(self.config['reverse_max_images']), allow_none=True)
        url_values = _string_list(parameters.get('image_url'), limit=int(self.config['reverse_max_images']), allow_none=True)
        requests: list[tuple[str, str]] = [('token', token) for token in token_values] + [('url', url) for url in url_values]
        requests = requests[:int(self.config['reverse_max_images'])]
        if not goal or not requests:
            return ToolExecution('Error: goal and at least one image_token or image_url are required.')
        source_urls = list(source_urls or [])

        def token_source_url(token: str) -> str | None:
            text = token[4:] if token.startswith('img_') else token
            if text.isdigit() and 0 <= int(text) - 1 < len(source_urls):
                return source_urls[int(text) - 1]
            return None

        def work(request: tuple[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
            worker_started = time.perf_counter()
            kind, value = request
            resolve_started = time.perf_counter()
            if kind == 'token':
                query_image = resolve_image_ref(value, registry)
                public_source = token_source_url(value)
            else:
                public_source = _http_url(value)
                query_image = self.client.fetch_image(public_source) if public_source else None
            input_resolve_ms = _elapsed_ms(resolve_started)
            if query_image is None:
                return ({'input': value, 'error': 'image could not be resolved or decoded'}, {'input_resolve_ms': input_resolve_ms, 'reverse_search_backend_ms': 0.0, 'candidate_fetch_ms': 0.0, 'mei_total_ms': 0.0, 'mei_model_ms': 0.0, 'worker_wall_ms': _elapsed_ms(worker_started)})
            reverse_started = time.perf_counter()
            top_k = int(self.config['reverse_top_k'])
            pool_size = max(10, top_k * int(self.config['image_candidate_pool_multiplier']))
            try:
                raw_matches = self.client.reverse_image_search(_jpeg_bytes(query_image), top_k=pool_size, source_url=public_source)
            except Exception as exc:
                return ({'input': value, 'error': f'{type(exc).__name__}: {exc}'}, {'input_resolve_ms': input_resolve_ms, 'reverse_search_backend_ms': _elapsed_ms(reverse_started), 'candidate_fetch_ms': 0.0, 'mei_total_ms': 0.0, 'mei_model_ms': 0.0, 'worker_wall_ms': _elapsed_ms(worker_started)})
            reverse_search_backend_ms = _elapsed_ms(reverse_started)
            ranked_candidates = []
            for raw in raw_matches[:pool_size]:
                image_urls = _result_image_urls(raw)
                if not image_urls:
                    continue
                ranked_candidates.append({'title': str(raw.get('title') or raw.get('name') or 'Untitled'), 'snippet': str(raw.get('snippet') or raw.get('description') or '')[:500], 'source_url': _result_source_url(raw), 'image_url': image_urls[0], '_fallback_image_urls': image_urls[1:]})
            candidates, candidate_images, candidate_prefetch = self._validate_ranked_image_candidates(ranked_candidates, limit=top_k)
            mei_result = None
            mei_error = None
            mei_timing = None
            mei = None
            candidate_fetch_ms = float(candidate_prefetch.get('image_prefetch_ms') or 0.0)
            try:
                mei = self._mei_instance()
                if mei is not None and candidates:
                    mei_result = mei.summarize_reverse_candidates(goal=goal, query_image=query_image, candidates=candidates, candidate_images=candidate_images)
                    mei_timing = mei.take_last_timing()
            except (FatalLLMGatewayError, RetryExhaustedLLMGatewayError):
                raise
            except Exception as exc:
                mei_error = f'{type(exc).__name__}: {exc}'
                if mei is not None:
                    mei_timing = mei.take_last_timing()
            result = {'input': value, 'goal': goal, 'candidates': candidates, 'visual_comparison': mei_result or {'rational': 'mei VLM unavailable; matches are metadata-only'}}
            if mei_error:
                result['mei_error'] = mei_error
            mei_timing = mei_timing or {}
            return (result, {'input_resolve_ms': input_resolve_ms, 'reverse_search_backend_ms': reverse_search_backend_ms, 'candidate_fetch_ms': candidate_fetch_ms, 'mei_total_ms': float(mei_timing.get('total_ms') or 0.0), 'mei_model_ms': float(mei_timing.get('mei_model_ms') or 0.0), 'mei_timing': mei_timing, **candidate_prefetch, 'worker_wall_ms': _elapsed_ms(worker_started)})
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            from neos.workflow.coevolution import parallel_map
            completed = parallel_map(executor, work, requests)
        parallel_wall_ms = _elapsed_ms(parallel_started)
        results = [item[0] for item in completed]
        operation_timings = [item[1] for item in completed]
        return ToolExecution(json.dumps(results, ensure_ascii=False, indent=2), metadata={'n_inputs': len(requests), 'n_successful': sum(('error' not in item for item in results)), 'pixels_returned': 0, 'engine': 'google_lens', 'timing': {'parallel_wall_ms': parallel_wall_ms, **_timing_values(operation_timings, 'input_resolve_ms'), **_timing_values(operation_timings, 'reverse_search_backend_ms'), **_timing_values(operation_timings, 'candidate_fetch_ms'), **_timing_values(operation_timings, 'mei_total_ms'), **_timing_values(operation_timings, 'mei_model_ms'), 'image_prefetch_failures': sum((int(item.get('image_prefetch_failures') or 0) for item in operation_timings)), 'image_prefetch_fallback_successes': sum((int(item.get('image_prefetch_fallback_successes') or 0) for item in operation_timings)), 'ranked_candidates_considered': sum((int(item.get('ranked_candidates_considered') or 0) for item in operation_timings)), 'ranked_candidates_returned': sum((int(item.get('ranked_candidates_returned') or 0) for item in operation_timings)), 'operations': operation_timings}})

    def fetch_image(self, parameters: dict[str, Any], registry_size: int=0) -> ToolExecution:
        raw = parameters.get('urls')
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.startswith('[') and stripped.endswith(']') or (stripped.startswith('{') and stripped.endswith('}')):
                raise ValueError('protocol normalization missed a complete JSON-container-looking string')
            items = [raw]
        elif isinstance(raw, list):
            items = raw
        else:
            return ToolExecution('Error: urls must be a URL string, a list of URL strings, or a list of {url, caption} objects.')
        requests: list[_FetchRequest] = []
        for item in items:
            if isinstance(item, dict):
                if not isinstance(item.get('url'), str) or ('caption' in item and (not isinstance(item.get('caption'), str))):
                    raise ValueError('FetchImage object items require string url/caption fields')
                url = _http_url(item.get('url'))
                caption = str(item.get('caption') or '').strip()
            elif isinstance(item, str):
                url = _http_url(item)
                caption = ''
            else:
                raise ValueError('FetchImage urls items must be strings or {url, caption} objects')
            if url:
                requests.append(_FetchRequest(len(requests), url, caption))
            if len(requests) >= int(self.config['fetch_max_images']):
                break
        if not requests:
            return ToolExecution('Error: urls must contain at least one http(s) image URL.')

        def work(request: _FetchRequest):
            worker_started = time.perf_counter()
            fetch_started = time.perf_counter()
            image = self.client.fetch_image(request.url)
            fetch_ms = _elapsed_ms(fetch_started)
            resize_started = time.perf_counter()
            if image is not None:
                image = downscale(image, int(self.config['fetch_max_side']))
            resize_ms = _elapsed_ms(resize_started)
            return (request, image, {'fetch_image_ms': fetch_ms, 'resize_ms': resize_ms, 'worker_wall_ms': _elapsed_ms(worker_started)})
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            completed = list(executor.map(work, requests))
        parallel_wall_ms = _elapsed_ms(parallel_started)
        images: list[Image.Image] = []
        records = []
        output_urls = []
        operation_timings = []
        for request, image, operation_timing in completed:
            operation_timings.append(operation_timing)
            if image is None:
                records.append({'url': request.url, 'caption': request.caption, 'status': 'failed'})
                continue
            images.append(image)
            output_urls.append(request.url)
            records.append({'url': request.url, 'caption': request.caption, 'status': 'loaded', 'size': [image.width, image.height]})
        refs = output_refs(registry_size, len(images))
        return ToolExecution(render_visual_result('FetchImage', records, refs), images=images, metadata={'n_requested': len(requests), 'n_images': len(images), 'image_source_urls': output_urls, 'image_kinds': ['fetch'] * len(images), 'output_refs': refs, 'visual_result_schema_version': VISUAL_RESULT_SCHEMA_VERSION, 'parallel': True, 'timing': {'parallel_wall_ms': parallel_wall_ms, **_timing_values(operation_timings, 'fetch_image_ms'), **_timing_values(operation_timings, 'resize_ms'), 'operations': operation_timings}})

    @staticmethod
    def _box(value: Any) -> tuple[float, float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            box = tuple((float(part) for part in value))
        except (TypeError, ValueError):
            return None
        left, top, right, bottom = box
        if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
            return None
        return box

    def crop_image(self, parameters: dict[str, Any], registry: list[Image.Image]) -> ToolExecution:
        config = parameters.get('crop_config')
        if not isinstance(config, dict) or not config:
            return ToolExecution('Error: crop_config must be a non-empty object.')
        requests: list[_CropRequest] = []
        errors: list[str] = []
        for token, raw_boxes in config.items():
            single = self._box(raw_boxes)
            boxes = [single] if single is not None else raw_boxes
            if not isinstance(boxes, (list, tuple)):
                errors.append(f'{token}: boxes must be one box or a list of boxes')
                continue
            per_image = 0
            for raw_box in boxes:
                box = self._box(raw_box)
                if box is None:
                    errors.append(f'{token}: invalid normalized box {raw_box!r}')
                    continue
                if per_image >= int(self.config['crop_max_per_image']):
                    errors.append(f'{token}: boxes beyond the per-image limit were ignored')
                    break
                requests.append(_CropRequest(len(requests), str(token), box))
                per_image += 1
                if len(requests) >= int(self.config['crop_max_total']):
                    break
            if len(requests) >= int(self.config['crop_max_total']):
                break
        if not requests:
            return ToolExecution('Error: no valid crop boxes. ' + '; '.join(errors))

        def work(request: _CropRequest):
            worker_started = time.perf_counter()
            image = resolve_image_ref(request.image_token, registry)
            if image is None:
                return (request, None, None, {'crop_ms': _elapsed_ms(worker_started), 'worker_wall_ms': _elapsed_ms(worker_started)})
            left, top, right, bottom = request.box
            pixel_box = (max(0, min(image.width - 1, int(left * image.width / 1000))), max(0, min(image.height - 1, int(top * image.height / 1000))), max(1, min(image.width, int(right * image.width / 1000 + 0.999999))), max(1, min(image.height, int(bottom * image.height / 1000 + 0.999999))))
            crop = image.convert('RGB').crop(pixel_box)
            return (request, crop, pixel_box, {'crop_ms': _elapsed_ms(worker_started), 'worker_wall_ms': _elapsed_ms(worker_started)})
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            completed = list(executor.map(work, requests))
        parallel_wall_ms = _elapsed_ms(parallel_started)
        images = []
        records = []
        operation_timings = []
        for request, crop, pixel_box, operation_timing in completed:
            operation_timings.append(operation_timing)
            if crop is None:
                records.append({'image_token': request.image_token, 'box': request.box, 'status': 'unresolved'})
                continue
            images.append(crop)
            records.append({'image_token': request.image_token, 'box': list(request.box), 'pixel_box': list(pixel_box), 'status': 'cropped', 'size': [crop.width, crop.height]})
        if errors:
            records.append({'warnings': errors})
        refs = output_refs(len(registry), len(images))
        return ToolExecution(render_visual_result('CropImage', records, refs), images=images, metadata={'n_requested': len(requests), 'n_images': len(images), 'image_source_urls': [None] * len(images), 'image_kinds': ['crop'] * len(images), 'output_refs': refs, 'visual_result_schema_version': VISUAL_RESULT_SCHEMA_VERSION, 'parallel': True, 'timing': {'parallel_wall_ms': parallel_wall_ms, **_timing_values(operation_timings, 'crop_ms'), 'operations': operation_timings}})

    def execute(self, name: str, parameters: dict[str, Any], registry: list[Image.Image] | None=None, source_urls: list[str | None] | None=None) -> ToolExecution:
        registry = list(registry or [])
        if name not in NEOS_TOOL_NAMES:
            return ToolExecution(f'Error: unknown tool {name!r}.')
        schema_version = str(self.config['tool_schema_version'])
        argument_error = validate_tool_arguments(name, parameters, schema_version=schema_version)
        if argument_error is not None:
            return ToolExecution(f'Error: invalid arguments for {name}: {argument_error}', metadata={'schema_validation_error': True, 'runtime_type_validation_success': False, 'tool_schema_version': schema_version, 'pixels_returned': 0})
        execution = None
        if name == 'TextSearch':
            execution = self.text_search(parameters)
        elif name == 'WebVisit':
            execution = self.web_visit(parameters)
        elif name == 'ImageSearch':
            execution = self.image_search(parameters)
        elif name == 'ReverseImageSearch':
            execution = self.reverse_image_search(parameters, registry, source_urls)
        elif name == 'FetchImage':
            execution = self.fetch_image(parameters, len(registry))
        elif name == 'CropImage':
            execution = self.crop_image(parameters, registry)
        if execution is None:
            raise AssertionError(f'unhandled tool {name!r}')
        if name in {'FetchImage', 'CropImage'} and self.config['reuse_loaded_images']:
            from neos.workflow.visual_register import reuse_loaded_images
            execution = reuse_loaded_images(name, execution, registry, self.config)
        execution.metadata = dict(execution.metadata or {})
        execution.metadata.update({'runtime_type_validation_success': True, 'runtime_argument_types': _type_signature(parameters), 'tool_schema_version': schema_version})
        return execution
