from __future__ import annotations
import base64
import io
import json
import os
import re
import threading
import time
from collections.abc import Sequence
from typing import Any, Callable
from PIL import Image
from neos.workflow.mei_trace import get_mei_trace_store
WEB_MEI_PROMPT_VERSION = 'neos-web-mei-paper-exact'
IMAGE_MEI_PROMPT_VERSION = 'neos-image-mei-derived'
REVERSE_MEI_PROMPT_VERSION = 'neos-reverse-mei-derived'
WEB_MEI_SYSTEM_PROMPT = 'You are a precise multimodal-aware web content analyst. Return a valid JSON object only, with no Markdown or commentary. Never include navigation menus, sidebars, cookie banners, footers, or advertisements.'
WEB_MEI_USER_PROMPT = 'Given the webpage content, the user goal, and the list of images found on the page: (1) locate the sections, tables, names, dates, numbers, and claims directly related to the goal, and explain briefly why the page is relevant; (2) extract the strongest supporting content from the article body while preserving qualifications and source claims; (3) summarize how that evidence answers or contributes to the goal without introducing unsupported facts; and (4) select only page images that may help verify or answer the goal, retaining their alt, url, and a short caption. If the page contains no direct evidence, return empty evidence and an empty image list. Output exactly the keys rational, evidence, summary, and relevant_images, with no additional keys.\n\nUser goal:\n{goal}\n\nWebpage content:\n{page_text}\n\nPage image references:\n{image_refs}\n'
IMAGE_MEI_SYSTEM_PROMPT = 'You are a precise visual-search candidate analyst. Inspect pixels, not only titles or captions. Return one valid JSON object and no Markdown.'
IMAGE_MEI_USER_PROMPT = 'Rank the candidate images for the stated visual goal. For each candidate, report a relevance level (high, medium, low, or none) and a concise pixel-grounded caption. Metadata can guide inspection but is not visual proof.\n\nReturn exactly:\n{{"rational": string, "candidates": [{{"index": integer, "relevance": "high|medium|low|none", "caption": string}}]}}\n\nSearch query: {query}\nVisual goal: {goal}\nCandidate metadata:\n{candidates}\n'
REVERSE_MEI_SYSTEM_PROMPT = 'You are a precise reverse-image comparison analyst. Compare visible pixels and return one valid JSON object with no Markdown.'
REVERSE_MEI_USER_PROMPT = 'The first image is the query observation. Remaining images are reverse-search candidates in metadata order. Compare their pixels for the stated goal. Use match_state visual_match, partial_match, not_match, or insufficient_evidence.\n\nReturn exactly:\n{{"rational": string, "candidates": [{{"index": integer, "match_state": "visual_match|partial_match|not_match|insufficient_evidence", "caption": string}}]}}\n\nGoal: {goal}\nCandidate metadata:\n{candidates}\n'

def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def _extract_json(text: str) -> dict[str, Any] | None:
    value = str(text or '').strip()
    if value.startswith('```'):
        value = re.sub('\\A```(?:json)?\\s*|\\s*```\\Z', '', value, flags=re.I)
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search('\\{.*\\}', value, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

def _image_data_uri(image: Image.Image, max_side: int=1024) -> str:
    rgb = image.convert('RGB')
    if max(rgb.size) > max_side:
        scale = max_side / max(rgb.size)
        rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format='JPEG', quality=88, optimize=False)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()

class NeosMEI:
    pass

    def __init__(self, gateway=None, *, model: str | None=None, fallback_models: Sequence[str] | str | None=None):
        if gateway is None:
            from neos.eval.gateway import LLMGateway
            from neos.eval.secrets import DASHSCOPE_MEI_FALLBACK_MODELS
            selected_model = model or os.getenv('NEOS_MEI_MODEL_SPEC', 'qwen3.7-flash')
            selected_fallbacks = DASHSCOPE_MEI_FALLBACK_MODELS if fallback_models is None else fallback_models
            gateway = LLMGateway(model=selected_model, fallback_models=selected_fallbacks)
        self.gateway = gateway
        self.model = str(model or getattr(gateway, 'model', None) or type(gateway).__qualname__)
        if fallback_models is None:
            fallback_models = getattr(gateway, 'fallback_models', ())
        if isinstance(fallback_models, str):
            fallback_models = fallback_models.split(',')
        seen = {self.model}
        normalized_fallbacks = []
        for value in fallback_models or ():
            candidate = str(value).strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            normalized_fallbacks.append(candidate)
        self.fallback_models = tuple(normalized_fallbacks)
        self.max_completion_tokens = int(os.getenv('NEOS_MEI_MAX_COMPLETION_TOKENS', '4096'))
        if self.max_completion_tokens <= 0:
            raise ValueError('NEOS_MEI_MAX_COMPLETION_TOKENS must be positive')
        self.max_concurrency = int(os.getenv('NEOS_MEI_MAX_CONCURRENCY', '256'))
        if self.max_concurrency < 0:
            raise ValueError('NEOS_MEI_MAX_CONCURRENCY must be zero (unbounded) or positive')
        self._gateway_semaphore = threading.BoundedSemaphore(self.max_concurrency) if self.max_concurrency > 0 else None
        concurrency_env = {'neos_web_mei': 'NEOS_WEB_MEI_MAX_CONCURRENCY', 'neos_image_mei': 'NEOS_IMAGE_MEI_MAX_CONCURRENCY', 'neos_reverse_mei': 'NEOS_REVERSE_MEI_MAX_CONCURRENCY'}
        self.mei_concurrency_by_namespace: dict[str, int] = {}
        self._gateway_semaphores: dict[str, threading.BoundedSemaphore | None] = {}
        for namespace, env_name in concurrency_env.items():
            raw_value = os.getenv(env_name)
            if raw_value is None:
                concurrency = self.max_concurrency
                semaphore = self._gateway_semaphore
            else:
                concurrency = int(raw_value)
                if concurrency < 0:
                    raise ValueError(f'{env_name} must be zero (unbounded) or positive')
                semaphore = threading.BoundedSemaphore(concurrency) if concurrency > 0 else None
            self.mei_concurrency_by_namespace[namespace] = concurrency
            self._gateway_semaphores[namespace] = semaphore
        self._timing_local = threading.local()
        self.trace_store = get_mei_trace_store()

    def _capture_trace(self, *, namespace: str, prompt_version: str, messages: list[dict[str, Any]], max_completion_tokens: int, raw_output: str, parsed_output: dict[str, Any] | None, descriptor: dict[str, Any]) -> None:
        if self.trace_store is None:
            return
        timing = getattr(self._timing_local, 'last', None)
        if not isinstance(timing, dict):
            timing = {}
            self._timing_local.last = timing
        try:
            captured = self.trace_store.record(namespace=namespace, prompt_version=prompt_version, model=self.model, fallback_models=self.fallback_models, messages=messages, max_completion_tokens=max_completion_tokens, raw_output=raw_output, parsed_output=parsed_output, timing=timing, descriptor=descriptor)
        except Exception as exc:
            timing['mei_trace_capture_error'] = f'{type(exc).__name__}: {exc}'
            return
        timing['mei_trace_id'] = captured['trace_id']
        timing['mei_example_id'] = captured['example_id']
        timing['mei_trace_asset_count'] = captured['asset_count']

    def take_last_timing(self) -> dict[str, Any] | None:
        pass
        value = getattr(self._timing_local, 'last', None)
        self._timing_local.last = None
        return dict(value) if isinstance(value, dict) else None

    def _invoke(self, namespace: str, compute: Callable[[], str]) -> str:
        started = time.perf_counter()
        compute_ms = 0.0
        admission_wait_ms = 0.0
        compute_called = False
        gateway_call: dict[str, Any] | None = None
        mei_concurrency = self.mei_concurrency_by_namespace.get(namespace, self.max_concurrency)
        gateway_semaphore = self._gateway_semaphores.get(namespace, self._gateway_semaphore)

        def timed_compute() -> str:
            nonlocal compute_ms, compute_called, gateway_call
            nonlocal admission_wait_ms
            compute_called = True
            admission_started = time.perf_counter()
            if gateway_semaphore is not None:
                gateway_semaphore.acquire()
            admission_wait_ms += (time.perf_counter() - admission_started) * 1000
            compute_started = time.perf_counter()
            try:
                return compute()
            finally:
                compute_ms += (time.perf_counter() - compute_started) * 1000
                if gateway_semaphore is not None:
                    gateway_semaphore.release()
                take_last_call = getattr(self.gateway, 'take_last_call', None)
                if callable(take_last_call):
                    value = take_last_call()
                    gateway_call = dict(value) if isinstance(value, dict) else None
        try:
            value = timed_compute()
        except Exception as exc:
            total_ms = (time.perf_counter() - started) * 1000
            self._timing_local.last = {'namespace': namespace, 'model': self.model, 'fallback_models': list(self.fallback_models), 'compute_called': compute_called, 'mei_model_ms': round(compute_ms, 3), 'mei_admission_wait_ms': round(admission_wait_ms, 3), 'mei_max_concurrency': mei_concurrency, 'total_ms': round(total_ms, 3), 'error': f'{type(exc).__name__}: {exc}'}
            if gateway_call is not None:
                self._timing_local.last['gateway_call'] = gateway_call
                self._timing_local.last['actual_model'] = gateway_call.get('actual_model')
                self._timing_local.last['fallback_used'] = gateway_call.get('fallback_used')
            raise
        total_ms = (time.perf_counter() - started) * 1000
        self._timing_local.last = {'namespace': namespace, 'model': self.model, 'fallback_models': list(self.fallback_models), 'compute_called': compute_called, 'mei_model_ms': round(compute_ms, 3), 'mei_admission_wait_ms': round(admission_wait_ms, 3), 'mei_max_concurrency': mei_concurrency, 'total_ms': round(total_ms, 3)}
        if gateway_call is not None:
            self._timing_local.last['gateway_call'] = gateway_call
            self._timing_local.last['actual_model'] = gateway_call.get('actual_model')
            self._timing_local.last['fallback_used'] = gateway_call.get('fallback_used')
        return value

    def summarize_web(self, *, url: str, goal: str, page_text: str, image_refs: list[dict[str, str]]) -> dict[str, Any]:
        prompt = WEB_MEI_USER_PROMPT.format(goal=goal, page_text=page_text, image_refs=_compact_json(image_refs))
        messages = [{'role': 'system', 'content': WEB_MEI_SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}]

        def compute() -> str:
            return self.gateway.chat(messages, max_completion_tokens=self.max_completion_tokens)
        raw = self._invoke('neos_web_mei', compute)
        parsed = _extract_json(raw)
        self._capture_trace(namespace='neos_web_mei', prompt_version=WEB_MEI_PROMPT_VERSION, messages=messages, max_completion_tokens=self.max_completion_tokens, raw_output=raw, parsed_output=parsed, descriptor={'url': url, 'goal': goal, 'image_refs': image_refs})
        return parsed or {'raw_mei_response': raw}

    def summarize_image_candidates(self, *, query: str, goal: str, candidates: list[dict[str, Any]], images: list[Image.Image | None]) -> dict[str, Any]:
        metadata = [{'index': index, 'title': candidate.get('title') or '', 'source': candidate.get('source') or '', 'source_url': candidate.get('source_url') or '', 'image_url': candidate.get('image_url') or ''} for index, candidate in enumerate(candidates, 1)]
        content: list[dict[str, Any]] = [{'type': 'text', 'text': IMAGE_MEI_USER_PROMPT.format(query=query, goal=goal, candidates=_compact_json(metadata))}]
        valid_digests = []
        for index, image in enumerate(images, 1):
            if image is None:
                valid_digests.append(None)
                continue
            content.extend([{'type': 'text', 'text': f'Candidate image {index}:'}, {'type': 'image_url', 'image_url': {'url': _image_data_uri(image)}}])
        messages = [{'role': 'system', 'content': IMAGE_MEI_SYSTEM_PROMPT}, {'role': 'user', 'content': content}]

        def compute() -> str:
            return self.gateway.chat(messages, max_completion_tokens=self.max_completion_tokens)
        raw = self._invoke('neos_image_mei', compute)
        parsed = _extract_json(raw)
        self._capture_trace(namespace='neos_image_mei', prompt_version=IMAGE_MEI_PROMPT_VERSION, messages=messages, max_completion_tokens=self.max_completion_tokens, raw_output=raw, parsed_output=parsed, descriptor={'query': query, 'goal': goal, 'metadata': metadata})
        return parsed or {'raw_mei_response': raw}

    def summarize_reverse_candidates(self, *, goal: str, query_image: Image.Image, candidates: list[dict[str, Any]], candidate_images: list[Image.Image | None]) -> dict[str, Any]:
        metadata = [{'index': index, 'title': candidate.get('title') or '', 'source_url': candidate.get('source_url') or '', 'image_url': candidate.get('image_url') or ''} for index, candidate in enumerate(candidates, 1)]
        content: list[dict[str, Any]] = [{'type': 'text', 'text': REVERSE_MEI_USER_PROMPT.format(goal=goal, candidates=_compact_json(metadata))}, {'type': 'text', 'text': 'Query observation:'}, {'type': 'image_url', 'image_url': {'url': _image_data_uri(query_image)}}]
        digests: list[str | None] = []
        for index, image in enumerate(candidate_images, 1):
            if image is None:
                digests.append(None)
                continue
            content.extend([{'type': 'text', 'text': f'Candidate image {index}:'}, {'type': 'image_url', 'image_url': {'url': _image_data_uri(image)}}])
        messages = [{'role': 'system', 'content': REVERSE_MEI_SYSTEM_PROMPT}, {'role': 'user', 'content': content}]

        def compute() -> str:
            return self.gateway.chat(messages, max_completion_tokens=self.max_completion_tokens)
        raw = self._invoke('neos_reverse_mei', compute)
        parsed = _extract_json(raw)
        self._capture_trace(namespace='neos_reverse_mei', prompt_version=REVERSE_MEI_PROMPT_VERSION, messages=messages, max_completion_tokens=self.max_completion_tokens, raw_output=raw, parsed_output=parsed, descriptor={'goal': goal, 'metadata': metadata})
        return parsed or {'raw_mei_response': raw}
