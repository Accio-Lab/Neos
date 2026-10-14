from __future__ import annotations
import itertools
import json
import logging
import math
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from neos.eval.models import AzureModel, ModelContextLengthExceededError, OpenAICompatModel, context_length_overflow_counts, reduced_max_tokens_for_context_error
logger = logging.getLogger(__name__)
NATIVE_MESSAGE_PROTOCOL = 'openai-chat-completions-native-tools'

class NeosModelRequestError(RuntimeError):
    pass

    def __init__(self, *, backend: str, cause: Exception, attempts: int, retryable: bool):
        self.backend = backend
        self.cause = cause
        self.attempts = attempts
        self.retryable = retryable
        super().__init__(f'model request failed backend={backend} attempts={attempts} retryable={retryable}: {type(cause).__name__}: {cause}')

def official_chat_template(model_path: str | Path) -> str:
    pass
    root = Path(model_path).expanduser().resolve()
    tokenizer_config = root / 'tokenizer_config.json'
    if not tokenizer_config.is_file():
        raise FileNotFoundError(f'missing tokenizer_config.json under {root}')
    payload = json.loads(tokenizer_config.read_text(encoding='utf-8'))
    template = payload.get('chat_template')
    if isinstance(template, str) and template:
        return template
    template_file = root / 'chat_template.jinja'
    if template_file.is_file():
        return template_file.read_text(encoding='utf-8')
    raise ValueError(f'model has no bundled official chat template: {root}')

def _model_family(model_path: str | Path) -> tuple[str, str]:
    config = json.loads((Path(model_path).expanduser().resolve() / 'config.json').read_text(encoding='utf-8'))
    model_type = str(config.get('model_type') or '')
    if model_type in {'qwen3_5', 'qwen3_5_moe'}:
        return ('qwen3.5', 'qwen3_coder')
    raise ValueError(f'unsupported native Neos model_type={model_type!r}')

def _expected_reasoning_parser(family: str, model_path: str | Path) -> str:
    if family != 'qwen3.5':
        raise ValueError(f'unsupported Neos model family: {family!r}')
    return 'qwen3'

def _local_sglang_build() -> tuple[str | None, str | None]:
    pass
    try:
        import sglang
        version = str(getattr(sglang, '__version__', '') or '') or None
        module_path = Path(sglang.__file__).resolve()
        source_root = module_path.parents[2]
        commit = subprocess.run(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], check=False, capture_output=True, text=True, timeout=5).stdout.strip() or None
        return (version, commit)
    except Exception:
        return (None, None)

def model_family_and_native_tool_parser(model_path: str | Path) -> tuple[str, str]:
    pass
    return _model_family(model_path)

def _assistant_message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, 'model_dump'):
        value = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        value = dict(message)
    else:
        raise TypeError(f'unsupported assistant message type: {type(message)!r}')
    value['role'] = 'assistant'
    if value.get('content') is None:
        value['content'] = ''
    return value

class NeosOpenAICompatModel(OpenAICompatModel):
    pass
    official_multimodal_message_order = True
    message_protocol = NATIVE_MESSAGE_PROTOCOL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._neos_template_lock = threading.Lock()
        self._neos_template_verified = False
        self._neos_context_length: int | None = None
        self._family: str | None = None
        self._tool_parser: str | None = None
        self._reasoning_parser: str | None = None
        self._sglang_version: str | None = None
        self._sglang_git_commit: str | None = None

    def ensure_neos_chat_template(self, expected_context_length: int | None=None) -> None:
        if self._neos_template_verified and (expected_context_length is None or self._neos_context_length == expected_context_length):
            return
        with self._neos_template_lock:
            try:
                from urllib.request import urlopen
                with urlopen(self._server_info_url(), timeout=15) as response:
                    info = json.loads(response.read().decode('utf-8'))
            except Exception as exc:
                raise RuntimeError(f'Cannot verify the native Neos server through {self._server_info_url()}.') from exc
            model_path = info.get('model_path')
            if not model_path:
                raise RuntimeError('SGLang /server_info did not expose model_path')
            family, expected_parser = _model_family(model_path)
            actual_parser = str(info.get('tool_call_parser') or '')
            if actual_parser != expected_parser:
                raise RuntimeError(f'{family} must use SGLang --tool-call-parser {expected_parser}; server reports {actual_parser!r}.')
            expected_reasoning_parser = _expected_reasoning_parser(family, model_path)
            actual_reasoning_parser = str(info.get('reasoning_parser') or '')
            if actual_reasoning_parser != expected_reasoning_parser:
                raise RuntimeError(f'{family} must use SGLang --reasoning-parser {expected_reasoning_parser}; server reports {actual_reasoning_parser!r}.')
            if expected_context_length is not None:
                actual_context = info.get('context_length')
                if actual_context != expected_context_length:
                    raise RuntimeError(f'Neos context mismatch: expected {expected_context_length}, server reports {actual_context!r}.')
                self._neos_context_length = expected_context_length
            self._family = family
            self._tool_parser = actual_parser
            self._reasoning_parser = actual_reasoning_parser
            self._sglang_version = str(info.get('version') or '') or None
            _, self._sglang_git_commit = _local_sglang_build()
            self._neos_template_verified = True

    def backend_manifest(self) -> dict[str, Any]:
        if not self._neos_template_verified:
            self.ensure_neos_chat_template()
        return {'type': 'local_openai_compatible', 'message_protocol': NATIVE_MESSAGE_PROTOCOL, 'model': self.model, 'base_url': self.base_url, 'model_family': self._family, 'tool_call_parser': self._tool_parser, 'reasoning_parser': self._reasoning_parser, 'sglang_version': self._sglang_version, 'sglang_git_commit': self._sglang_git_commit, 'native_tools': True, 'raw_completion_available': False, 'raw_completion_unavailable_reason': 'SGLang 0.5.9 /chat/completions exposes only the parser-processed assistant message; no safe same-sample raw trace hook is installed'}

    def chat(self, messages: list[dict], max_tokens: int=4096, temperature: float=0.0, top_p: float | None=None, tools: list[dict] | None=None, tool_choice: str='auto') -> dict[str, Any]:
        if tools:
            self.ensure_neos_chat_template()
        last_err: Exception | None = None
        request_max_tokens = max_tokens
        attempt_timings = []
        self._state.last_response_metadata = None
        for attempt in range(self.max_retries):
            attempt_started = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {}
                if top_p is not None:
                    kwargs['top_p'] = top_p
                if tools:
                    kwargs.update(tools=tools, tool_choice=tool_choice, parallel_tool_calls=False)
                response = self._client().chat.completions.create(model=self.model, messages=messages, max_tokens=request_max_tokens, temperature=temperature, **kwargs)
                choice = response.choices[0]
                usage = getattr(response, 'usage', None)
                if hasattr(usage, 'model_dump'):
                    usage = usage.model_dump()
                elif not isinstance(usage, dict):
                    usage = None
                attempt_timings.append({'attempt': attempt + 1, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3), 'status': 'success'})
                assistant = _assistant_message_dict(choice.message)
                self._state.last_response_metadata = {'request': {'api_surface': '/chat/completions', 'base_url': self.base_url, 'model': self.model, 'message_count': len(messages), 'tool_schema_count': len(tools or []), 'tool_choice': tool_choice if tools else None, 'parallel_tool_calls': False if tools else None, 'requested_max_tokens': max_tokens, 'effective_max_tokens': request_max_tokens}, 'response': {'request_id': str(getattr(response, 'id', '') or '') or None, 'finish_reason': choice.finish_reason, 'usage': usage or {}, 'assistant_message': assistant}, 'attempts': attempt_timings}
                return assistant
            except Exception as exc:
                last_err = exc
                attempt_timings.append({'attempt': attempt + 1, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3), 'status': 'error', 'error_type': type(exc).__name__, 'status_code': getattr(exc, 'status_code', None)})
                reduced = reduced_max_tokens_for_context_error(exc, request_max_tokens)
                if reduced is not None:
                    request_max_tokens = reduced
                    continue
                overflow = context_length_overflow_counts(exc)
                if overflow is not None:
                    input_tokens, context_tokens = overflow
                    if context_tokens - input_tokens <= 1:
                        raise ModelContextLengthExceededError(input_tokens=input_tokens, context_tokens=context_tokens, original_error=exc) from exc
                if getattr(exc, 'status_code', None) == 400:
                    break
                time.sleep(min(2 ** attempt * 5, 60))
        assert last_err is not None
        raise NeosModelRequestError(backend=self.base_url, cause=last_err, attempts=len(attempt_timings), retryable=getattr(last_err, 'status_code', None) != 400) from last_err

class NeosAzureNativeModel(AzureModel):
    pass
    message_protocol = NATIVE_MESSAGE_PROTOCOL

    def backend_manifest(self) -> dict[str, Any]:
        result = super().backend_manifest()
        result.update(message_protocol=NATIVE_MESSAGE_PROTOCOL, native_messages_preserved=True, raw_completion_available=False, raw_completion_unavailable_reason='gateway Chat Completions response exposes parsed native fields only')
        return result

    def chat(self, messages: list[dict], max_tokens: int=4096, temperature: float=0.0, top_p: float | None=None, tools: list[dict] | None=None, tool_choice: str='auto') -> dict[str, Any]:
        del temperature, top_p
        last_err: Exception | None = None
        self._state.last_response_metadata = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens}
                if tools:
                    kwargs.update(tools=tools, tool_choice=tool_choice, parallel_tool_calls=False)
                else:
                    kwargs['reasoning_effort'] = self.requested_reasoning_effort
                response = self._client().chat.completions.create(**kwargs)
                if not response.choices:
                    raise RuntimeError('Relay gateway returned no choices')
                choice = response.choices[0]
                assistant = _assistant_message_dict(choice.message)
                self._state.last_response_metadata = {'request': {'api_surface': '/chat/completions', 'model': self.model, 'message_count': len(messages), 'tool_schema_count': len(tools or []), 'tool_choice': tool_choice if tools else None, 'parallel_tool_calls': False if tools else None, 'max_completion_tokens': max_tokens}, 'response': {**response.model_dump(), 'request_id': str(getattr(response, 'id', '') or '') or None}}
                return assistant
            except Exception as exc:
                last_err = exc
                if getattr(exc, 'status_code', None) == 400:
                    break
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2 ** attempt * 5, 60))
        assert last_err is not None
        raise NeosModelRequestError(backend=f'relaychat:{self.model}', cause=last_err, attempts=attempt + 1, retryable=getattr(last_err, 'status_code', None) != 400) from last_err

class NeosRoundRobinOpenAICompatModel:
    pass
    official_multimodal_message_order = True

    def __init__(self, base_urls: list[str], model: str):
        if not base_urls:
            raise ValueError('at least one Neos endpoint is required')
        self.model = model
        self.backends = [NeosOpenAICompatModel(base_url=url, model=model) for url in base_urls]
        self._counter = itertools.count()
        self._counter_lock = threading.Lock()
        self._state = threading.local()
        self._route_lock = threading.RLock()
        self._trajectory_routes: dict[str, int] = {}
        self._trajectory_weights: dict[str, float] = {}
        self._active_trajectories = [0 for _ in self.backends]
        self._active_trajectory_weight = [0.0 for _ in self.backends]
        self._route_tiebreaker = itertools.count()

    def _assign_trajectory(self, trajectory_key: str, *, estimated_weight: float=1.0, replica_load: list[int] | tuple[int, ...] | None=None) -> int:
        with self._route_lock:
            existing = self._trajectory_routes.get(trajectory_key)
            if existing is not None:
                return existing
            try:
                weight = float(estimated_weight)
            except (TypeError, ValueError):
                weight = 1.0
            if not math.isfinite(weight) or weight <= 0:
                weight = 1.0
            loads = [max(0, int(value)) for value in replica_load] if replica_load is not None and len(replica_load) == len(self.backends) else [0 for _ in self.backends]
            scores = [(loads[index], self._active_trajectory_weight[index], self._active_trajectories[index]) for index in range(len(self.backends))]
            minimum = min(scores)
            candidates = [index for index, score in enumerate(scores) if score == minimum]
            index = candidates[next(self._route_tiebreaker) % len(candidates)]
            self._trajectory_routes[trajectory_key] = index
            self._trajectory_weights[trajectory_key] = weight
            self._active_trajectories[index] += 1
            self._active_trajectory_weight[index] += weight
            return index

    def assign_trajectory(self, trajectory_key: str, *, estimated_weight: float=1.0, replica_load: list[int] | tuple[int, ...] | None=None) -> int:
        pass
        key = str(trajectory_key)
        if not key:
            raise ValueError('trajectory_key must be non-empty')
        return self._assign_trajectory(key, estimated_weight=estimated_weight, replica_load=replica_load)

    def trajectory_replica(self, trajectory_key: str) -> int | None:
        with self._route_lock:
            return self._trajectory_routes.get(str(trajectory_key))

    @contextmanager
    def trajectory_session(self, trajectory_key: str) -> Iterator[dict[str, Any]]:
        pass
        key = str(trajectory_key)
        if not key:
            raise ValueError('trajectory_key must be non-empty')
        index = self._assign_trajectory(key)
        previous = getattr(self._state, 'trajectory_key', None)
        self._state.trajectory_key = key
        try:
            yield {'replica_index': index, 'base_url': self.backends[index].base_url}
        finally:
            if previous is None:
                try:
                    del self._state.trajectory_key
                except AttributeError:
                    pass
            else:
                self._state.trajectory_key = previous

    def release_trajectory(self, trajectory_key: str) -> None:
        pass
        key = str(trajectory_key)
        with self._route_lock:
            index = self._trajectory_routes.pop(key, None)
            if index is None:
                return
            weight = self._trajectory_weights.pop(key, 1.0)
            self._active_trajectories[index] -= 1
            self._active_trajectory_weight[index] = max(0.0, self._active_trajectory_weight[index] - float(weight))
            if self._active_trajectories[index] < 0:
                self._active_trajectories[index] = 0
                raise AssertionError('negative sticky trajectory route load')

    def trajectory_route_snapshot(self) -> dict[str, Any]:
        with self._route_lock:
            return {'active_trajectories_per_replica': list(self._active_trajectories), 'active_predicted_weight_per_replica': [round(value, 3) for value in self._active_trajectory_weight], 'registered_trajectories': len(self._trajectory_routes)}

    def _next_backend(self) -> NeosOpenAICompatModel:
        trajectory_key = getattr(self._state, 'trajectory_key', None)
        if trajectory_key is not None:
            index = self._assign_trajectory(str(trajectory_key))
            backend = self.backends[index]
            self._state.last_backend = backend
            self._state.last_route = {'policy': 'least-loaded-trajectory-sticky', 'replica_index': index}
            return backend
        with self._counter_lock:
            index = next(self._counter) % len(self.backends)
        backend = self.backends[index]
        self._state.last_backend = backend
        self._state.last_route = {'policy': 'per-call-round-robin-unbound', 'replica_index': index}
        return backend

    def ensure_neos_chat_template(self, expected_context_length: int | None=None) -> None:
        for backend in self.backends:
            backend.ensure_neos_chat_template(expected_context_length)

    def backend_manifest(self) -> dict[str, Any]:
        manifests = [backend.backend_manifest() for backend in self.backends]
        return {'type': 'local_openai_compatible_round_robin', 'message_protocol': NATIVE_MESSAGE_PROTOCOL, 'model': self.model, 'base_urls': [backend.base_url for backend in self.backends], 'routing': 'least-loaded-trajectory-sticky-with-unbound-round-robin', 'trajectory_routing': {'assignment': 'least-model-inflight-then-predicted-weight-then-count', 'lifetime': 'trajectory-until-terminal-release', 'prefix_cache_locality': True, 'unbound_call_fallback': 'per-call-round-robin'}, 'replicas': manifests}

    def take_last_response_metadata(self) -> dict | None:
        backend = getattr(self._state, 'last_backend', None)
        take = getattr(backend, 'take_last_response_metadata', None)
        value = take() if callable(take) else None
        if value is None:
            return None
        value = dict(value)
        value['trajectory_routing'] = dict(getattr(self._state, 'last_route', {}) or {})
        return value

    def chat(self, *args, **kwargs) -> dict[str, Any]:
        return self._next_backend().chat(*args, **kwargs)

def load_model_backend(spec: str):
    if spec.startswith(('relaychat:', 'azure:')):
        return NeosAzureNativeModel(model=spec.split(':', 1)[1])
    url, _, model = spec.rpartition(':')
    if url.startswith('http') and model:
        base_urls = [value.strip() for value in url.split(',') if value.strip()]
        if len(base_urls) > 1:
            return NeosRoundRobinOpenAICompatModel(base_urls, model)
        return NeosOpenAICompatModel(base_url=url, model=model)
    raise ValueError(f'cannot parse Neos model backend spec: {spec!r}')
