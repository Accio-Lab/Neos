from __future__ import annotations
import json
import logging
import os
import re
import subprocess
import threading
import time
from collections.abc import Sequence
from types import SimpleNamespace
from neos.eval import secrets
from neos.protocol import VISIT_PAGE_TEXT_CHARS
from neos.tools.online_client import OnlineSearchClient
logger = logging.getLogger(__name__)
_tls = threading.local()
_fatal_circuits: dict[str, str] = {}
_fatal_circuit_lock = threading.Lock()

class FatalLLMGatewayError(RuntimeError):
    pass

class NonRetryableLLMGatewayError(RuntimeError):
    pass

class RetryExhaustedLLMGatewayError(RuntimeError):
    pass

def _azure_client():
    if getattr(_tls, 'azure_client', None) is None:
        import openai
        _tls.azure_client = openai.AzureOpenAI(default_headers=secrets.request_headers('AZURE'), api_key=secrets.AZURE_API_KEY, api_version=secrets.AZURE_API_VERSION, azure_endpoint=secrets.endpoint(secrets.AZURE_ENDPOINT, 'NEOS_AZURE_ENDPOINT'))
    return _tls.azure_client

def _deepseek_client():
    if getattr(_tls, 'deepseek_client', None) is None:
        import openai
        _tls.deepseek_client = openai.OpenAI(default_headers=secrets.request_headers('DEEPSEEK'), api_key=secrets.DEEPSEEK_API_KEY, base_url=secrets.endpoint(secrets.DEEPSEEK_ENDPOINT, 'NEOS_DEEPSEEK_ENDPOINT'))
    return _tls.deepseek_client

def _dashscope_client():
    if getattr(_tls, 'dashscope_client', None) is None:
        import openai
        _tls.dashscope_client = openai.OpenAI(default_headers=secrets.request_headers('DASHSCOPE'), api_key=secrets.DASHSCOPE_API_KEY, base_url=secrets.endpoint(secrets.DASHSCOPE_ENDPOINT, 'NEOS_DASHSCOPE_ENDPOINT'), max_retries=0)
    return _tls.dashscope_client

def _gemini_client():
    if getattr(_tls, 'gemini_client', None) is None:
        import openai
        _tls.gemini_client = openai.OpenAI(default_headers=secrets.request_headers('GEMINI'), api_key=secrets.GEMINI_API_KEY, base_url=secrets.endpoint(secrets.GEMINI_ENDPOINT, 'NEOS_GEMINI_ENDPOINT'))
    return _tls.gemini_client

def _relay_chat_client():
    if getattr(_tls, 'relay_chat_client', None) is None:
        import openai
        _tls.relay_chat_client = openai.OpenAI(default_headers=secrets.request_headers('RELAY_CHAT'), api_key=secrets.RELAY_CHAT_API_KEY, base_url=secrets.endpoint(secrets.RELAY_CHAT_ENDPOINT, 'NEOS_RELAY_CHAT_ENDPOINT'))
    return _tls.relay_chat_client

def _local_mei_client():
    if getattr(_tls, 'local_mei_client', None) is None:
        import openai
        _tls.local_mei_client = openai.OpenAI(api_key=secrets.LOCAL_MEI_API_KEY, base_url=secrets.endpoint(secrets.LOCAL_MEI_ENDPOINT, 'NEOS_LOCAL_MEI_ENDPOINT'), max_retries=0, timeout=300)
    return _tls.local_mei_client

def _is_deepseek_model(model: str) -> bool:
    return str(model).lower().startswith('deepseek-')

def _is_dashscope_model(model: str) -> bool:
    return str(model).lower().startswith('qwen')

def _is_gemini_model(model: str) -> bool:
    return str(model).lower().startswith('google/gemini-')

def _is_gpt_model(model: str) -> bool:
    return str(model).lower().startswith('gpt-')

def _is_local_mei_model(model: str) -> bool:
    return str(model).lower().startswith('local/')

def _local_mei_model_name(model: str) -> str:
    value = str(model)
    selected = value.split('/', 1)[1] if '/' in value else value
    if not selected:
        raise ValueError('local MEI model name must not be empty')
    return selected

def _is_relay_responses_model(model: str) -> bool:
    return str(model).lower().startswith('relay/')

def _relay_model_name(model: str) -> str:
    value = str(model)
    return value.split('/', 1)[1] if '/' in value else value

def _provider_name(model: str) -> str:
    if _is_local_mei_model(model):
        return 'local_mei'
    if _is_relay_responses_model(model):
        return 'relay_responses'
    if _is_deepseek_model(model):
        return 'deepseek'
    if _is_dashscope_model(model):
        return 'dashscope'
    if _is_gemini_model(model):
        return 'google'
    if _is_gpt_model(model):
        return 'relay_chat'
    return 'azure'

def _response_payload(response) -> dict:
    dump = getattr(response, 'model_dump', None)
    if not callable(dump):
        return {}
    value = dump(exclude_none=True)
    return value if isinstance(value, dict) else {}

def _response_usage(response) -> dict:
    pass
    usage = _response_payload(response).get('usage')
    return dict(usage) if isinstance(usage, dict) else {}

def _systemic_failure(message: str) -> bool:
    lowered = str(message).lower()
    return any((marker in lowered for marker in ('limiting error', '86400 seconds', 'quota exceeded', 'quota exhausted', 'insufficient_quota', 'invalid api key', 'unauthorized', 'authentication', 'model_not_found', 'deploymentnotfound')))

def _transient_failure(message: str) -> bool:
    lowered = str(message).lower()
    return any((marker in lowered for marker in ('throttling.concurrency', 'too many concurrent requests', 'rate_limit', 'rate limit', 'temporarily unavailable', 'service unavailable', 'timeout')))

def _open_fatal_circuit(model: str, message: str) -> None:
    with _fatal_circuit_lock:
        _fatal_circuits[model] = message

def _check_fatal_circuit(model: str) -> None:
    with _fatal_circuit_lock:
        message = _fatal_circuits.get(model)
    if message:
        raise FatalLLMGatewayError(f'{model} provider circuit is open after a systemic failure: {message}')

def _reset_fatal_circuits_for_tests() -> None:
    with _fatal_circuit_lock:
        _fatal_circuits.clear()

def _completion_parts(response, model: str) -> tuple[str, str]:
    payload = _response_payload(response)
    if payload.get('success') is False:
        message = str(payload.get('message') or json.dumps(payload, ensure_ascii=False))
        rendered = f'{model} returned success=false: {message[:4000]}'
        if _systemic_failure(rendered):
            _open_fatal_circuit(model, rendered)
            raise FatalLLMGatewayError(rendered)
        if _transient_failure(rendered):
            raise RuntimeError(rendered)
        raise NonRetryableLLMGatewayError(rendered)
    choices = getattr(response, 'choices', None)
    if not choices:
        raise RuntimeError(f'{model} returned no completion choices; payload={json.dumps(payload, ensure_ascii=False)[:4000]}')
    message = choices[0].message
    content = message.content or ''
    reasoning = getattr(message, 'reasoning_content', None)
    if reasoning is None:
        reasoning = getattr(message, 'reasoning', None)
    if not isinstance(reasoning, str):
        choice_payload = (payload.get('choices') or [{}])[0]
        message_payload = choice_payload.get('message') if isinstance(choice_payload, dict) else None
        message_payload = message_payload if isinstance(message_payload, dict) else {}
        reasoning = message_payload.get('reasoning_content')
        if reasoning is None:
            reasoning = message_payload.get('reasoning')
    reasoning = str(reasoning or '').strip()
    content = str(content)
    local_thinking_enabled = _is_local_mei_model(model) and os.getenv('NEOS_LOCAL_MEI_ENABLE_THINKING', '0').strip().lower() in {'1', 'true'}
    inline = re.match('\\A\\s*<think>\\s*(.*?)\\s*</think>\\s*(.*)\\Z', content, flags=re.DOTALL) if local_thinking_enabled else None
    if inline:
        if not reasoning:
            reasoning = inline.group(1).strip()
        content = inline.group(2).strip()
    if not str(content).strip():
        raise RuntimeError(f'{model} returned an empty completion')
    return (str(content), reasoning)

def _completion_text(response, model: str) -> str:
    pass
    return _completion_parts(response, model)[0]

def _responses_input(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                kind = str(item.get('type') or '')
                if kind in {'text', 'input_text'}:
                    parts.append({'type': 'input_text', 'text': str(item.get('text') or '')})
                elif kind in {'image_url', 'input_image'}:
                    image = item.get('image_url')
                    url = image.get('url') if isinstance(image, dict) else image
                    if not isinstance(url, str) or not url:
                        raise ValueError('relay Responses image_url must be a non-empty string')
                    parts.append({'type': 'input_image', 'image_url': url})
                else:
                    raise ValueError(f'unsupported relay Responses content type: {kind}')
            content = parts
        converted.append({'role': str(message.get('role') or 'user'), 'content': content})
    return converted

def _responses_text(payload: dict) -> str:
    if payload.get('success') is False:
        raise RuntimeError('relay Responses API returned success=false: ' + str(payload.get('message') or json.dumps(payload, ensure_ascii=False))[:4000])
    status = str(payload.get('status') or '')
    if status and status != 'completed':
        raise RuntimeError('relay Responses API did not complete: ' + json.dumps({'status': status, 'error': payload.get('error'), 'incomplete_details': payload.get('incomplete_details')}, ensure_ascii=False)[:4000])
    values = []
    if isinstance(payload.get('output_text'), str):
        values.append(payload['output_text'])
    for item in payload.get('output') or []:
        if not isinstance(item, dict) or item.get('type') != 'message':
            continue
        for content in item.get('content') or []:
            if isinstance(content, dict) and content.get('type') in {'output_text', 'text'}:
                values.append(str(content.get('text') or ''))
    text = ''.join(values).strip()
    if not text:
        raise RuntimeError('relay Responses API returned no output_text')
    return text

class _RelayCompletion:

    def __init__(self, payload: dict, text: str) -> None:
        self._payload = payload
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]

    def model_dump(self, **_kwargs):
        return self._payload

def _relay_responses_request(selected_model: str, messages: list[dict], max_completion_tokens: int):
    payload = {'model': _relay_model_name(selected_model), 'input': _responses_input(messages), 'max_output_tokens': int(max_completion_tokens), 'reasoning': {'effort': secrets.RELAY_REASONING_EFFORT}, 'store': False}
    command = ['curl', '--silent', '--show-error', '--fail-with-body', '--connect-timeout', '30', '--max-time', str(secrets.RELAY_TIMEOUT_SECONDS), secrets.endpoint(secrets.RELAY_RESPONSES_ENDPOINT, 'NEOS_RELAY_RESPONSES_ENDPOINT'), '-H', 'Content-Type: application/json', '--data-binary', '@-']
    headers = secrets.request_headers('RELAY_RESPONSES')
    if secrets.RELAY_RESPONSES_API_KEY:
        headers.setdefault('Authorization', 'Bearer ' + secrets.RELAY_RESPONSES_API_KEY)
    for name, value in headers.items():
        command += ['-H', f'{name}: {value}']
    completed = subprocess.run(command, input=json.dumps(payload, ensure_ascii=False, separators=(',', ':')), text=True, capture_output=True, timeout=secrets.RELAY_TIMEOUT_SECONDS + 15, check=False)
    if completed.returncode:
        raise RuntimeError(f'relay Responses curl failed rc={completed.returncode}: {(completed.stdout or completed.stderr)[:4000]}')
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'relay Responses returned invalid JSON: {completed.stdout[:4000]}') from exc
    text = _responses_text(response)
    return _RelayCompletion(response, text)

class LLMGateway:
    pass

    def __init__(self, model: str | None=None, max_retries: int=3, fallback_models: Sequence[str] | str | None=None):
        self.model = model or secrets.JUDGE_MODEL
        if max_retries <= 0:
            raise ValueError('max_retries must be positive')
        self.max_retries = max_retries
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
        self._call_local = threading.local()

    @property
    def model_chain(self) -> tuple[str, ...]:
        return (self.model, *self.fallback_models)

    def take_last_call(self) -> dict | None:
        pass
        value = getattr(self._call_local, 'last', None)
        self._call_local.last = None
        return dict(value) if isinstance(value, dict) else None

    def _finish_call_record(self, *, started: float, configured_models: tuple[str, ...], attempts: list[dict], actual_model: str | None, reasoning_content: str='', error: BaseException | None=None) -> None:
        attempted_models = list(dict.fromkeys((row['model'] for row in attempts)))
        record = {'configured_models': list(configured_models), 'attempted_models': attempted_models, 'actual_model': actual_model, 'fallback_used': bool(actual_model != configured_models[0] if actual_model is not None else len(attempted_models) > 1), 'attempts': attempts, 'total_ms': round((time.perf_counter() - started) * 1000, 3)}
        if error is not None:
            record['error'] = f'{type(error).__name__}: {error}'[:4000]
        if reasoning_content:
            record['reasoning_content_present'] = True
            record['reasoning_content_characters'] = len(reasoning_content)
        self._call_local.last = record

    @staticmethod
    def _request_completion(selected_model: str, messages: list[dict], max_completion_tokens: int):
        if _is_local_mei_model(selected_model):
            extra_body: dict = {'repetition_penalty': 1.0}
            thinking = os.getenv('NEOS_LOCAL_MEI_ENABLE_THINKING')
            if thinking is not None:
                normalized = thinking.strip().lower()
                if normalized not in {'0', '1', 'false', 'true'}:
                    raise ValueError('NEOS_LOCAL_MEI_ENABLE_THINKING must be 0/1/false/true')
                extra_body['chat_template_kwargs'] = {'enable_thinking': normalized in {'1', 'true'}}
            return _local_mei_client().chat.completions.create(model=_local_mei_model_name(selected_model), messages=messages, max_tokens=max_completion_tokens, temperature=0.0, top_p=1.0, extra_body=extra_body)
        if _is_relay_responses_model(selected_model):
            return _relay_responses_request(selected_model, messages, max_completion_tokens)
        if _is_gemini_model(selected_model):
            return _gemini_client().chat.completions.create(model=selected_model, messages=messages, max_completion_tokens=max_completion_tokens, temperature=0.0)
        if _is_deepseek_model(selected_model):
            return _deepseek_client().chat.completions.create(model=selected_model, messages=messages, max_tokens=max(max_completion_tokens, secrets.DEEPSEEK_JUDGE_MIN_COMPLETION_TOKENS), temperature=0.0)
        if _is_dashscope_model(selected_model):
            return _dashscope_client().chat.completions.create(model=selected_model, messages=messages, max_completion_tokens=max(max_completion_tokens, secrets.DASHSCOPE_MIN_COMPLETION_TOKENS), temperature=0.8)
        if _is_gpt_model(selected_model):
            return _relay_chat_client().chat.completions.create(model=selected_model, messages=messages, max_tokens=max_completion_tokens)
        return _azure_client().chat.completions.create(model=selected_model, messages=messages, max_completion_tokens=max_completion_tokens)

    def chat(self, messages: list[dict], max_completion_tokens: int=8192, model: str | None=None) -> str:
        pass
        last_err: Exception | None = None
        configured_models = (str(model),) if model is not None else self.model_chain
        started = time.perf_counter()
        attempts: list[dict] = []
        for route_index, selected_model in enumerate(configured_models):
            is_final_model = route_index + 1 == len(configured_models)
            attempt_limit = self.max_retries if is_final_model else 1
            for attempt in range(attempt_limit):
                attempt_started = time.perf_counter()
                try:
                    _check_fatal_circuit(selected_model)
                    response = self._request_completion(selected_model, messages, max_completion_tokens)
                    text, reasoning_content = _completion_parts(response, selected_model)
                    attempt_record = {'model': selected_model, 'route_index': route_index, 'attempt': attempt + 1, 'ok': True, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3)}
                    usage = _response_usage(response)
                    if usage:
                        attempt_record['usage'] = usage
                    attempts.append(attempt_record)
                    self._finish_call_record(started=started, configured_models=configured_models, attempts=attempts, actual_model=selected_model, reasoning_content=reasoning_content)
                    return text
                except NonRetryableLLMGatewayError as exc:
                    last_err = exc
                    attempts.append({'model': selected_model, 'route_index': route_index, 'attempt': attempt + 1, 'ok': False, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3), 'error': f'{type(exc).__name__}: {exc}'[:4000]})
                    if is_final_model:
                        wrapped = RuntimeError(f'LLMGateway non-retryable failure: {exc}')
                        self._finish_call_record(started=started, configured_models=configured_models, attempts=attempts, actual_model=None, error=wrapped)
                        raise wrapped from exc
                    break
                except FatalLLMGatewayError as exc:
                    last_err = exc
                    attempts.append({'model': selected_model, 'route_index': route_index, 'attempt': attempt + 1, 'ok': False, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3), 'error': f'{type(exc).__name__}: {exc}'[:4000]})
                    if is_final_model:
                        self._finish_call_record(started=started, configured_models=configured_models, attempts=attempts, actual_model=None, error=exc)
                        raise
                    break
                except Exception as exc:
                    routed_error: Exception = exc
                    if _systemic_failure(str(exc)):
                        _open_fatal_circuit(selected_model, str(exc)[:4000])
                        routed_error = FatalLLMGatewayError(f'{selected_model} systemic provider failure: {exc}')
                    last_err = routed_error
                    attempts.append({'model': selected_model, 'route_index': route_index, 'attempt': attempt + 1, 'ok': False, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3), 'error': f'{type(routed_error).__name__}: {routed_error}'[:4000]})
                    if isinstance(routed_error, FatalLLMGatewayError):
                        if is_final_model:
                            self._finish_call_record(started=started, configured_models=configured_models, attempts=attempts, actual_model=None, error=routed_error)
                            raise routed_error from exc
                        break
                    wait = min(2 ** attempt * 5, 120)
                    if attempt + 1 < attempt_limit:
                        logger.warning('LLMGateway retry model=%s %d/%d: %s; sleep %.1fs', selected_model, attempt + 1, attempt_limit, routed_error, wait)
                        time.sleep(wait)
                        continue
                    break
            if not is_final_model:
                logger.warning('LLMGateway fallback %s -> %s after: %s', selected_model, configured_models[route_index + 1], last_err)
        exhausted = RetryExhaustedLLMGatewayError(f"LLMGateway exhausted model route {' -> '.join(configured_models)}: {last_err}")
        self._finish_call_record(started=started, configured_models=configured_models, attempts=attempts, actual_model=None, error=exhausted)
        raise exhausted

class SearchGateway(OnlineSearchClient):
    pass
_default_search_gateway: SearchGateway | None = None
_default_search_lock = threading.Lock()

def _shared_search_gateway() -> SearchGateway:
    global _default_search_gateway
    if _default_search_gateway is None:
        with _default_search_lock:
            if _default_search_gateway is None:
                _default_search_gateway = SearchGateway()
    return _default_search_gateway

def fetch_page_text(url: str, max_chars: int=VISIT_PAGE_TEXT_CHARS, timeout: int=20) -> str:
    pass
    return _shared_search_gateway().fetch_page_text(url, max_chars=max_chars, timeout=timeout)
