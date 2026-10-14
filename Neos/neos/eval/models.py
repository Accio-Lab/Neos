from __future__ import annotations
import base64
import functools
import json
import logging
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen
from neos.chat_template import NEOS_CHAT_TEMPLATE_PATH, load_neos_chat_template
from neos.protocol import parse_serialized_assistant_turn
logger = logging.getLogger(__name__)
_tls = threading.local()
AZURE_NATIVE_MESSAGE_PROTOCOL = 'openai-chat-completions-native-tools-official-multimodal'
AZURE_MAX_IMAGES_PER_REQUEST = 50
AZURE_MAX_DECODED_IMAGE_BYTES = 8 * 1024 * 1024
_CONTEXT_LENGTH_ERROR_RE = re.compile('maximum context length of\\s+(?P<context>[0-9]+)\\s+tokens.*?requested a total of\\s+(?P<total>[0-9]+)\\s+tokens:\\s*(?P<input>[0-9]+)\\s+tokens from the input messages and\\s*(?P<completion>[0-9]+)\\s+tokens for the completion', re.IGNORECASE | re.DOTALL)
_INPUT_CONTEXT_LENGTH_ERROR_RE = re.compile("the input\\s*\\((?P<input>[0-9]+)\\s+tokens\\)\\s+is longer than\\s+the model(?:'s|’s) context length\\s*\\((?P<context>[0-9]+)\\s+tokens\\)", re.IGNORECASE | re.DOTALL)

class ModelContextLengthExceededError(RuntimeError):
    pass

    def __init__(self, *, input_tokens: int, context_tokens: int, original_error: BaseException | str) -> None:
        self.input_tokens = int(input_tokens)
        self.context_tokens = int(context_tokens)
        self.original_error = str(original_error)
        super().__init__(f'model input exceeds the available native context: input_tokens={self.input_tokens}, context_tokens={self.context_tokens}')

def context_length_overflow_counts(error: BaseException | str) -> tuple[int, int] | None:
    pass
    rendered = str(error)
    match = _INPUT_CONTEXT_LENGTH_ERROR_RE.search(rendered)
    if match is None:
        match = _CONTEXT_LENGTH_ERROR_RE.search(rendered)
    if match is None:
        return None
    return (int(match.group('input')), int(match.group('context')))

def reduced_max_tokens_for_context_error(error: BaseException | str, requested_max_tokens: int) -> int | None:
    pass
    match = _CONTEXT_LENGTH_ERROR_RE.search(str(error))
    if match is None:
        return None
    context_tokens = int(match.group('context'))
    input_tokens = int(match.group('input'))
    available = context_tokens - input_tokens - 1
    if available <= 0 or available >= requested_max_tokens:
        return None
    return available

def _message_value(message, name: str):
    pass
    if isinstance(message, dict):
        return message.get(name)
    value = getattr(message, name, None)
    if value is not None:
        return value
    model_extra = getattr(message, 'model_extra', None)
    if isinstance(model_extra, dict):
        return model_extra.get(name)
    return None

def _reasoning_block(reasoning: str, close: bool) -> str:
    pass
    text = reasoning.strip()
    if not text.startswith('<think>'):
        text = '<think>\n' + text
    if close and (not text.rstrip().endswith('</think>')):
        text += '\n</think>'
    return text

def _native_tool_calls_block(message) -> str:
    pass
    raw_calls = _message_value(message, 'tool_calls') or []
    blocks = []
    for raw_call in raw_calls:
        function = _message_value(raw_call, 'function') or raw_call
        name = _message_value(function, 'name')
        arguments = _message_value(function, 'arguments')
        if not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = arguments
        payload = {'name': str(name), 'arguments': {} if arguments is None else arguments}
        blocks.append('<tool_call>\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n</tool_call>')
    return '\n'.join(blocks)

def reconstruct_assistant_text(message, finish_reason: str | None=None) -> str:
    pass
    raw_content = _message_value(message, 'content')
    raw_reasoning = _message_value(message, 'reasoning_content') or _message_value(message, 'reasoning')
    content = '' if raw_content is None else str(raw_content)
    reasoning = '' if raw_reasoning is None else str(raw_reasoning)
    native_calls = _native_tool_calls_block(message)
    if native_calls and '<tool_call>' not in content:
        content = f'{content.rstrip()}\n{native_calls}'.lstrip()
    if not reasoning.strip():
        return content
    if '<think>' in content:
        return content
    if '</think>' in content:
        return _reasoning_block(reasoning, close=False) + '\n' + content.lstrip()
    if content:
        return _reasoning_block(reasoning, close=True) + '\n' + content
    truncated = str(finish_reason or '').lower() == 'length'
    return _reasoning_block(reasoning, close=not truncated)

def _text_first_multimodal_content(content):
    pass
    if not isinstance(content, list):
        return content
    text_blocks = [part for part in content if part.get('type') == 'text']
    other_blocks = [part for part in content if part.get('type') != 'text']
    return [*text_blocks, *other_blocks]

def neos_messages_to_openai_native(messages: list[dict]) -> list[dict]:
    pass
    converted: list[dict] = []
    pending_tool_ids: list[str] = []
    assistant_index = 0
    for message in messages:
        role = message.get('role')
        content = message.get('content')
        if role in {'system', 'user'}:
            if pending_tool_ids:
                raise ValueError('user/system message appeared before all native tool results were supplied')
            converted.append({'role': role, 'content': _text_first_multimodal_content(content)})
            continue
        if role == 'assistant':
            if pending_tool_ids:
                raise ValueError('assistant message appeared before all native tool results were supplied')
            assistant_index += 1
            parsed = parse_serialized_assistant_turn(str(content or ''))
            if not parsed.valid:
                raise ValueError(f'cannot reconstruct native assistant history from an invalid Neos turn: {parsed.error}')
            if parsed.action_kind == 'tool_calls':
                native_calls = []
                for call_index, call in enumerate(parsed.tool_calls, start=1):
                    call_id = f'call_neos_a{assistant_index:03d}_c{call_index:02d}'
                    pending_tool_ids.append(call_id)
                    native_calls.append({'id': call_id, 'type': 'function', 'function': {'name': call['name'], 'arguments': json.dumps(call['arguments'], ensure_ascii=False, separators=(',', ':'))}})
                converted.append({'role': 'assistant', 'content': None, 'tool_calls': native_calls})
            else:
                converted.append({'role': 'assistant', 'content': parsed.answer or ''})
            continue
        if role == 'tool':
            if not pending_tool_ids:
                raise ValueError('native tool result has no preceding tool call')
            converted.append({'role': 'tool', 'tool_call_id': pending_tool_ids.pop(0), 'content': _text_first_multimodal_content(content)})
            continue
        raise ValueError(f'unsupported message role for Azure native tools: {role!r}')
    if pending_tool_ids:
        raise ValueError(f'missing {len(pending_tool_ids)} native tool result message(s)')
    return converted

def _image_part_url(part: dict) -> str | None:
    if not isinstance(part, dict) or part.get('type') != 'image_url':
        return None
    image_url = part.get('image_url')
    if isinstance(image_url, dict):
        value = image_url.get('url')
    else:
        value = image_url
    return value if isinstance(value, str) else None

def _replace_image_part_url(part: dict, url: str) -> dict:
    replacement = dict(part)
    image_url = part.get('image_url')
    if isinstance(image_url, dict):
        replacement['image_url'] = {**image_url, 'url': url}
    else:
        replacement['image_url'] = {'url': url}
    return replacement

def _decoded_data_uri_size(data_uri: str) -> int | None:
    header, separator, payload = data_uri.partition(',')
    if not separator or ';base64' not in header.lower():
        return None
    padding = len(payload) - len(payload.rstrip('='))
    return max(0, len(payload) * 3 // 4 - padding)

@functools.lru_cache(maxsize=64)
def _azure_reencode_data_uri(data_uri: str, max_decoded_bytes: int) -> str:
    pass
    import io
    from PIL import Image
    header, separator, payload = data_uri.partition(',')
    if not separator or ';base64' not in header.lower():
        return data_uri
    raw = base64.b64decode(payload, validate=True)
    if len(raw) <= max_decoded_bytes:
        return data_uri
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert('RGB')
    if max(image.size) > 2048:
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    quality = 90
    while True:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= max_decoded_bytes:
            return 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii')
        width, height = image.size
        if max(width, height) <= 384:
            raise ValueError(f'cannot fit Azure image payload below {max_decoded_bytes} decoded bytes')
        image = image.resize((max(1, int(width * 0.8)), max(1, int(height * 0.8))), Image.Resampling.LANCZOS)
        quality = max(65, quality - 5)

def prepare_azure_native_messages(native_messages: list[dict], *, max_images: int=AZURE_MAX_IMAGES_PER_REQUEST, max_decoded_image_bytes: int=AZURE_MAX_DECODED_IMAGE_BYTES) -> tuple[list[dict], dict]:
    pass
    if max_images <= 0:
        raise ValueError('max_images must be positive')
    if max_decoded_image_bytes <= 0:
        raise ValueError('max_decoded_image_bytes must be positive')
    image_positions: list[tuple[int, int, str]] = []
    for message_index, message in enumerate(native_messages):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if _image_part_url(part) is not None:
                image_positions.append((message_index, part_index, str(message.get('role') or '')))
    fixed_positions = [position for position in image_positions if position[2] != 'tool']
    tool_positions = [position for position in image_positions if position[2] == 'tool']
    if len(fixed_positions) > max_images:
        raise ValueError(f'question/non-tool images exceed Azure request limit: {len(fixed_positions)} > {max_images}')
    tool_image_budget = max_images - len(fixed_positions)
    omitted_positions = set(tool_positions[:max(0, len(tool_positions) - tool_image_budget)])
    prepared: list[dict] = []
    reencoded = 0
    for message_index, message in enumerate(native_messages):
        content = message.get('content')
        if not isinstance(content, list):
            prepared.append(message)
            continue
        new_content = []
        for part_index, part in enumerate(content):
            position = (message_index, part_index, str(message.get('role') or ''))
            if position in omitted_positions:
                continue
            url = _image_part_url(part)
            decoded_size = _decoded_data_uri_size(url) if url is not None else None
            if url is not None and decoded_size is not None and (decoded_size > max_decoded_image_bytes):
                url = _azure_reencode_data_uri(url, max_decoded_image_bytes)
                part = _replace_image_part_url(part, url)
                reencoded += 1
            new_content.append(part)
        prepared.append({**message, 'content': new_content})
    sent_images = len(image_positions) - len(omitted_positions)
    return (prepared, {'provider_limit': max_images, 'input_images': len(image_positions), 'sent_images': sent_images, 'question_or_non_tool_images': len(fixed_positions), 'tool_images': len(tool_positions), 'omitted_oldest_tool_images': len(omitted_positions), 'max_decoded_image_bytes': max_decoded_image_bytes, 'reencoded_oversized_images': reencoded})

def _native_response_to_neos_action(message, finish_reason: str | None) -> str:
    pass
    native_calls = _native_tool_calls_block(message)
    if native_calls:
        return native_calls
    content = str(_message_value(message, 'content') or '').strip()
    if not content:
        return ''
    parsed = parse_serialized_assistant_turn(content)
    if parsed.valid:
        return content
    if str(finish_reason or '').lower() == 'length':
        return content
    safe_content = re.sub('</?(?:think|tool_call|tool_response|answer)>', lambda match: '[' + match.group(0)[1:-1] + ']', content, flags=re.IGNORECASE)
    return f'<answer>{safe_content}</answer>'

def image_to_data_uri(path: str | Path, max_side: int=1024, quality: int=85) -> str:
    pass
    import io
    from PIL import Image
    img = Image.open(path).convert('RGB')
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

def image_content(path: str | Path, **kw) -> dict:
    return {'type': 'image_url', 'image_url': {'url': image_to_data_uri(path, **kw)}}

class OpenAICompatModel:
    pass

    def __init__(self, base_url: str, model: str, api_key: str='', max_retries: int=8):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self._template_check_lock = threading.Lock()
        self._template_verified = False
        self._verified_context_length: int | None = None
        self._state = threading.local()

    def take_last_response_metadata(self) -> dict | None:
        metadata = getattr(self._state, 'last_response_metadata', None)
        self._state.last_response_metadata = None
        return metadata

    def _client(self):
        if getattr(_tls, 'oai_client', None) is None or _tls.oai_client_base != self.base_url:
            from openai import OpenAI
            _tls.oai_client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600, max_retries=0)
            _tls.oai_client_base = self.base_url
        return _tls.oai_client

    def _server_info_url(self) -> str:
        parts = urlsplit(self.base_url)
        path = parts.path.rstrip('/')
        if path.endswith('/v1'):
            path = path[:-3]
        return urlunsplit((parts.scheme, parts.netloc, path + '/server_info', '', ''))

    @staticmethod
    def _template_content_from_server_value(value: str) -> str | None:
        pass
        if not value:
            return None
        if '{%' in value or '{{' in value:
            return value
        candidate = Path(value)
        try:
            if candidate.is_file():
                return candidate.read_text(encoding='utf-8')
        except OSError:
            return None
        return None

    def ensure_neos_chat_template(self, expected_context_length=None):
        self._template_verified = True
        self._verified_context_length = expected_context_length

    def chat(self, messages: list[dict], max_tokens: int=4096, temperature: float=0.0, top_p: float | None=None, tools: list[dict] | None=None, tool_choice: str='auto') -> str:
        if tools:
            self.ensure_neos_chat_template()
        last_err: Exception | None = None
        request_max_tokens = max_tokens
        attempt_timings = []
        self._state.last_response_metadata = None
        for attempt in range(self.max_retries):
            attempt_started = time.perf_counter()
            try:
                kwargs = {}
                if top_p is not None:
                    kwargs['top_p'] = top_p
                if tools:
                    kwargs['tools'] = tools
                    kwargs['tool_choice'] = tool_choice
                resp = self._client().chat.completions.create(model=self.model, messages=messages, max_tokens=request_max_tokens, temperature=temperature, **kwargs)
                choice = resp.choices[0]
                usage = getattr(resp, 'usage', None)
                if hasattr(usage, 'model_dump'):
                    usage = usage.model_dump()
                elif not isinstance(usage, dict):
                    usage = None
                attempt_timings.append({'attempt': attempt + 1, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3), 'status': 'success'})
                self._state.last_response_metadata = {'request': {'api_surface': '/chat/completions', 'base_url': self.base_url, 'model': self.model, 'message_count': len(messages), 'tool_schema_count': len(tools or []), 'tool_choice': tool_choice if tools else None, 'requested_max_tokens': max_tokens, 'effective_max_tokens': request_max_tokens}, 'response': {'finish_reason': choice.finish_reason, 'usage': usage or {}}, 'attempts': attempt_timings}
                return reconstruct_assistant_text(choice.message, finish_reason=choice.finish_reason)
            except Exception as e:
                last_err = e
                attempt_timings.append({'attempt': attempt + 1, 'latency_ms': round((time.perf_counter() - attempt_started) * 1000, 3), 'status': 'error', 'error_type': type(e).__name__, 'status_code': getattr(e, 'status_code', None)})
                reduced = reduced_max_tokens_for_context_error(e, request_max_tokens)
                if reduced is not None:
                    logger.warning('Reducing max_tokens from %d to %d to fit the server-reported context window', request_max_tokens, reduced)
                    request_max_tokens = reduced
                    continue
                overflow = context_length_overflow_counts(e)
                if overflow is not None:
                    input_tokens, context_tokens = overflow
                    if context_tokens - input_tokens <= 1:
                        raise ModelContextLengthExceededError(input_tokens=input_tokens, context_tokens=context_tokens, original_error=e) from e
                if getattr(e, 'status_code', None) == 400:
                    break
                time.sleep(min(2 ** attempt * 5, 60))
                logger.warning('OpenAICompatModel retry %d: %s', attempt + 1, e)
        raise RuntimeError(f'OpenAICompatModel failed: {last_err}')

class AzureModel:
    pass
    official_multimodal_message_order = True
    message_protocol = AZURE_NATIVE_MESSAGE_PROTOCOL

    def __init__(self, model: str, max_retries: int=8):
        self.model = model
        self.max_retries = max_retries
        self.requested_reasoning_effort = 'xhigh'
        self._state = threading.local()
        self._preflight_logged = False

    def _client(self):
        if getattr(_tls, 'relay_chat_native_client', None) is None:
            import openai
            from neos.eval import secrets
            _tls.relay_chat_native_client = openai.OpenAI(default_headers=secrets.request_headers('RELAY_CHAT'), api_key=secrets.RELAY_CHAT_API_KEY, base_url=secrets.endpoint(secrets.RELAY_CHAT_ENDPOINT, 'NEOS_RELAY_CHAT_ENDPOINT'))
        return _tls.relay_chat_native_client

    def ensure_neos_chat_template(self, expected_context_length=None):
        self._template_verified = True
        self._verified_context_length = expected_context_length

    def backend_manifest(self) -> dict:
        return {'type': 'relay_openai_native_tools', 'message_protocol': self.message_protocol, 'client': 'openai.OpenAI', 'base_url': self._client().base_url.__str__(), 'api_surface': '/chat/completions', 'model': self.model, 'official_multimodal_order': 'text_then_images', 'native_tools': True, 'requested_reasoning_effort': self.requested_reasoning_effort, 'sent_reasoning_effort_with_tools': None, 'sent_reasoning_effort_without_tools': self.requested_reasoning_effort, 'reasoning_note': 'Relay tool turns use the model default reasoning setting; a no-tools finalization turn uses xhigh.'}

    def take_last_response_metadata(self) -> dict | None:
        metadata = getattr(self._state, 'last_response_metadata', None)
        self._state.last_response_metadata = None
        return metadata

    def chat(self, messages: list[dict], max_tokens: int=4096, temperature: float=0.0, top_p: float | None=None, tools: list[dict] | None=None, tool_choice: str='auto') -> str:
        native_messages = neos_messages_to_openai_native(messages)
        native_messages, image_budget = prepare_azure_native_messages(native_messages)
        last_err: Exception | None = None
        self._state.last_response_metadata = None
        for attempt in range(self.max_retries):
            try:
                kwargs = {'model': self.model, 'messages': native_messages, 'max_tokens': max_tokens}
                if tools:
                    kwargs['tools'] = tools
                    kwargs['tool_choice'] = tool_choice
                else:
                    kwargs['reasoning_effort'] = self.requested_reasoning_effort
                response = self._client().chat.completions.create(**kwargs)
                if not response.choices:
                    payload = response.model_dump()
                    raise RuntimeError('Relay gateway returned no choices: ' + str(payload.get('message') or payload)[:1000])
                choice = response.choices[0]
                response_payload = response.model_dump()
                self._state.last_response_metadata = {'request': {'api_surface': '/chat/completions', 'model': self.model, 'message_count': len(native_messages), 'message_roles': [message.get('role') for message in native_messages], 'initial_user_content_order': [part.get('type') for message in native_messages if message.get('role') == 'user' and isinstance(message.get('content'), list) for part in message['content']][:16], 'tool_schema_count': len(tools or []), 'tool_choice': tool_choice if tools else None, 'max_completion_tokens': max_tokens, 'reasoning_effort_requested': self.requested_reasoning_effort, 'reasoning_effort_sent': None if tools else self.requested_reasoning_effort, 'image_budget': image_budget}, 'response': response_payload}
                return _native_response_to_neos_action(choice.message, finish_reason=choice.finish_reason)
            except Exception as exc:
                last_err = exc
                status_code = getattr(exc, 'status_code', None)
                if status_code == 400 or (isinstance(exc, RuntimeError) and 'returned no choices' in str(exc)):
                    break
                if attempt + 1 < self.max_retries:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning('RelayChatModel retry %d/%d: %s; sleep %.1fs', attempt + 1, self.max_retries, exc, wait)
                    time.sleep(wait)
        raise RuntimeError(f'RelayChatModel failed after {attempt + 1} attempt(s): {last_err}')

def load_model_backend(spec: str):
    pass
    if spec.startswith(('relaychat:', 'azure:')):
        return AzureModel(model=spec.split(':', 1)[1])
    url, _, model = spec.rpartition(':')
    if url.startswith('http') and model:
        return OpenAICompatModel(base_url=url, model=model)
    raise ValueError(f'cannot parse model backend spec: {spec!r}')
