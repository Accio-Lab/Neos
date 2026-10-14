from __future__ import annotations
from neos.identifiers import new_id
import json
import re
from collections.abc import Iterable
from typing import Any
from neos.workflow.toolsets import redact_disabled_tool_mentions
from neos.workflow.protocol import QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION, NeosAssistantTurn
from neos.workflow.schemas import CROP_IMAGE_MAX_PER_IMAGE, CROP_IMAGE_MAX_TOTAL, FETCH_IMAGE_MAX_URLS, IMAGE_SEARCH_MAX_QUERIES, NEOS_ARRAY_TOOL_SCHEMA, NEOS_TOOL_NAMES, REVERSE_IMAGE_SEARCH_MAX_INPUTS, WEB_VISIT_MAX_PAGES, validate_tool_arguments
_SERIALIZATION_TAG_RE = re.compile('</?(?:think|thinking|tool_call|tool_response|answer)>', re.IGNORECASE)
_TOOL_BLOCK_RE = re.compile('<tool_call>\\s*(.*?)\\s*</tool_call>', re.IGNORECASE | re.DOTALL)
_ANSWER_BLOCK_RE = re.compile('<answer>\\s*(.*?)\\s*</answer>', re.IGNORECASE | re.DOTALL)
_SPECIAL_TOKEN_RE = re.compile('<\\|(?:im_start|im_end|assistant|endoftext)\\|>', re.IGNORECASE)
_TOOL_NAME_BY_CASEFOLD = {name.casefold(): name for name in NEOS_TOOL_NAMES}
_TOOL_NAME_ALIASES = {'websearch': 'TextSearch', 'text_search': 'TextSearch', 'web_visit': 'WebVisit', 'websummary': 'WebVisit', 'web_summary': 'WebVisit', 'image_search': 'ImageSearch', 'reverse_image_search': 'ReverseImageSearch', 'fetch_image': 'FetchImage', 'crop_image': 'CropImage', 'image_zoom_in': 'CropImage'}
_TOOL_NAME_ALIASES = {alias: name for alias, name in _TOOL_NAME_ALIASES.items() if name in NEOS_TOOL_NAMES}

def _text_content(value: Any) -> str | None:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return None
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict) or item.get('type') != 'text':
            return None
        parts.append(str(item.get('text') or ''))
    return '\n'.join(parts)

def _clean_text(value: str) -> str:
    value = _SPECIAL_TOKEN_RE.sub('', value)
    value = _SERIALIZATION_TAG_RE.sub('', value)
    return value.strip()

def _merge_reasoning(*parts: str) -> str:
    values = [_clean_text(part) for part in parts if str(part or '').strip()]
    return '\n\n'.join((value for value in values if value)).strip()

def _adaptation(original_error: str | None, transforms: Iterable[str], *, profile: str=QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, adapter_version: str=QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION, **extra: Any) -> dict:
    return {'profile': profile, 'adapter_version': adapter_version, 'original_error': original_error, 'transforms': list(dict.fromkeys(transforms)), **extra}

def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            from json_repair import loads as repair_json_loads
            decoded = repair_json_loads(value, skip_json_loads=True)
        except Exception:
            return None
    if isinstance(decoded, str) and decoded != value:
        try:
            return json.loads(decoded)
        except (json.JSONDecodeError, TypeError):
            pass
    return decoded

def _canonical_tool_name(value: Any) -> str | None:
    name = str(value or '').strip()
    if not name:
        return None
    if name in NEOS_TOOL_NAMES:
        return name
    folded = name.casefold()
    return _TOOL_NAME_BY_CASEFOLD.get(folded) or _TOOL_NAME_ALIASES.get(folded)

def _flatten_strings(value: Any, *, keys: tuple[str, ...]=()) -> list[str]:
    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    for item in values:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        selected = None
        for key in keys:
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip():
                selected = candidate.strip()
                break
        if selected:
            result.append(selected)
    return result

def _goal(value: Any, *, question: str, reasoning: str) -> str:
    values = _flatten_strings(value, keys=('goal', 'text', 'query'))
    if values:
        return '\n'.join(dict.fromkeys(values))[:4000]
    question = str(question or '').strip()
    if question:
        return f"Extract evidence from these results that is relevant to answering the user's question: {question[:3000]}"
    reasoning = _clean_text(reasoning)
    if reasoning:
        return 'Extract evidence needed for the current reasoning step: ' + reasoning[-3000:]
    return "Extract evidence relevant to the user's question."

def _normalize_token(value: Any, available_image_refs: set[str]) -> str | None:
    token = str(value or '').strip()
    if re.fullmatch('img_\\d+', token):
        return token if not available_image_refs or token in available_image_refs else None
    match = re.search('(\\d+)$', token)
    if match:
        candidate = f'img_{int(match.group(1))}'
        if not available_image_refs or candidate in available_image_refs:
            return candidate
    return None

def _collect_boxes(value: Any) -> list[list[float]]:
    if isinstance(value, str):
        decoded = _decode_json(value)
        return _collect_boxes(decoded) if decoded is not None else []
    if isinstance(value, list) and len(value) == 4 and all((isinstance(item, (int, float)) and (not isinstance(item, bool)) for item in value)):
        return [[float(item) for item in value]]
    if isinstance(value, dict):
        if all((key in value for key in ('left', 'top', 'right', 'bottom'))):
            coordinates = [value['left'], value['top'], value['right'], value['bottom']]
            if all((isinstance(item, (int, float)) and (not isinstance(item, bool)) for item in coordinates)):
                return [[float(item) for item in coordinates]]
        for key in ('box', 'bbox', 'bbox_2d', 'crop'):
            if key in value:
                return _collect_boxes(value[key])
        return []
    if not isinstance(value, list):
        return []
    if len(value) > 4 and all((isinstance(item, (int, float)) and (not isinstance(item, bool)) for item in value)):
        return [[float(item) for item in value[:4]]]
    result: list[list[float]] = []
    for item in value:
        result.extend(_collect_boxes(item))
    return result

def _normalize_box(box: list[float]) -> list[int] | None:
    if box and all((0.0 <= value <= 1.0 for value in box)):
        box = [value * 1000.0 for value in box]
    left, top, right, bottom = box
    if min(left, right) >= 0 and max(left, right) > 1000:
        scale = 1000.0 / max(left, right)
        left, right = (left * scale, right * scale)
    if min(top, bottom) >= 0 and max(top, bottom) > 1000:
        scale = 1000.0 / max(top, bottom)
        top, bottom = (top * scale, bottom * scale)
    left, top, right, bottom = [max(0.0, min(1000.0, value)) for value in (left, top, right, bottom)]
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    if left == right or top == bottom:
        return None
    return [round(left), round(top), round(right), round(bottom)]

def _function_from_object(value: Any) -> tuple[str | None, Any]:
    if not isinstance(value, dict):
        return (None, None)
    function = value.get('function')
    if isinstance(function, dict):
        return (_canonical_tool_name(function.get('name')), function.get('arguments'))
    name = value.get('name') or value.get('tool') or value.get('tool_name')
    arguments = value.get('arguments', value.get('args'))
    if arguments is None:
        arguments = {key: item for key, item in value.items() if key not in {'name', 'tool', 'tool_name', 'type', 'id', 'index'}}
    return (_canonical_tool_name(name), arguments)

def _normalize_call(name: str | None, arguments: Any, *, question: str, reasoning: str, available_image_refs: set[str], tool_schema_version: str) -> tuple[str, dict[str, Any], list[str]] | None:
    if name is None:
        return None
    transforms: list[str] = []
    decoded = _decode_json(arguments)
    if name == 'TextSearch':
        if isinstance(decoded, (str, list)):
            decoded = {'query': decoded}
            transforms.append('wrap_textsearch_arguments')
        if not isinstance(decoded, dict):
            return None
        args = dict(decoded)
        if 'query' not in args and 'queries' in args:
            args['query'] = args.pop('queries')
            transforms.append('rename_queries_to_query')
        if 'query' not in args and set(args) == {'goal'}:
            args['query'] = args.pop('goal')
            transforms.append('use_textsearch_goal_as_query')
        queries = _flatten_strings(args.get('query'), keys=('query',))
        if not queries:
            return None
        args = {'query': queries}
        if set(decoded if isinstance(decoded, dict) else {}) - {'query', 'queries'}:
            transforms.append('drop_unsupported_textsearch_options')
    elif name == 'WebVisit':
        if isinstance(decoded, list):
            decoded = {'url': decoded}
            transforms.append('wrap_webvisit_batch')
        if not isinstance(decoded, dict):
            return None
        raw = dict(decoded)
        if 'url' not in raw and 'urls' in raw:
            raw['url'] = raw.pop('urls')
            transforms.append('rename_urls_to_url')
        urls = _flatten_strings(raw.get('url'), keys=('url',))
        if not urls:
            return None
        item_goals = []
        values = raw.get('url') if isinstance(raw.get('url'), list) else []
        for item in values:
            if isinstance(item, dict) and item.get('goal'):
                item_goals.append(item['goal'])
        goal_value = raw.get('goal') or item_goals
        if not raw.get('goal'):
            transforms.append('supply_webvisit_goal')
        if len(urls) > WEB_VISIT_MAX_PAGES:
            urls = urls[:WEB_VISIT_MAX_PAGES]
            transforms.append('truncate_webvisit_batch')
        args = {'url': urls, 'goal': _goal(goal_value, question=question, reasoning=reasoning)}
    elif name == 'ImageSearch':
        if isinstance(decoded, (str, list)):
            decoded = {'query': decoded}
            transforms.append('wrap_imagesearch_arguments')
        if not isinstance(decoded, dict):
            return None
        raw = dict(decoded)
        if 'query' not in raw and 'queries' in raw:
            raw['query'] = raw.pop('queries')
            transforms.append('rename_queries_to_query')
        queries = _flatten_strings(raw.get('query'), keys=('query',))
        if not queries:
            return None
        if len(queries) > IMAGE_SEARCH_MAX_QUERIES:
            queries = queries[:IMAGE_SEARCH_MAX_QUERIES]
            transforms.append('truncate_imagesearch_batch')
        if not raw.get('goal'):
            transforms.append('supply_imagesearch_goal')
        args = {'query': queries, 'goal': _goal(raw.get('goal'), question=question, reasoning=reasoning)}
    elif name == 'ReverseImageSearch':
        if not isinstance(decoded, dict):
            return None
        raw = dict(decoded)
        if 'query' in raw and (not any((key in raw for key in ('image', 'image_token', 'image_tokens', 'image_url')))):
            converted = _normalize_call('ImageSearch', {'query': raw.get('query'), 'goal': raw.get('goal')}, question=question, reasoning=reasoning, available_image_refs=available_image_refs, tool_schema_version=tool_schema_version)
            if converted is not None:
                converted_name, converted_args, converted_changes = converted
                return (converted_name, converted_args, ['convert_reverse_query_to_imagesearch', *converted_changes])
        if 'image_token' not in raw and 'image_tokens' in raw:
            raw['image_token'] = raw.pop('image_tokens')
            transforms.append('rename_image_tokens')
        if 'goal' not in raw and 'goals' in raw:
            raw['goal'] = raw.pop('goals')
            transforms.append('rename_goals_to_goal')
        ambiguous = raw.pop('image', None)
        if ambiguous is not None:
            values = ambiguous if isinstance(ambiguous, list) else [ambiguous]
            tokens: list[str] = []
            urls: list[str] = []
            for item in values:
                if isinstance(item, dict):
                    item = item.get('image_token') or item.get('image_url') or item.get('url')
                token = _normalize_token(item, available_image_refs)
                if token:
                    tokens.append(token)
                elif isinstance(item, str) and item.startswith(('http://', 'https://')):
                    urls.append(item)
            if tokens and 'image_token' not in raw:
                raw['image_token'] = tokens
            if urls and 'image_url' not in raw:
                raw['image_url'] = urls
            transforms.append('classify_reverse_image_inputs')
        tokens = [token for value in _flatten_strings(raw.get('image_token'), keys=('image_token',)) if (token := _normalize_token(value, available_image_refs))]
        urls = _flatten_strings(raw.get('image_url'), keys=('image_url', 'url'))
        urls = [url for url in urls if url.startswith(('http://', 'https://'))]
        combined = [('token', value) for value in tokens] + [('url', value) for value in urls]
        if not combined:
            return None
        if len(combined) > REVERSE_IMAGE_SEARCH_MAX_INPUTS:
            combined = combined[:REVERSE_IMAGE_SEARCH_MAX_INPUTS]
            transforms.append('truncate_reverse_image_inputs')
        tokens = [value for kind, value in combined if kind == 'token']
        urls = [value for kind, value in combined if kind == 'url']
        args = {'goal': _goal(raw.get('goal'), question=question, reasoning=reasoning)}
        if tokens:
            args['image_token'] = tokens
        if urls:
            args['image_url'] = urls
    elif name == 'FetchImage':
        if isinstance(decoded, (str, list)):
            decoded = {'urls': decoded}
            transforms.append('wrap_fetchimage_arguments')
        if not isinstance(decoded, dict):
            return None
        raw = dict(decoded)
        if 'urls' not in raw and 'url' in raw:
            raw['urls'] = raw.pop('url')
            transforms.append('rename_url_to_urls')
        values = raw.get('urls')
        values = values if isinstance(values, list) else [values]
        urls: list[str | dict[str, str]] = []
        for item in values:
            if isinstance(item, str) and item.strip():
                urls.append(item.strip())
            elif isinstance(item, dict) and isinstance(item.get('url'), str):
                normalized_item = {'url': item['url'].strip()}
                if isinstance(item.get('caption'), str) and item['caption'].strip():
                    normalized_item['caption'] = item['caption'].strip()
                urls.append(normalized_item)
            elif isinstance(item, dict) and isinstance(item.get('source_url'), str):
                normalized_item = {'url': item['source_url'].strip()}
                caption = item.get('caption') or item.get('title')
                if isinstance(caption, str) and caption.strip():
                    normalized_item['caption'] = caption.strip()
                urls.append(normalized_item)
        if not urls:
            return None
        if len(urls) > FETCH_IMAGE_MAX_URLS:
            urls = urls[:FETCH_IMAGE_MAX_URLS]
            transforms.append('truncate_fetchimage_batch')
        args = {'urls': urls}
        if set(raw) - {'urls', 'url'}:
            transforms.append('drop_fetchimage_followup_options')
    elif name == 'CropImage':
        if not isinstance(decoded, dict):
            return None
        raw = dict(decoded)
        config = raw.get('crop_config')
        if config is None and isinstance(raw.get('config'), dict):
            config = raw['config']
            transforms.append('rename_config_to_crop_config')
        if config is None:
            source = raw.get('image_token') or raw.get('image') or raw.get('source') or raw.get('image_url') or raw.get('url')
            boxes = raw.get('boxes', raw.get('box', raw.get('crop')))
            token = _normalize_token(source, available_image_refs)
            if token and boxes is not None:
                config = {token: boxes}
                transforms.append('construct_crop_config')
            elif isinstance(source, list) and boxes is not None:
                tokens = [token for value in source if (token := _normalize_token(value, available_image_refs))]
                if len(tokens) == len(source) and isinstance(boxes, list) and (len(boxes) == len(tokens)):
                    config = dict(zip(tokens, boxes))
                    transforms.append('construct_parallel_crop_config')
            elif isinstance(source, str) and source.startswith(('http://', 'https://')):
                fetch_args = {'urls': source}
                fetch_args['urls'] = [fetch_args['urls']]
                if validate_tool_arguments('FetchImage', fetch_args, schema_version=tool_schema_version) is None:
                    return ('FetchImage', fetch_args, ['fetch_before_crop_url'])
        if not isinstance(config, dict):
            return None
        normalized_config: dict[str, Any] = {}
        total = 0
        for raw_token, raw_boxes in config.items():
            token = _normalize_token(raw_token, available_image_refs)
            if token is None:
                continue
            boxes: list[list[int]] = []
            for raw_box in _collect_boxes(raw_boxes):
                box = _normalize_box(raw_box)
                if box is not None and box not in boxes:
                    boxes.append(box)
            if not boxes:
                continue
            if len(boxes) > CROP_IMAGE_MAX_PER_IMAGE:
                boxes = boxes[:CROP_IMAGE_MAX_PER_IMAGE]
                transforms.append('truncate_crop_boxes_per_image')
            remaining = CROP_IMAGE_MAX_TOTAL - total
            if remaining <= 0:
                break
            boxes = boxes[:remaining]
            total += len(boxes)
            normalized_config[token] = boxes[0] if len(boxes) == 1 else boxes
        if not normalized_config:
            return None
        args = {'crop_config': normalized_config}
        if args != {'crop_config': config}:
            transforms.append('normalize_crop_config')
    else:
        return None
    if validate_tool_arguments(name, args, schema_version=tool_schema_version) is not None:
        return None
    return (name, args, transforms)

def _native_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False, separators=(',', ':'))}}

def _generated_call_id(payload: str) -> str:
    return 'call_adapter_' + new_id()

def _tool_payloads(content: str) -> list[str]:
    openings = list(re.finditer('<tool_call>', content, flags=re.IGNORECASE))
    if openings:
        recovered = []
        for index, opening in enumerate(openings):
            boundary = openings[index + 1].start() if index + 1 < len(openings) else len(content)
            suffix = content[opening.end():boundary]
            suffix = re.split('</tool_call>|</(?:think|thinking|answer)>|<\\|(?:im_start|im_end)[^>]*\\|>', suffix, maxsplit=1, flags=re.IGNORECASE)[0]
            payload = suffix.strip()
            if payload:
                recovered.append(payload)
        if recovered:
            return recovered
    payloads = [match.group(1).strip() for match in _TOOL_BLOCK_RE.finditer(content)]
    if payloads:
        return payloads
    closing = re.search('</tool_call>\\s*$', content, flags=re.IGNORECASE)
    if closing:
        prefix = content[:closing.start()].strip()
        object_start = prefix.find('{')
        if object_start >= 0:
            return [prefix[object_start:]]
    return []

def _answer_from_content(content: str) -> tuple[str | None, str, list[str]]:
    transforms: list[str] = []
    answers = list(_ANSWER_BLOCK_RE.finditer(content))
    if answers:
        selected = answers[-1]
        answer = _clean_text(selected.group(1))
        reasoning = _clean_text(content[:selected.start()] + '\n' + content[selected.end():])
        if answer:
            return (answer, reasoning, ['extract_answer_block'])
    unclosed = list(re.finditer('<answer>', content, flags=re.IGNORECASE))
    if unclosed:
        selected = unclosed[-1]
        answer = _clean_text(content[selected.end():])
        reasoning = _clean_text(content[:selected.start()])
        if answer:
            return (answer, reasoning, ['close_unclosed_answer'])
    think_closes = list(re.finditer('</(?:think|thinking)>', content, flags=re.IGNORECASE))
    if think_closes:
        selected = think_closes[-1]
        before = _clean_text(content[:selected.start()])
        after = _clean_text(content[selected.end():])
        if after:
            return (after, before, ['split_at_orphan_think_close'])
        if before:
            return (before, '', ['drop_trailing_orphan_think_close'])
    if _SERIALIZATION_TAG_RE.search(content):
        return (None, '', transforms)
    cleaned = _clean_text(content)
    explicit = _explicit_answer_from_reasoning(cleaned)
    if explicit:
        transforms.append('extract_explicit_answer_from_visible_content')
        return (explicit, cleaned, transforms)
    return (None, '', transforms)
_EXPLICIT_ANSWER_PATTERNS = (re.compile('(?:^|\\n)[^\\n]{0,100}?\\b(?:final|concise|concisely\\s+supported)?\\s*answer(?:\\s+is)?\\s*[:：]\\s*[\\"“]?([^\\n]{1,2048}?)[\\"”]?\\s*(?:\\n|$)', re.IGNORECASE), re.compile('(?:^|\\n)\\s*(?:final\\s+answer|answer)\\s*[:：]\\s*(.+?)\\s*$', re.IGNORECASE | re.DOTALL), re.compile('(?:therefore|thus|so),?\\s+(?:the\\s+)?(?:final\\s+)?answer\\s+is\\s*[:：]?\\s*(.+?)\\s*$', re.IGNORECASE | re.DOTALL), re.compile('(?:i(?:\'ll|\\s+will)\\s+(?:answer|output)|the\\s+final\\s+concise\\s+answer\\s+is)\\s*[:：]?\\s*[\\"“]?(.+?)[\\"”]?\\s*$', re.IGNORECASE | re.DOTALL))

def _explicit_answer_from_reasoning(reasoning: str) -> str | None:
    tail = _clean_text(reasoning)[-6000:]
    candidates: list[tuple[int, str]] = []
    for pattern in _EXPLICIT_ANSWER_PATTERNS:
        for match in pattern.finditer(tail):
            candidate = match.group(1).strip()
            if 0 < len(candidate) <= 2048 and (not _SERIALIZATION_TAG_RE.search(candidate)):
                candidates.append((match.start(), candidate))
    if not candidates:
        return None
    answer = max(candidates, key=lambda item: item[0])[1]
    return answer.strip().strip('"“”')

def adapt_native_assistant_message(message: dict[str, Any] | None, strict: NeosAssistantTurn, *, question: str, available_image_refs: Iterable[str], forced_final: bool=False, tool_schema_version: str='native_array', profile: str=QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, adapter_version: str=QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION) -> NeosAssistantTurn:
    pass
    if strict.valid or not isinstance(message, dict):
        return strict
    content = _text_content(message.get('content'))
    if content is None:
        return strict
    raw_reasoning = message.get('reasoning_content')
    if raw_reasoning is None:
        raw_reasoning = message.get('reasoning')
    if raw_reasoning is not None and (not isinstance(raw_reasoning, str)):
        return strict
    reasoning = _clean_text(str(raw_reasoning or ''))
    image_refs = set(available_image_refs)
    calls = message.get('tool_calls') or []
    transforms: list[str] = []
    if isinstance(calls, list) and calls:
        valid_calls: list[tuple[int, str, dict[str, Any], list[str], str]] = []
        for index, raw_call in enumerate(calls):
            name, raw_arguments = _function_from_object(raw_call)
            normalized = _normalize_call(name, raw_arguments, question=question, reasoning=reasoning, available_image_refs=image_refs, tool_schema_version=tool_schema_version)
            if normalized is None:
                continue
            normalized_name, arguments, changes = normalized
            call_id = str(raw_call.get('id') or '').strip() if isinstance(raw_call, dict) else ''
            if not call_id:
                call_id = _generated_call_id(json.dumps(raw_call, sort_keys=True, default=str))
                changes.append('supply_tool_call_id')
            valid_calls.append((index, normalized_name, arguments, changes, call_id))
        if len(valid_calls) == 1:
            index, name, arguments, changes, call_id = valid_calls[0]
            transforms.extend(changes)
            if len(calls) > 1:
                transforms.append('select_only_schema_valid_native_call')
            visible = _clean_text(content)
            if visible:
                reasoning = _merge_reasoning(reasoning, visible)
                transforms.append('move_tool_turn_content_to_reasoning')
            if not transforms:
                transforms.append('normalize_native_tool_call')
            normalized_message = {'role': 'assistant', 'content': '', 'tool_calls': [_native_call(call_id, name, arguments)]}
            if reasoning:
                normalized_message['reasoning_content'] = reasoning
            return NeosAssistantTurn(valid=True, reasoning=reasoning, action_kind='tool_call', tool_call={'id': call_id, 'name': name, 'arguments': arguments}, message=normalized_message, adaptation=_adaptation(strict.error, transforms, profile=profile, adapter_version=adapter_version, selected_native_call_index=index))
    textual_calls: list[tuple[int, str, dict[str, Any], list[str], str]] = []
    textual_answers: list[tuple[int, str]] = []
    for index, payload in enumerate(_tool_payloads(content)):
        repaired_json = False
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            value = _decode_json(payload)
            repaired_json = value is not None
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for item in values:
            if isinstance(item, dict) and str(item.get('name') or '').casefold() in {'answer', 'finalanswer', 'final_answer', 'final', 'pyfinaltext'}:
                answer_arguments = _decode_json(item.get('arguments'))
                if isinstance(answer_arguments, dict):
                    answer = next((answer_arguments.get(key) for key in ('answer', 'final', 'text', 'content', 'output') if isinstance(answer_arguments.get(key), str)), None)
                    if isinstance(answer, str) and 0 < len(answer.strip()) <= 2048:
                        textual_answers.append((index, _clean_text(answer)))
                continue
            name, raw_arguments = _function_from_object(item)
            normalized = _normalize_call(name, raw_arguments, question=question, reasoning=reasoning, available_image_refs=image_refs, tool_schema_version=tool_schema_version)
            if normalized is None:
                continue
            normalized_name, arguments, changes = normalized
            if repaired_json:
                changes = ['repair_hermes_json', *changes]
            textual_calls.append((index, normalized_name, arguments, changes, payload))
    if len(textual_calls) > 1:
        unique_calls: dict[str, tuple[int, str, dict[str, Any], list[str], str]] = {}
        for call in textual_calls:
            marker = json.dumps([call[1], call[2]], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            unique_calls.setdefault(marker, call)
        if len(unique_calls) == 1:
            textual_calls = [next(iter(unique_calls.values()))]
            textual_calls[0][3].append('collapse_identical_repeated_tool_calls')
    if textual_answers and (not textual_calls or textual_answers[-1][0] >= textual_calls[-1][0]):
        index, answer = textual_answers[-1]
        transforms.append('normalize_answer_named_pseudotool')
        remainder = _TOOL_BLOCK_RE.sub('', content)
        reasoning = _merge_reasoning(reasoning, remainder)
        normalized_message = {'role': 'assistant', 'content': answer}
        if reasoning:
            normalized_message['reasoning_content'] = reasoning
        return NeosAssistantTurn(valid=True, reasoning=reasoning, action_kind='answer', answer=answer, message=normalized_message, adaptation=_adaptation(strict.error, transforms, profile=profile, adapter_version=adapter_version, textual_block_index=index))
    if len(textual_calls) == 1:
        index, name, arguments, changes, payload = textual_calls[-1]
        transforms.extend(changes)
        transforms.append('recover_textual_tool_call')
        remainder = _TOOL_BLOCK_RE.sub('', content)
        remainder = re.sub('<tool_call>.*$', '', remainder, flags=re.IGNORECASE | re.DOTALL)
        reasoning = _merge_reasoning(reasoning, remainder)
        call_id = _generated_call_id(payload)
        normalized_message = {'role': 'assistant', 'content': '', 'tool_calls': [_native_call(call_id, name, arguments)]}
        if reasoning:
            normalized_message['reasoning_content'] = reasoning
        return NeosAssistantTurn(valid=True, reasoning=reasoning, action_kind='tool_call', tool_call={'id': call_id, 'name': name, 'arguments': arguments}, message=normalized_message, adaptation=_adaptation(strict.error, transforms, profile=profile, adapter_version=adapter_version, textual_block_index=index))
    answer, extra_reasoning, answer_transforms = _answer_from_content(content)
    if answer:
        reasoning = _merge_reasoning(reasoning, extra_reasoning)
        transforms.extend(answer_transforms)
        normalized_message = {'role': 'assistant', 'content': answer}
        if reasoning:
            normalized_message['reasoning_content'] = reasoning
        return NeosAssistantTurn(valid=True, reasoning=reasoning, action_kind='answer', answer=answer, message=normalized_message, adaptation=_adaptation(strict.error, transforms, profile=profile, adapter_version=adapter_version))
    if reasoning:
        answer = _explicit_answer_from_reasoning(reasoning)
        if answer is None and forced_final and (len(reasoning) <= 2048):
            answer = reasoning
            transforms.append('promote_short_forced_final_reasoning')
        if answer:
            transforms.append('extract_explicit_answer_from_reasoning')
            normalized_message = {'role': 'assistant', 'reasoning_content': reasoning, 'content': answer}
            return NeosAssistantTurn(valid=True, reasoning=reasoning, action_kind='answer', answer=answer, message=normalized_message, adaptation=_adaptation(strict.error, transforms, profile=profile, adapter_version=adapter_version))
    return strict

def adapt_qwen35_native_assistant_message(message: dict[str, Any] | None, strict: NeosAssistantTurn, *, question: str, available_image_refs: Iterable[str], forced_final: bool=False) -> NeosAssistantTurn:
    pass
    return adapt_native_assistant_message(message, strict, question=question, available_image_refs=available_image_refs, forced_final=forced_final, tool_schema_version=NEOS_ARRAY_TOOL_SCHEMA, profile=QWEN35_NATIVE_COMPAT_ASSISTANT_FORMAT_PROFILE, adapter_version=QWEN35_NATIVE_COMPAT_FORMAT_ADAPTER_VERSION)

def build_format_retry_instruction(error: str | None, *, forced_final: bool) -> str:
    pass
    if forced_final:
        action = 'Return the final answer now as ordinary assistant text; tools are disabled.'
    else:
        action = 'Retry the same turn with exactly one action: either one native function call through the supplied tools field, or the final answer as ordinary text.'
    rendered_error = _SPECIAL_TOKEN_RE.sub('', str(error or 'invalid assistant message'))
    rendered_error = _SERIALIZATION_TAG_RE.sub('[reserved serialization token]', rendered_error)
    rendered_error = redact_disabled_tool_mentions(rendered_error)
    return f'Your previous response could not be accepted by the native message schema ({rendered_error[:600]}). {action} Do not print XML tags, JSON wrappers, tool narration, or more than one function call.'
