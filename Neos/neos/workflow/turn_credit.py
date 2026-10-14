from __future__ import annotations
import asyncio
import base64
import copy
import fcntl
import io
import json
import math
import os
import re
import socket
import time
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageDraw
from neos.eval import secrets
from neos.workflow.credit_transport import CreditControl, CreditDeadlineError, current_control, guarded_credit_call, run_credit_worker
CREDIT_SCHEMA_VERSION = 'neos-failed-turn-credit'
CREDIT_PROMPT_VERSION = 'neos-turn-credit-inline-productive-visible'
COMPONENT_CREDIT_SCHEMA_VERSION = 'neos-failed-turn-credit-causal-components'
COMPONENT_CREDIT_PROMPT_VERSION = 'neos-turn-credit-causal-cot-act-productive-visible'
CREDIT_IMAGE_RENDER_VERSION = 'bounded-min-side-labelled-images'
DEFAULT_CREDIT_MODEL = 'qwen3.7-plus'
VALID_TURN_STATUSES = frozenset({'productive', 'neutral', 'causal_error', 'downstream_consequence', 'recovery', 're_derailment'})
CAUSAL_TURN_STATUSES = frozenset({'causal_error', 're_derailment'})
_JSON_FENCE_RE = re.compile('```(?:json)?\\s*(.*?)```', re.DOTALL | re.IGNORECASE)
CREDIT_SYSTEM_PROMPT = "You are a causal credit-assignment judge for a multimodal web-research policy.\n\nThe candidate trajectory has already been judged to have a wrong final answer. Compare its complete trajectory with one independently verified successful reference trajectory for the same question. Identify policy-authored assistant turns that causally pushed the candidate onto a wrong branch. This is causal process supervision, not imitation learning and not another final-answer grading task.\n\nRules:\n1. The reference is one valid solution, not the only valid solution. Never penalize a different search query, evidence order, tool choice, or reasoning path merely for differing from the reference when it remains valid and useful.\n2. Grade assistant decisions only. A failed URL, empty search result, provider timeout, or misleading webpage is an environment observation, not itself a policy error. The policy may be at fault only when it ignores contrary evidence, overclaims what pixels/text establish, repeatedly pursues a disproven hypothesis, or makes an unjustified decision from that observation.\n3. Distinguish a causal error from its downstream consequences. Mark the decision turn that makes the wrong branch materially more likely as causal_error; later turns that merely continue the same unrecovered branch are downstream_consequence, not additional root errors.\n4. Detect recovery. If the candidate later rejects or repairs the earlier mistake and again approaches a valid solution, mark that turn recovery. If it subsequently makes a new independent error, mark that later turn re_derailment. There may be multiple error events only when genuine recovery or independent re-branching occurred between them.\n5. Use visual evidence only when the corresponding image is available in the candidate's history up to that turn. Titles, URLs, captions, snippets, and MEI summaries are not pixel evidence. img_N numbering is local to each trajectory.\n6. A wrong final-answer turn is causal only if it introduces/selects an unsupported answer after viable evidence remained. If it merely states the inevitable result of an earlier unrecovered mistake, mark it downstream_consequence.\n7. Return only significant causal/recovery events and genuinely productive turns. Omitted turns are neutral only outside an active error branch. During an active error branch, unannotated turns receive downstream-consequence credit through the post-processing rule. Use only candidate span IDs shown in the trajectory; do not invent or duplicate IDs.\n\nEvent meanings:\n- causal_error: first currently unrecovered decision that diverts the solution;\n- recovery: explicitly repairs an earlier error and restores a viable route;\n- re_derailment: a new causal error after recovery, or an independent later error.\n\nList a turn under productive_turns only when it establishes valid evidence or materially advances a viable solution. An explicitly productive turn receives positive credit even if an earlier error has not yet been repaired; it must not be relabeled as a downstream consequence. Productive is not automatically recovery: only an explicit recovery event clears the earlier error, and other intervening turns retain their chronological downstream decay. Outside an active error branch, omitted turns are neutral.\n\nSome terminal tool events are marked observation_delivered_to_policy=false: the tool ran, but its result did not fit the remaining policy context and was not presented to the policy. Their audit-only response/pixels are intentionally absent here. Do not infer those unavailable results or blame the policy for not using them. Images may be uniformly enlarged and/or padded for API compatibility; the border and img_N label are not evidence and no new scene detail is created.\n\nReturn strict JSON only, matching the requested schema. Severity and confidence are numbers in [0,1]. Keep rationales concise and evidence-specific."
CAUSAL_COMPONENT_RULES = "\n\nAdditional rule, ONLY for events you label causal_error or re_derailment:\nSeparate that assistant turn into cot (its reasoning) and act (its visible tool\ncall, tool arguments, or final answer). Return components.cot and components.act,\neach with status, severity, confidence, and a concise rationale.\n- error: this component makes the causal mistake; penalize only this component.\n- productive: this component independently establishes valid evidence or makes\n  useful progress; a small positive credit is appropriate.\n- neutral: not culpable, but no independently useful progress; no change.\nAt least one component of each causal_error/re_derailment must be error.\nA correct reasoning conclusion followed by an unjustified repeated action or\nwrong visible answer does NOT make the reasoning erroneous. Conversely, do not\nautomatically punish a useful evidence-gathering action just because reasoning\nwas mistaken. Do not label both components error merely because the turn's\noverall event is causal. Correct but redundant text is neutral, not necessarily\nproductive. An absent/empty component must be neutral. When an event includes\ncomponent_token_presence, a false component had no newly generated tokens and\nmust be neutral even if a normalized answer was recovered from its reasoning.\nDo not split any other turn: recovery events must have components=null;\nproductive_turns and all omitted/downstream turns keep the existing turn-level\ncontract. Never turn an environment fetch failure into a policy error by itself.\nThe existing reasoning and tool_call/answer fields are the component contents;\ndo not repeat those contents or invent additional component span IDs.\n"

def causal_component_credit_enabled() -> bool:
    return os.getenv('NEOS_TURN_CREDIT_CAUSAL_COMPONENTS', '0').lower() in {'1', 'true', 'yes', 'on'}

def credit_schema_version() -> str:
    return COMPONENT_CREDIT_SCHEMA_VERSION if causal_component_credit_enabled() else CREDIT_SCHEMA_VERSION

def credit_prompt_version() -> str:
    return COMPONENT_CREDIT_PROMPT_VERSION if causal_component_credit_enabled() else CREDIT_PROMPT_VERSION

def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')

def _float01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0

def _json_object(text: str) -> dict[str, Any] | None:
    value = str(text or '').strip()
    candidates = [value]
    match = _JSON_FENCE_RE.search(value)
    if match:
        candidates.insert(0, match.group(1).strip())
    first, last = (value.find('{'), value.rfind('}'))
    if first >= 0 and last > first:
        candidates.append(value[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None

def _encode_credit_image(raw: Image.Image, *, ref: str, label: str) -> dict[str, Any]:
    pass
    max_side = max(256, int(os.getenv('NEOS_TURN_CREDIT_IMAGE_MAX_SIDE', '1024')))
    quality = max(45, min(95, int(os.getenv('NEOS_TURN_CREDIT_IMAGE_JPEG_QUALITY', '82'))))
    min_side = max(16, min(256, int(os.getenv('NEOS_TURN_CREDIT_IMAGE_MIN_SIDE', '32'))))
    source_size = raw.size
    image = raw.convert('RGB')
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    scale = min(max(1.0, min_side / min(image.size)), max_side / max(image.size))
    if scale > 1.0:
        size = tuple((min(max_side, max(1, math.ceil(value * scale))) for value in image.size))
        image = image.resize(size, Image.Resampling.NEAREST)
    labelled = Image.new('RGB', (max(128, min_side, image.width), max(min_side, image.height) + 28), 'white')
    labelled.paste(image, (0, 28))
    ImageDraw.Draw(labelled).text((8, 7), label, fill='black')
    buffer = io.BytesIO()
    labelled.save(buffer, format='JPEG', quality=quality, optimize=True)
    payload = buffer.getvalue()
    return {'ref': ref, 'width': labelled.width, 'height': labelled.height, 'source_width': source_size[0], 'source_height': source_size[1], 'rendering_version': CREDIT_IMAGE_RENDER_VERSION, 'data_url': 'data:image/jpeg;base64,' + base64.b64encode(payload).decode('ascii')}

def encode_rollout_visuals(images: list[Any] | None) -> list[dict[str, Any]]:
    pass
    encoded: list[dict[str, Any]] = []
    for index, raw in enumerate(images or [], 1):
        if not isinstance(raw, Image.Image):
            continue
        encoded.append(_encode_credit_image(raw, ref=f'img_{index}', label=f'CANDIDATE img_{index}'))
    return encoded

@lru_cache(maxsize=16)
def _encoded_reference_image(path_string: str, ref: str) -> dict[str, Any]:
    path = Path(path_string).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    return _encode_credit_image(image, ref=ref, label=f'REFERENCE {ref}')

@lru_cache(maxsize=32)
def _load_reference_bundle(path_string: str) -> dict[str, Any]:
    packed = re.fullmatch('(.+\\.jsonl)#byte=(\\d+),(\\d+)', path_string)
    path = Path(packed.group(1) if packed else path_string).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if packed:
        offset, length = (int(packed.group(2)), int(packed.group(3)))
        if length <= 0 or offset + length > path.stat().st_size:
            raise ValueError('reference JSONL byte range is invalid')
        with path.open('rb') as handle:
            handle.seek(offset)
            payload = handle.read(length)
        if len(payload) != length:
            raise ValueError('reference JSONL record was truncated')
    else:
        payload = path.read_bytes()
    trajectory = json.loads(payload)
    messages = trajectory.get('messages') or []
    visuals = []
    for item in trajectory.get('image_registry') or []:
        if not isinstance(item, dict):
            continue
        image_path = str(item.get('path') or '')
        ref = str(item.get('ref') or f'img_{len(visuals) + 1}')
        if not image_path:
            continue
        visuals.append(_encoded_reference_image(image_path, ref))
    return {'path': f'{path}#byte={offset},{length}' if packed else str(path), 'messages': copy.deepcopy(messages), 'visuals': copy.deepcopy(visuals), 'final_answer': str(trajectory.get('final_answer') or ''), 'turn_count': len(trajectory.get('turns') or [])}

def _credit_schema() -> dict[str, Any]:
    schema = {'schema_version': credit_schema_version(), 'candidate_overall_summary': 'brief causal account', 'events': [{'span_id': 'candidate Txx.Ay ID', 'type': 'causal_error | recovery | re_derailment', 'severity': '0..1; zero for recovery', 'confidence': '0..1', 'error_category': 'short snake_case label or empty string', 'rationale': 'concise evidence-specific reason', 'reference_support': ['reference assistant-turn numbers, if useful']}], 'productive_turns': [{'span_id': 'candidate Txx.Ay ID', 'confidence': '0..1', 'rationale': 'valid evidence or useful progress'}], 'confidence': '0..1'}
    if causal_component_credit_enabled():
        schema['events'][0]['components'] = {name: {'status': 'error | productive | neutral', 'severity': '0..1; zero unless error', 'confidence': '0..1', 'rationale': 'component-specific causal explanation'} for name in ('cot', 'act')}
    return schema

def _credit_response_format(policy_spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    pass
    mode = os.getenv('NEOS_TURN_CREDIT_STRUCTURED_OUTPUT', 'json_schema').strip().lower()
    if mode in {'', '0', 'false', 'none', 'off'}:
        return None
    if mode == 'json_object':
        return {'type': 'json_object'}
    if mode != 'json_schema':
        raise ValueError('NEOS_TURN_CREDIT_STRUCTURED_OUTPUT must be json_schema, json_object, or off')
    span_ids = [str(item.get('span_id') or '') for item in policy_spans]
    if not span_ids or any((not value for value in span_ids)):
        raise ValueError('cannot build credit JSON schema without policy span IDs')
    span_id = {'type': 'string', 'enum': span_ids}
    event = {'type': 'object', 'additionalProperties': False, 'properties': {'span_id': span_id, 'type': {'type': 'string', 'enum': ['causal_error', 'recovery', 're_derailment']}, 'severity': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'error_category': {'type': 'string'}, 'rationale': {'type': 'string'}, 'reference_support': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['span_id', 'type', 'severity', 'confidence', 'error_category', 'rationale', 'reference_support']}
    if causal_component_credit_enabled():
        component = {'type': 'object', 'additionalProperties': False, 'properties': {'status': {'type': 'string', 'enum': ['error', 'productive', 'neutral']}, 'severity': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'rationale': {'type': 'string'}}, 'required': ['status', 'severity', 'confidence', 'rationale']}
        event['properties']['components'] = {'type': ['object', 'null'], 'additionalProperties': False, 'properties': {'cot': copy.deepcopy(component), 'act': copy.deepcopy(component)}, 'required': ['cot', 'act']}
        event['required'].append('components')
    productive = {'type': 'object', 'additionalProperties': False, 'properties': {'span_id': span_id, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}, 'rationale': {'type': 'string'}}, 'required': ['span_id', 'confidence', 'rationale']}
    schema = {'type': 'object', 'additionalProperties': False, 'properties': {'schema_version': {'type': 'string', 'enum': [credit_schema_version()]}, 'candidate_overall_summary': {'type': 'string'}, 'events': {'type': 'array', 'minItems': 1, 'maxItems': len(span_ids), 'items': event}, 'productive_turns': {'type': 'array', 'maxItems': len(span_ids), 'items': productive}, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1}}, 'required': ['schema_version', 'candidate_overall_summary', 'events', 'productive_turns', 'confidence']}
    return {'type': 'json_schema', 'json_schema': {'name': 'neos_failed_turn_credit', 'strict': True, 'schema': schema}}

def _artifact_ref(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    if value.get('type') == 'artifact':
        artifact = value.get('artifact') or {}
        return str(artifact.get('ref') or '') or None
    url = value.get('url')
    if isinstance(url, dict):
        return _artifact_ref(url)
    return None

def _append_inline_trajectory(content: list[dict[str, Any]], *, label: str, events: list[dict[str, Any]], visuals: list[dict[str, Any]]) -> None:
    visual_map = {str(item.get('ref') or ''): item for item in visuals}
    attached: set[str] = set()
    content.append({'type': 'text', 'text': f'\n===== {label} TRAJECTORY START ====='})
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        role = str(event.get('role') or 'unknown')
        span_id = str(event.get('span_id') or '')
        heading = f'{label} event {index + 1} role={role}'
        if span_id:
            heading += f' span_id={span_id}'
        if role == 'tool' and event.get('observation_delivered_to_policy') is False:
            payload = {'role': 'tool', 'turn': event.get('turn'), 'name': event.get('name'), 'observation_delivered_to_policy': False, 'content': 'Tool execution completed, but no response was inserted into the policy context before termination. No result or pixels were available to the policy.'}
            content.append({'type': 'text', 'text': heading + '\n' + json.dumps(payload, separators=(',', ':'))})
            continue
        payload = copy.deepcopy(event)
        component_spans = payload.pop('component_spans', None)
        payload.pop('component_span_version', None)
        if causal_component_credit_enabled() and isinstance(component_spans, dict):
            payload['component_token_presence'] = {name: int(part.get('end', 0)) > int(part.get('start', 0)) for name, part in component_spans.items()}
        image_refs: list[str] = []
        explicit_refs = payload.pop('image_refs', None) or payload.pop('output_refs', None) or []
        if isinstance(explicit_refs, str):
            explicit_refs = [explicit_refs]
        image_refs.extend((str(ref) for ref in explicit_refs if str(ref)))
        raw_content = payload.get('content')
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'text':
                    text_parts.append(str(part.get('text') or ''))
                elif part.get('type') in {'image_url', 'input_image'}:
                    ref = _artifact_ref(part.get('image_url') or part)
                    if ref:
                        image_refs.append(ref)
            payload['content'] = '\n'.join(text_parts)
        from neos.workflow.visual_results import reused_refs_from_observation
        reused_refs = list(dict.fromkeys(list(payload.get('reused_refs') or []) + (reused_refs_from_observation(payload.get('content')) if role == 'tool' else [])))
        if reused_refs:
            known_at_event = attached | set(image_refs)
            unknown = set(reused_refs) - (known_at_event & set(visual_map))
            if unknown:
                raise ValueError(f'{label} reuses images before their first observation: {sorted(unknown)}')
            payload['reused_refs'] = reused_refs
        content.append({'type': 'text', 'text': heading + '\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':'))})
        for ref in dict.fromkeys(image_refs):
            visual = visual_map.get(ref)
            if visual is None:
                raise ValueError(f'{label} trajectory references missing visual {ref}')
            if ref in attached:
                content.append({'type': 'text', 'text': f'{label} {ref} reuses its earlier observation; pixels are not duplicated.'})
                continue
            content.extend([{'type': 'text', 'text': f'{label} {ref} visible in the preceding event'}, {'type': 'image_url', 'image_url': {'url': visual['data_url']}}])
            attached.add(ref)
    unattached = sorted(set(visual_map) - attached)
    if unattached:
        raise ValueError(f'{label} trajectory has unattached visuals: {unattached}')
    content.append({'type': 'text', 'text': f'===== {label} TRAJECTORY END =====\n'})

def _build_credit_messages(*, question: str, ground_truth: str, reference: dict[str, Any], failed_trace: list[dict[str, Any]], failed_visuals: list[dict[str, Any]], policy_spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    del question, ground_truth
    span_ids = [str(item.get('span_id') or '') for item in policy_spans]
    if not span_ids or any((not value for value in span_ids)):
        raise ValueError('candidate policy spans are missing stable span IDs')
    if len(set(span_ids)) != len(span_ids):
        raise ValueError('candidate policy span IDs are not unique')
    content: list[dict[str, Any]] = []
    _append_inline_trajectory(content, label='VERIFIED_REFERENCE', events=reference['messages'], visuals=reference['visuals'])
    _append_inline_trajectory(content, label='FAILED_CANDIDATE', events=failed_trace, visuals=failed_visuals)
    content.append({'type': 'text', 'text': 'Candidate span IDs, in chronological order: ' + json.dumps(span_ids, ensure_ascii=False, separators=(',', ':')) + '\nReturn strict JSON only using this compact schema once:\n' + json.dumps(_credit_schema(), ensure_ascii=False, separators=(',', ':'))})
    return [{'role': 'system', 'content': CREDIT_SYSTEM_PROMPT + (CAUSAL_COMPONENT_RULES if causal_component_credit_enabled() else '')}, {'role': 'user', 'content': content}]

def _response_payload(value: Any) -> dict[str, Any]:
    if hasattr(value, 'model_dump'):
        payload = value.model_dump(exclude_none=False)
        return payload if isinstance(payload, dict) else {}
    return value if isinstance(value, dict) else {}

def _usage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get('usage') or {}
    completion = usage.get('completion_tokens_details') or {}
    prompt = usage.get('prompt_tokens_details') or {}
    return {'input_tokens': int(usage.get('prompt_tokens') or 0), 'output_tokens': int(usage.get('completion_tokens') or 0), 'reasoning_tokens': int(completion.get('reasoning_tokens') or 0), 'cached_input_tokens': int(prompt.get('cached_tokens') or 0), 'image_input_tokens': int(prompt.get('image_tokens') or 0), 'total_tokens': int(usage.get('total_tokens') or 0)}

def _stream_qwen_credit(messages: list[dict[str, Any]], model: str, *, max_tokens: int | None=None, response_format: dict[str, Any] | None=None) -> dict[str, Any]:
    pass
    if max_tokens is None:
        max_tokens = int(os.getenv('NEOS_TURN_CREDIT_MAX_COMPLETION_TOKENS', '16384'))
    max_tokens = max(1024, int(max_tokens))
    effort = os.getenv('NEOS_TURN_CREDIT_REASONING_EFFORT', 'xhigh')
    attempts = max(1, int(os.getenv('NEOS_TURN_CREDIT_API_ATTEMPTS', '1')))
    attempt_records = []
    for attempt in range(1, attempts + 1):
        control = current_control()
        if control is not None:
            control.check()
        request_kwargs = dict(model=model, messages=messages, reasoning_effort=effort, max_completion_tokens=max_tokens, temperature=0.1, stream=True, stream_options={'include_usage': True})
        if response_format is not None:
            request_kwargs['response_format'] = response_format
        transport = run_credit_worker({'kwargs': request_kwargs, 'auth': {'api_key': secrets.DASHSCOPE_API_KEY, 'base_url': secrets.endpoint(secrets.DASHSCOPE_ENDPOINT, 'NEOS_DASHSCOPE_ENDPOINT'), 'headers': secrets.request_headers('DASHSCOPE')}})
        content = transport.get('content', '').strip()
        reasoning = transport.get('reasoning_content', '').strip()
        error = transport.get('transport_error')
        parsed = _json_object(content)
        if error:
            parse_error = ('qwen credit judge stream failed after returning a partial response: ' if content or reasoning else 'qwen credit request failed: ') + error
        elif not content:
            parse_error = 'qwen credit judge returned empty visible content'
        elif parsed is None:
            parse_error = 'qwen credit judge returned non-JSON content'
        else:
            parse_error = None
        result = {**transport, 'content': content, 'reasoning_content': reasoning, 'parsed': parsed, 'parse_error': parse_error, 'usage': _usage_summary({'usage': transport.get('usage') or {}}), 'attempt': attempt, 'attempt_errors': attempt_records, 'elapsed_seconds': transport.get('elapsed_seconds', transport.get('supervisor_elapsed_seconds', 0)), 'reasoning_effort': effort, 'max_completion_tokens': max_tokens, 'structured_output': response_format.get('type') if response_format else 'off', 'stream': True}
        if error:
            attempt_records.append({'attempt': attempt, 'error': error, 'content': content, 'reasoning_content': reasoning, 'usage': result['usage'], 'usage_complete': transport.get('usage_complete', False), 'response_id': transport.get('response_id'), 'finish_reason': transport.get('finish_reason'), 'elapsed_seconds': result['elapsed_seconds'], 'timeout_kind': transport.get('timeout_kind')})
            if attempt < attempts:
                if control is not None:
                    control.check()
                time.sleep(min(2 ** (attempt - 1), 10))
                continue
        return result
    raise RuntimeError('credit HTTP attempt loop produced no result')

def _component_ranges(span: dict[str, Any]) -> dict[str, tuple[int, int]]:
    pass
    components = span.get('component_spans')
    if not isinstance(components, dict) or set(components) != {'cot', 'act'}:
        raise ValueError(f"missing CoT/Act token partition for {span.get('span_id')}")
    result = {}
    for name in ('cot', 'act'):
        part = components[name]
        if not isinstance(part, dict) or 'start' not in part or 'end' not in part:
            raise ValueError(f'invalid {name} token partition')
        start, end = (int(part['start']), int(part['end']))
        if start != part['start'] or end != part['end'] or end < start:
            raise ValueError(f'invalid {name} token bounds')
        result[name] = (start, end)
    if result['cot'][0] != int(span['start']) or result['cot'][1] != result['act'][0] or result['act'][1] != int(span['end']):
        raise ValueError('CoT/Act ranges must partition the whole generated turn')
    return result

def _validate_component_credit(raw: Any, span: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != {'cot', 'act'}:
        raise ValueError('causal_error/re_derailment requires both cot and act ratings')
    ranges = _component_ranges(span)
    result = {}
    for name in ('cot', 'act'):
        item = raw[name]
        if not isinstance(item, dict) or item.get('status') not in {'error', 'productive', 'neutral'}:
            raise ValueError(f'invalid {name} credit status')
        if ranges[name][0] == ranges[name][1] and item['status'] != 'neutral':
            raise ValueError(f'absent {name} component must be neutral')
        result[name] = {'status': item['status'], 'severity': _float01(item.get('severity')) if item['status'] == 'error' else 0.0, 'confidence': _float01(item.get('confidence')), 'rationale': str(item.get('rationale') or '')[:1200]}
    if not any((item['status'] == 'error' for item in result.values())):
        raise ValueError('a causal event needs at least one erroneous CoT/Act component')
    return result

def _validate_credit_payload(payload: dict[str, Any], policy_spans: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = [str(item['span_id']) for item in policy_spans]
    expected = set(expected_ids)
    spans_by_id = {str(item['span_id']): item for item in policy_spans}
    split_components = causal_component_credit_enabled()
    if split_components and payload.get('schema_version') != credit_schema_version():
        raise ValueError('credit response uses the wrong component-credit schema version')
    raw_events = payload.get('events') or []
    raw_productive = payload.get('productive_turns') or []
    if not isinstance(raw_events, list) or not isinstance(raw_productive, list):
        raise ValueError('credit judge events/productive_turns must be lists')
    event_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_events:
        if not isinstance(raw, dict):
            raise ValueError('credit event is not an object')
        span_id = str(raw.get('span_id') or '')
        event_type = str(raw.get('type') or '')
        if span_id not in expected or span_id in event_by_id:
            raise ValueError(f'invalid or duplicate credit event span_id {span_id!r}')
        if event_type not in {'causal_error', 'recovery', 're_derailment'}:
            raise ValueError(f'invalid credit event type {event_type!r}')
        event_by_id[span_id] = {'span_id': span_id, 'status': event_type, 'severity': 0.0 if event_type == 'recovery' else _float01(raw.get('severity')), 'confidence': _float01(raw.get('confidence')), 'caused_wrong_branch': event_type in CAUSAL_TURN_STATUSES, 'error_category': str(raw.get('error_category') or '')[:120], 'rationale': str(raw.get('rationale') or '')[:1200], 'reference_support': [str(value)[:120] for value in (raw.get('reference_support') or [])[:12]]}
        if split_components:
            if event_type in CAUSAL_TURN_STATUSES:
                event_by_id[span_id]['components'] = _validate_component_credit(raw.get('components'), spans_by_id[span_id])
            elif raw.get('components') is not None:
                raise ValueError('recovery must stay turn-level: components must be null')
    productive_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_productive:
        if isinstance(raw, str):
            raw = {'span_id': raw, 'confidence': 1.0, 'rationale': ''}
        if not isinstance(raw, dict):
            raise ValueError('productive turn is not an object')
        span_id = str(raw.get('span_id') or '')
        if span_id not in expected or span_id in productive_by_id or span_id in event_by_id:
            raise ValueError(f'invalid or duplicate productive span_id {span_id!r}')
        productive_by_id[span_id] = {'confidence': _float01(raw.get('confidence')), 'rationale': str(raw.get('rationale') or '')[:1200]}
    turns = []
    active_error: dict[str, Any] | None = None
    downstream_distance = 0
    causal_count = 0
    for span_id in expected_ids:
        event = event_by_id.get(span_id)
        if event is not None:
            status = event['status']
            if status in CAUSAL_TURN_STATUSES:
                causal_count += 1
                active_error = event
                downstream_distance = 0
            elif status == 'recovery':
                active_error = None
                downstream_distance = 0
            turns.append(dict(event, downstream_distance=0))
            continue
        if active_error is not None:
            downstream_distance += 1
        productive = productive_by_id.get(span_id)
        if active_error is not None and productive is None:
            turns.append({'span_id': span_id, 'status': 'downstream_consequence', 'severity': active_error['severity'], 'confidence': active_error['confidence'], 'caused_wrong_branch': False, 'error_category': active_error['error_category'], 'rationale': 'deterministic downstream consequence of ' + active_error['span_id'], 'reference_support': active_error['reference_support'], 'downstream_distance': downstream_distance})
            continue
        turns.append({'span_id': span_id, 'status': 'productive' if productive is not None else 'neutral', 'severity': 0.0, 'confidence': productive['confidence'] if productive else 0.0, 'caused_wrong_branch': False, 'error_category': '', 'rationale': productive['rationale'] if productive else '', 'reference_support': [], 'downstream_distance': 0})
    if causal_count == 0:
        raise ValueError('credit judge identified no causal error turn')
    return {'schema_version': credit_schema_version(), 'candidate_overall_summary': str(payload.get('candidate_overall_summary') or '')[:2400], 'reference_is_valid_comparator': True, 'answer_judge_disagreement': False, 'turns': turns, 'failure_events': [copy.deepcopy(event) for event in event_by_id.values() if event['status'] in CAUSAL_TURN_STATUSES], 'confidence': _float01(payload.get('confidence')), 'causal_turn_count': causal_count}

class _DistributedCreditLimiter:
    pass

    def __init__(self, limit: int):
        self.limit = int(limit)
        self.leases: dict[str, float] = {}
        self.cancelled: dict[str, float] = {}

    async def acquire(self, token: str, timeout: float, lease_seconds: float) -> bool:
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            now = time.monotonic()
            self.cancelled = {key: expiry for key, expiry in self.cancelled.items() if expiry > now}
            if token in self.cancelled:
                return False
            self.leases = {key: expiry for key, expiry in self.leases.items() if expiry > now}
            if token in self.leases:
                self.leases[token] = now + float(lease_seconds)
                return True
            if len(self.leases) < self.limit:
                self.leases[token] = now + float(lease_seconds)
                return True
            await asyncio.sleep(0.1)
        return False

    def release(self, token: str) -> None:
        self.leases.pop(token, None)
        self.cancelled[token] = time.monotonic() + 3600

async def _distributed_slot() -> tuple[Any | None, str | None]:
    limit = max(1, int(os.getenv('NEOS_TURN_CREDIT_MAX_CONCURRENCY', '128')))
    try:
        import ray
        if not ray.is_initialized():
            return (None, None)
        job = re.sub('[^A-Za-z0-9_-]', '-', os.getenv('RL_JOB_ID', 'default'))[:80]
        name = f'neos-turn-credit-limiter-{job}'
        actor_class = ray.remote(max_concurrency=max(256, limit * 2))(_DistributedCreditLimiter)
        actor = actor_class.options(name=name, namespace='neos', lifetime='detached', get_if_exists=True).remote(limit)
        token = uuid.uuid4().hex
        timeout = float(os.getenv('NEOS_TURN_CREDIT_SLOT_TIMEOUT', '1800'))
        lease = float(os.getenv('NEOS_TURN_CREDIT_SLOT_LEASE_SECONDS', '1800'))
        control = current_control()
        if control is not None:
            control.check()
            timeout = min(timeout, control.remaining())
        try:
            acquired = await asyncio.wait_for(actor.acquire.remote(token, timeout, lease), timeout=timeout + 5)
        except BaseException:
            await _release_distributed_slot(actor, token)
            raise
        if not acquired:
            raise TimeoutError(f'timed out waiting for distributed credit slot ({limit})')
        return (actor, token)
    except ImportError:
        return (None, None)

async def _release_distributed_slot(actor: Any | None, token: str | None) -> None:
    if actor is None or token is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(actor.release.remote(token)), timeout=10)
    except Exception:
        pass

def _append_audit(row: dict[str, Any]) -> str | None:
    root_value = os.getenv('NEOS_TURN_CREDIT_AUDIT_ROOT', '').strip()
    if not root_value:
        return None
    root = Path(root_value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    host = re.sub('[^A-Za-z0-9_.-]', '_', socket.gethostname())
    path = root / f'credit-{host}-pid{os.getpid()}.jsonl'
    payload = json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 384)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, payload.encode('utf-8'))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    return str(path)

async def assign_failed_trajectory_credit(**kwargs) -> dict[str, Any]:
    control = current_control() or CreditControl()
    try:
        return await guarded_credit_call(_assign_failed_trajectory_credit_impl, kwargs, control)
    except (CreditDeadlineError, asyncio.CancelledError):
        _append_audit({'schema_version': credit_schema_version(), 'prompt_version': credit_prompt_version(), 'created_at_unix': time.time(), 'qid': kwargs.get('qid'), 'rollout_request_id': kwargs.get('rollout_request_id'), 'model': os.getenv('NEOS_TURN_CREDIT_MODEL', DEFAULT_CREDIT_MODEL), 'applied': False, 'reason': 'credit_trajectory_deadline', 'last_provider': control.last_response, 'wall_seconds': round(time.monotonic() - (control.started or time.monotonic()), 3)})
        raise

async def _assign_failed_trajectory_credit_impl(*, question: str, ground_truth: str, failed_trace: list[dict[str, Any]], failed_visuals: list[dict[str, Any]], policy_spans: list[dict[str, Any]], reference_path: str, qid: str, rollout_request_id: str) -> dict[str, Any]:
    pass
    model = os.getenv('NEOS_TURN_CREDIT_MODEL', DEFAULT_CREDIT_MODEL)
    reference = _load_reference_bundle(reference_path)
    messages = _build_credit_messages(question=question, ground_truth=ground_truth, reference=reference, failed_trace=failed_trace, failed_visuals=failed_visuals, policy_spans=policy_spans)
    response_format = _credit_response_format(policy_spans)
    primary_max_tokens = max(1024, int(os.getenv('NEOS_TURN_CREDIT_MAX_COMPLETION_TOKENS', '16384')))
    repair_max_tokens = max(primary_max_tokens, int(os.getenv('NEOS_TURN_CREDIT_REPAIR_MAX_COMPLETION_TOKENS', str(primary_max_tokens * 2))))
    slot_actor, slot_token = await _distributed_slot()
    started = time.monotonic()
    try:
        validation_attempts = max(1, int(os.getenv('NEOS_TURN_CREDIT_SCHEMA_ATTEMPTS', '2')))
        validation_errors = []
        response = None
        validated = None
        request_messages = messages
        for schema_attempt in range(1, validation_attempts + 1):
            control = current_control()
            if control is not None:
                control.check()
            attempt_response = None
            try:
                attempt_response = await asyncio.to_thread(_stream_qwen_credit, request_messages, model, max_tokens=primary_max_tokens if schema_attempt == 1 else repair_max_tokens, response_format=response_format)
                response = attempt_response
                if response.get('parse_error'):
                    raise ValueError(str(response['parse_error']))
                if not isinstance(response.get('parsed'), dict):
                    raise ValueError('credit judge returned no JSON object')
                validated = _validate_credit_payload(response['parsed'], policy_spans)
                break
            except Exception as exc:
                if attempt_response is not None:
                    response = attempt_response
                validation_errors.append({'schema_attempt': schema_attempt, 'error': f'{type(exc).__name__}: {exc}'[:4000], 'content': attempt_response.get('content', '') if attempt_response else '', 'reasoning_content': attempt_response.get('reasoning_content', '') if attempt_response else '', 'usage': attempt_response.get('usage', {}) if attempt_response else {}, 'response_id': attempt_response.get('response_id') if attempt_response else None, 'finish_reason': attempt_response.get('finish_reason') if attempt_response else None, 'api_attempt': attempt_response.get('attempt') if attempt_response else None, 'api_attempt_errors': attempt_response.get('attempt_errors', []) if attempt_response else [], 'max_completion_tokens': attempt_response.get('max_completion_tokens') if attempt_response else primary_max_tokens if schema_attempt == 1 else repair_max_tokens, 'structured_output': attempt_response.get('structured_output') if attempt_response else response_format.get('type') if response_format else 'off', 'timeout_kind': attempt_response.get('timeout_kind') if attempt_response else None})
                if isinstance(exc, CreditDeadlineError) or (control is not None and control.cancelled.is_set()):
                    break
                if schema_attempt < validation_attempts and attempt_response is not None and attempt_response.get('transport_error'):
                    request_messages = messages
                    error_text = str(attempt_response['transport_error']).lower()
                    delay = 15 if '429' in error_text or 'too many requests' in error_text else 5
                    if control is not None:
                        delay = min(delay, max(0, control.remaining()))
                    await asyncio.sleep(delay)
                elif schema_attempt < validation_attempts and attempt_response is not None and (attempt_response.get('content') or attempt_response.get('reasoning_content')):
                    previous_assistant = {'role': 'assistant', 'content': attempt_response.get('content', '')}
                    if attempt_response.get('reasoning_content'):
                        previous_assistant['reasoning_content'] = attempt_response['reasoning_content']
                    request_messages = [*messages, previous_assistant, {'role': 'user', 'content': f'Your preceding raw response failed deterministic validation: {type(exc).__name__}: {exc}. Continue from the preceding reasoning instead of recomputing the trajectory comparison. Immediately emit one complete corrected JSON object. Use only supplied candidate span IDs, never duplicate an ID, and keep the causal analysis unchanged where possible. Return only significant events and productive turns; omitted turns remain neutral.'}]
                    if not previous_assistant['content']:
                        request_messages = [*messages, {'role': 'user', 'content': 'The previous attempt produced reasoning but no JSON. This is its incomplete reasoning, not additional evidence:\n' + str(attempt_response.get('reasoning_content', '')) + '\nValidation error: ' + str(exc) + '\nImmediately return a complete corrected JSON object using the supplied span IDs.'}]
                else:
                    request_messages = messages
        if response is None or validated is None:
            failure_audit = {'schema_version': credit_schema_version(), 'prompt_version': credit_prompt_version(), 'created_at_unix': time.time(), 'qid': qid, 'rollout_request_id': rollout_request_id, 'model': model, 'reference_path': reference['path'], 'applied': False, 'reason': 'credit_schema_validation_exhausted', 'validation_errors': validation_errors, 'last_provider': {key: value for key, value in response.items() if key not in {'parsed'}} if response is not None else None, 'structured_output': response_format.get('type') if response_format else 'off', 'primary_max_completion_tokens': primary_max_tokens, 'repair_max_completion_tokens': repair_max_tokens, 'wall_seconds': round(time.monotonic() - started, 3)}
            _append_audit(failure_audit)
            if control is not None and (control.cancelled.is_set() or control.remaining() <= 0):
                raise CreditDeadlineError('credit trajectory deadline exceeded')
            raise ValueError('credit judge schema validation exhausted')
        applied = bool(validated['reference_is_valid_comparator'] and (not validated['answer_judge_disagreement']))
        reason = 'credit_ready' if applied else 'credit_judge_declined_comparison'
        audit = {'schema_version': credit_schema_version(), 'prompt_version': credit_prompt_version(), 'created_at_unix': time.time(), 'qid': qid, 'rollout_request_id': rollout_request_id, 'model': model, 'reference_path': reference['path'], 'applied': applied, 'reason': reason, 'validated': validated, 'provider': {key: value for key, value in response.items() if key not in {'parsed'}}, 'schema_attempts': len(validation_errors) + 1, 'prior_schema_errors': validation_errors, 'structured_output': response_format.get('type') if response_format else 'off', 'primary_max_completion_tokens': primary_max_tokens, 'repair_max_completion_tokens': repair_max_tokens, 'wall_seconds': round(time.monotonic() - started, 3)}
        audit_path = _append_audit(audit)
        return {'schema_version': credit_schema_version(), 'eligible': True, 'applied': applied, 'reason': reason, 'model': model, 'reference_turn_count': reference['turn_count'], 'reference_visual_count': len(reference['visuals']), 'candidate_visual_count': len(failed_visuals), 'turns': validated['turns'] if applied else [], 'failure_events': validated['failure_events'] if applied else [], 'confidence': validated['confidence'], 'usage': response['usage'], 'api_attempt': response['attempt'], 'schema_attempts': len(validation_errors) + 1, 'elapsed_seconds': response['elapsed_seconds'], 'audit_path': audit_path}
    finally:
        await _release_distributed_slot(slot_actor, slot_token)

def _unwrap_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    data = getattr(value, 'data', None)
    if isinstance(data, dict):
        return data
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            unwrapped = item()
        except (TypeError, ValueError):
            return {}
        return unwrapped if isinstance(unwrapped, dict) else {}
    return {}

async def _assign_mixed_group_credits_async(jobs: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any] | BaseException]]:

    async def run_one(row_index: int, kwargs: dict[str, Any]):
        try:
            return (row_index, await assign_failed_trajectory_credit(**kwargs))
        except BaseException as exc:
            return (row_index, exc)
    return await asyncio.gather(*(run_one(row_index, kwargs) for row_index, kwargs in jobs))

def assign_complete_group_turn_credits(*, batch_keys: list[str], rm_scores: torch.Tensor, extra_fields: list[Any] | np.ndarray | None, num_repeat: int) -> tuple[list[Any] | None, dict[str, float]]:
    pass
    metrics = {'turn_credit/groups_total': 0.0, 'turn_credit/groups_all_correct_skipped': 0.0, 'turn_credit/groups_all_failed_skipped': 0.0, 'turn_credit/groups_mixed': 0.0, 'turn_credit/judge_calls': 0.0, 'turn_credit/judge_applied': 0.0, 'turn_credit/judge_fail_open': 0.0, 'turn_credit/no_verified_reference': 0.0}
    if extra_fields is None:
        return (None, metrics)
    normalized_fields = [_unwrap_mapping(value) for value in list(extra_fields)]
    if len(normalized_fields) != len(batch_keys) or rm_scores.shape[0] != len(batch_keys):
        raise ValueError('group turn-credit rows, keys, and rewards are misaligned')
    score_values = rm_scores.sum(dim=-1).detach().float().cpu().tolist()
    final_sessions: dict[str, tuple[int, int, str]] = {}
    for row_index, key in enumerate(batch_keys):
        parts = str(key).rsplit('_', 2)
        if len(parts) != 3:
            raise ValueError(f'unexpected trajectory key format: {key}')
        uid, session_id, output_index = (parts[0], parts[1], int(parts[2]))
        session_key = f'{uid}_{session_id}'
        if session_key not in final_sessions or final_sessions[session_key][0] < output_index:
            final_sessions[session_key] = (output_index, row_index, uid)
    groups: dict[str, list[int]] = defaultdict(list)
    for _, row_index, uid in final_sessions.values():
        groups[uid].append(row_index)
    metrics['turn_credit/groups_total'] = float(len(groups))
    jobs: list[tuple[int, dict[str, Any]]] = []

    def set_credit(row_index: int, value: dict[str, Any]) -> None:
        fields = normalized_fields[row_index]
        reward_info = _unwrap_mapping(fields.get('reward_extra_info'))
        if not reward_info:
            reward_info = {}
            fields['reward_extra_info'] = reward_info
        reward_info['turn_credit'] = value
        if os.getenv('NEOS_COEV_ROOT') and value.get('applied'):
            from neos.workflow.coevolution import observe_credit
            observe_credit(fields, value)
    for row_indices in groups.values():
        if len(row_indices) != int(num_repeat):
            for row_index in row_indices:
                set_credit(row_index, {'schema_version': credit_schema_version(), 'eligible': False, 'applied': False, 'reason': 'incomplete_group_no_credit'})
            continue
        outcomes = [score_values[row_index] > 0.5 for row_index in row_indices]
        if all(outcomes):
            metrics['turn_credit/groups_all_correct_skipped'] += 1.0
            continue
        if not any(outcomes):
            metrics['turn_credit/groups_all_failed_skipped'] += 1.0
            for row_index in row_indices:
                fields = normalized_fields[row_index]
                set_credit(row_index, {'schema_version': credit_schema_version(), 'eligible': bool(fields.get('turn_credit_eligible')), 'applied': False, 'reason': 'homogeneous_all_failed_no_update'})
            continue
        metrics['turn_credit/groups_mixed'] += 1.0
        for row_index, correct in zip(row_indices, outcomes, strict=True):
            if correct:
                continue
            fields = normalized_fields[row_index]
            reward_info = _unwrap_mapping(fields.get('reward_extra_info'))
            eligible = bool(fields.get('turn_credit_eligible'))
            if bool(reward_info.get('judge_parse_failed')):
                set_credit(row_index, {'schema_version': credit_schema_version(), 'eligible': eligible, 'applied': False, 'reason': 'answer_judge_parse_failed'})
                continue
            if not eligible:
                metrics['turn_credit/no_verified_reference'] += 1.0
                set_credit(row_index, {'schema_version': credit_schema_version(), 'eligible': False, 'applied': False, 'reason': 'no_verified_correct_reference'})
                continue
            if not bool(fields.get('turn_credit_capture')):
                set_credit(row_index, {'schema_version': credit_schema_version(), 'eligible': True, 'applied': False, 'reason': 'turn_credit_trace_not_captured'})
                continue
            trace = fields.get('turn_credit_trace') or []
            spans = fields.get('policy_turn_spans') or []
            reference_path = str(fields.get('reference_trajectory_path') or '')
            if not trace or not spans or (not reference_path):
                set_credit(row_index, {'schema_version': credit_schema_version(), 'eligible': True, 'applied': False, 'reason': 'incomplete_credit_inputs'})
                continue
            jobs.append((row_index, {'question': str(fields.get('question') or ''), 'ground_truth': str(fields.get('ground_truth') or ''), 'failed_trace': trace, 'failed_visuals': fields.get('_turn_credit_visuals') or [], 'policy_spans': spans, 'reference_path': reference_path, 'qid': str(fields.get('qid') or ''), 'rollout_request_id': str(fields.get('rollout_request_id') or '')}))
    metrics['turn_credit/judge_calls'] = float(len(jobs))
    if jobs:
        from neos.workflow.credit_prefetch import enabled, get_manager
        try:
            if enabled():
                results, prefetch_metrics = get_manager().resolve(jobs, batch_keys)
                metrics.update(prefetch_metrics)
            else:
                results = asyncio.run(_assign_mixed_group_credits_async(jobs))
        except RuntimeError as exc:
            if 'asyncio.run() cannot be called' not in str(exc):
                raise
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                results = executor.submit(lambda: asyncio.run(_assign_mixed_group_credits_async(jobs))).result()
        for row_index, result in results:
            if isinstance(result, BaseException):
                metrics['turn_credit/judge_fail_open'] += 1.0
                set_credit(row_index, {'schema_version': credit_schema_version(), 'eligible': True, 'applied': False, 'reason': 'turn_credit_fail_open', 'error': f'{type(result).__name__}: {result}'[:4000]})
            else:
                set_credit(row_index, result)
                metrics['turn_credit/judge_applied'] += float(bool(result.get('applied')))
    for fields in normalized_fields:
        fields.pop('_turn_credit_visuals', None)
        fields.pop('turn_credit_trace', None)
        fields.pop('turn_credit_capture', None)
    return (normalized_fields, metrics)

def reshape_failed_turn_advantages(advantages: torch.Tensor, response_mask: torch.Tensor, extra_fields: list[Any] | np.ndarray | None) -> tuple[torch.Tensor, dict[str, float]]:
    pass
    metrics = {'turn_credit/eligible_failed_rows': 0.0, 'turn_credit/applied_failed_rows': 0.0, 'turn_credit/reshaped_negative_rows': 0.0, 'turn_credit/causal_spans': 0.0, 'turn_credit/tokens_covered': 0.0, 'turn_credit/positive_tokens': 0.0, 'turn_credit/negative_tokens': 0.0, 'turn_credit/downstream_tokens': 0.0, 'turn_credit/component_split_spans': 0.0, 'turn_credit/cot_error_spans': 0.0, 'turn_credit/act_error_spans': 0.0, 'turn_credit/cot_protected_tokens': 0.0, 'turn_credit/act_protected_tokens': 0.0}
    if extra_fields is None:
        return (advantages, metrics)
    result = advantages.clone()
    rows = list(extra_fields)
    if len(rows) != result.shape[0]:
        raise ValueError(f'turn-credit extra_fields rows={len(rows)} != advantages rows={result.shape[0]}')
    for row_index, raw_fields in enumerate(rows):
        fields = _unwrap_mapping(raw_fields)
        reward_info = _unwrap_mapping(fields.get('reward_extra_info'))
        credit = _unwrap_mapping(reward_info.get('turn_credit'))
        if float(reward_info.get('acc') or 0.0) != 0.0:
            continue
        if credit.get('eligible'):
            metrics['turn_credit/eligible_failed_rows'] += 1.0
        reason = re.sub('[^a-z0-9_]+', '_', str(credit.get('reason') or 'missing').strip().lower()).strip('_') or 'missing'
        reason_key = f'turn_credit/reason/{reason}'
        metrics[reason_key] = metrics.get(reason_key, 0.0) + 1.0
        if not credit.get('applied'):
            continue
        spans = fields.get('policy_turn_spans') or []
        ratings = {str(item.get('span_id') or ''): item for item in credit.get('turns') or [] if isinstance(item, dict)}
        if not spans or not ratings:
            continue
        mask = response_mask[row_index].bool()
        if not bool(mask.any()):
            continue
        base_scalar = float(result[row_index][mask][0].item())
        if base_scalar >= -1e-08:
            continue
        magnitude = abs(base_scalar)
        productive_scale = max(0.0, float(os.getenv('NEOS_TURN_CREDIT_PRODUCTIVE_REWARD', '0.10')))
        recovery_scale = max(productive_scale, float(os.getenv('NEOS_TURN_CREDIT_RECOVERY_REWARD', '0.30')))
        downstream_start = max(0.0, float(os.getenv('NEOS_TURN_CREDIT_DOWNSTREAM_START', '0.50')))
        downstream_decay = min(1.0, max(0.0, float(os.getenv('NEOS_TURN_CREDIT_DOWNSTREAM_DECAY', '0.65'))))
        max_negative_multiplier = max(1.0, float(os.getenv('NEOS_TURN_CREDIT_MAX_NEGATIVE_MULTIPLIER', '4.0')))
        minimum_net_negative = max(0.0, float(os.getenv('NEOS_TURN_CREDIT_MIN_NET_NEGATIVE', '0.02')))
        signed = torch.zeros_like(result[row_index])
        covered = torch.zeros_like(mask)
        downstream_mask = torch.zeros_like(mask)
        causal_spans = 0
        for span in spans:
            if not isinstance(span, dict):
                continue
            span_id = str(span.get('span_id') or '')
            rating = ratings.get(span_id)
            if rating is None:
                continue
            start = max(0, int(span.get('start') or 0))
            end = min(result.shape[1], int(span.get('end') or 0))
            if end <= start:
                continue
            token_mask = mask[start:end]
            if not bool(token_mask.any()):
                continue
            status = str(rating.get('status') or 'neutral')
            severity = _float01(rating.get('severity'))
            confidence = _float01(rating.get('confidence'))
            if status in CAUSAL_TURN_STATUSES and isinstance(rating.get('components'), dict):
                ranges = _component_ranges(span)
                components = rating['components']
                causal_spans += 1
                metrics['turn_credit/component_split_spans'] += 1.0
                for name in ('cot', 'act'):
                    part = components[name]
                    part_status = str(part['status'])
                    part_confidence = _float01(part.get('confidence'))
                    if part_status == 'error':
                        value = -magnitude * max(0.5, _float01(part.get('severity')) * part_confidence)
                        metrics[f'turn_credit/{name}_error_spans'] += 1.0
                    elif part_status == 'productive':
                        value = magnitude * productive_scale * part_confidence
                    elif part_status == 'neutral':
                        value = 0.0
                    else:
                        raise ValueError(f'invalid {name} component status: {part_status}')
                    left = max(start, ranges[name][0])
                    right = min(end, ranges[name][1])
                    if right <= left:
                        continue
                    component_mask = mask[left:right]
                    signed[left:right][component_mask] = value
                    covered[left:right] |= component_mask
                    if part_status != 'error':
                        metrics[f'turn_credit/{name}_protected_tokens'] += float(component_mask.sum().item())
                continue
            if status == 'causal_error':
                signed_value = -magnitude * max(0.5, severity * confidence)
                causal_spans += 1
            elif status == 're_derailment':
                signed_value = -magnitude * max(0.5, severity * confidence)
                causal_spans += 1
            elif status == 'downstream_consequence':
                distance = max(1, int(rating.get('downstream_distance') or 1))
                signed_value = -magnitude * max(0.25, severity * confidence) * downstream_start * downstream_decay ** (distance - 1)
            elif status == 'recovery':
                signed_value = magnitude * recovery_scale * confidence
            elif status == 'productive':
                signed_value = magnitude * productive_scale * confidence
            else:
                signed_value = 0.0
            segment_signed = signed[start:end]
            segment_signed[token_mask] = signed_value
            covered[start:end] |= token_mask
            if status == 'downstream_consequence':
                downstream_mask[start:end] |= token_mask
        if causal_spans == 0 or not bool(covered.any()):
            continue
        active = signed[mask]
        positive = active > 0
        negative = active < 0
        if not bool(negative.any()):
            continue
        min_total = minimum_net_negative * magnitude * active.numel()
        positive_total = active[positive].sum() if bool(positive.any()) else active.new_tensor(0.0)
        negative_total = -active[negative].sum()
        required_negative = positive_total + min_total
        if float(negative_total.item()) < float(required_negative.item()):
            scale = min(max_negative_multiplier, float((required_negative / negative_total.clamp_min(1e-08)).item()))
            active[negative] *= scale
            negative_total = -active[negative].sum()
        if float(negative_total.item()) < float((positive_total + min_total).item()) and bool(positive.any()):
            allowed_positive = max(0.0, float((negative_total - min_total).item()))
            if float(positive_total.item()) > 0:
                active[positive] *= allowed_positive / float(positive_total.item())
        active.clamp_(min=-max_negative_multiplier * magnitude, max=recovery_scale * magnitude)
        result[row_index][mask] = active
        metrics['turn_credit/applied_failed_rows'] += 1.0
        metrics['turn_credit/causal_spans'] += float(causal_spans)
        metrics['turn_credit/tokens_covered'] += float(covered.sum().item())
        metrics['turn_credit/positive_tokens'] += float((active > 0).sum().item())
        metrics['turn_credit/negative_tokens'] += float((active < 0).sum().item())
        metrics['turn_credit/downstream_tokens'] += float(downstream_mask.sum().item())
        metrics['turn_credit/reshaped_negative_rows'] += 1.0
    return (result, metrics)
