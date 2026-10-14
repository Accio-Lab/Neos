from __future__ import annotations
import contextlib
import contextvars
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any
from neos.workflow.trace_jsonl import jsonl_rows, trace_record_paths, TraceReadAudit
from neos.workflow.shared_state_io import read_json, atomic_json
GOOD_TURN_STATES = {'productive', 'recovery'}
_LOCK = threading.RLock()
_STORES: dict[str, Any] = {}
_MEI_CONTEXT = contextvars.ContextVar('neos_coev_mei', default=None)

@contextlib.contextmanager
def mei_call_context(fields):
    token = _MEI_CONTEXT.set(fields.get('coev_mei_model'))
    try:
        yield
    finally:
        _MEI_CONTEXT.reset(token)

def bound_mei_model():
    expected = _MEI_CONTEXT.get()
    state = mei_snapshot()
    policy = read_json(root() / 'policy.json', {})
    if not expected or state.get('model') != expected or policy.get('phase') != 'rollout':
        raise RuntimeError('Retired or unversioned MEI request; do not cross the rollout boundary')
    return expected

def parallel_map(executor, function, values):
    if root() is None:
        return list(executor.map(function, values))
    pending = [executor.submit(contextvars.copy_context().run, function, value) for value in values]
    return [future.result() for future in pending]

def root() -> Path | None:
    value = os.getenv('NEOS_COEV_ROOT', '').strip()
    return Path(value) if value else None

def rollout_system_prompt(max_assistant_turns: int) -> str:
    from neos.workflow.protocol import MAX_ASSISTANT_TURNS, NEOS_SYSTEM_PROMPT
    if int(max_assistant_turns) != MAX_ASSISTANT_TURNS:
        raise ValueError(f'Neos requires {MAX_ASSISTANT_TURNS} assistant turns')
    return NEOS_SYSTEM_PROMPT

def append_event(kind: str, value: dict) -> None:
    destination = root()
    if destination is None:
        return
    path = destination / kind / f'{socket.gethostname()}-pid{os.getpid()}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {'schema': 'neos-coevolution', 'at_unix': time.time(), **value}
    with _LOCK, path.open('a') as out:
        out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        out.flush()

def mei_snapshot() -> dict | None:
    destination = root()
    if destination is None:
        return None
    state = read_json(destination / 'serving.json')
    if not state or state.get('status') not in {'ready', 'waiting_policy'}:
        raise RuntimeError('Co-evolution MEI is not deployed/ready')
    return state

def rollout_context() -> dict:
    destination = root()
    if destination is None:
        return {}
    state = read_json(destination / 'policy.json')
    mei = mei_snapshot()
    if not state or state.get('phase') != 'rollout':
        raise RuntimeError('Formal rollout started outside the co-evolution rollout gate')
    if state.get('mei_model') != mei.get('model'):
        raise RuntimeError('Policy/MEI version mismatch at rollout admission')
    return {'coev_policy_step': int(state['step']), 'coev_mei_version': int(mei['version']), 'coev_mei_model': mei['model']}

def trace_ids(value: Any) -> set[str]:
    result = set()
    if isinstance(value, dict):
        if value.get('mei_trace_id'):
            result.add(str(value['mei_trace_id']))
        for v in value.values():
            result.update(trace_ids(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            result.update(trace_ids(v))
    return result

def selected_mei_links(fields: dict, *, correct: bool, credit: dict | None=None) -> list[dict]:
    pass
    if not fields.get('coev_mei_model'):
        return []
    positive_turns = set()
    if not correct:
        if not credit or not credit.get('applied'):
            return []
        spans = {str(value.get('span_id')): value for value in fields.get('policy_turn_spans') or []}
        for entry in credit.get('turns') or []:
            if entry.get('status') in GOOD_TURN_STATES:
                span = spans.get(str(entry.get('span_id')), {})
                if span.get('accepted') and span.get('action_kind') == 'tool_call':
                    positive_turns.add(int(span['turn']))
    links = []
    for event in fields.get('agent_trace') or []:
        if event.get('role') != 'tool':
            continue
        turn = int(event.get('turn') or 0)
        if not correct and turn not in positive_turns:
            continue
        for identifier in sorted(trace_ids(event.get('metadata') or {})):
            links.append({'trace_id': identifier, 'turn': turn, 'tool': event.get('name'), 'label': 'answer_correct' if correct else 'credit_positive', 'policy_step': int(fields['coev_policy_step']), 'mei_version': int(fields['coev_mei_version']), 'mei_model': fields['coev_mei_model'], 'rollout_request_id': fields.get('rollout_request_id'), 'qid': fields.get('qid'), 'credit_audit_path': (credit or {}).get('audit_path')})
    return links

def observe_answer(fields: dict, *, correct: bool, judge_called: bool, judge_parse_failed: bool) -> None:
    if root() is None or not correct or (not judge_called) or judge_parse_failed:
        return
    links = selected_mei_links(fields, correct=True)
    if links:
        append_event('selections', {'event': 'correct_rollout', 'links': links})

def observe_credit(fields: dict, credit: dict) -> None:
    if root() is None:
        return
    links = selected_mei_links(fields, correct=False, credit=credit)
    if links:
        append_event('selections', {'event': 'selected_failed_rollout_positive_turns', 'links': links})

def save_rollout(agent_data) -> None:
    pass
    destination = root()
    if destination is None:
        return
    from neos.workflow.mei_trace import MEITraceStore
    key = str(destination / 'policy_rollouts')
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = MEITraceStore(key)
            _STORES[key] = store
    fields = agent_data.extra_fields
    import base64
    import io
    from PIL import Image
    from neos.workflow.protocol import NEOS_SYSTEM_PROMPT
    visuals = []
    for index, image in enumerate(agent_data.image_data or [], 1):
        if not isinstance(image, Image.Image):
            raise TypeError('Policy visual capture requires the actual decoded PIL images')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        visuals.append({'ref': f'img_{index}', 'data_url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()})
    inputs = [{'role': 'system', 'content': fields.get('coev_system_prompt') or NEOS_SYSTEM_PROMPT}, {'role': 'user', 'content': [{'type': 'text', 'text': fields.get('question', '')}, *[{'type': 'coev_visual_record', 'value': v} for v in visuals]]}]
    output = {'trajectory': fields.get('turn_credit_trace') or [], 'final_answer': fields.get('final_answer', ''), 'termination_reason': fields.get('termination_reason'), 'all_token_ids': list(agent_data.prompt_ids), 'response_mask': list(agent_data.response_mask), 'agent_trace': fields.get('agent_trace') or []}
    saved = store.record(namespace='neos_policy_rollout', prompt_version='coevolution-policy-rollout', model=str(fields.get('coev_policy_model') or os.getenv('MODEL_PATH', 'policy')), fallback_models=[], messages=inputs, max_completion_tokens=len(agent_data.response_mask), raw_output=json.dumps(output, ensure_ascii=False, separators=(',', ':')), parsed_output=None, timing=None, descriptor={k: fields.get(k) for k in ['qid', 'ground_truth', 'rollout_request_id', 'rollout_session_id', 'coev_policy_step', 'coev_mei_version', 'coev_mei_model']})
    fields['coev_rollout_trace_id'] = saved['trace_id']

def mei_io_identity(record: dict, target: dict) -> str:
    pass
    messages = (record.get('input') or {}).get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('missing captured MEI input messages')
    value = [record.get('namespace'), messages, target]
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def previously_trained_mei_io(destination: Path, first_step: int) -> set[str]:
    used = set()
    for result_path in (destination / 'job_results').glob('after-step-*.json'):
        step = int(result_path.stem.rsplit('-', 1)[-1])
        if step >= first_step or read_json(result_path, {}).get('status') != 'trained':
            continue
        selection = destination / 'mei_training' / f'after-step-{step}' / 'data' / 'selection.jsonl'
        if selection.is_file():
            used.update((row['io_identity'] for row in jsonl_rows(selection) if row.get('accepted') and row.get('io_identity')))
    return used

def build_mei_batch(destination: Path, *, first_step: int, last_step: int, mei_model: str, output: Path) -> dict:
    pass
    output.mkdir(parents=True, exist_ok=True)
    with TraceReadAudit(output / 'quarantine') as read_audit:
        return _build_mei_batch(destination, first_step=first_step, last_step=last_step, mei_model=mei_model, output=output, read_audit=read_audit)

def _build_mei_batch(destination: Path, *, first_step: int, last_step: int, mei_model: str, output: Path, read_audit: TraceReadAudit) -> dict:
    links: dict[str, list[dict]] = {}
    for path in (destination / 'selections').glob('*.jsonl'):
        for row in jsonl_rows(path, audit=read_audit):
            for link in row.get('links') or []:
                if first_step <= int(link['policy_step']) <= last_step and link['mei_model'] == mei_model:
                    links.setdefault(link['trace_id'], []).append(link)
    from neos.workflow.mei_trace import ASSET_URI_PREFIX
    from neos.training import mei_data as converter
    output.mkdir(parents=True, exist_ok=True)
    accepted = []
    audit = []
    found = set()
    actual_found = set()
    missing_markers = set()
    dedup = set()
    already_trained = previously_trained_mei_io(destination, first_step)
    accepted_ids = set()
    duplicate_trace_records = 0
    for trace_path in trace_record_paths(destination / 'mei_traces'):
        for record in jsonl_rows(trace_path, audit=read_audit):
            identifier = record.get('trace_id')
            if identifier not in links:
                continue
            if identifier in accepted_ids:
                duplicate_trace_records += 1
                continue
            found.add(identifier)
            if record.get('record_type') == 'drop_missing_trace_reference':
                missing_markers.add(identifier)
                audit.append({'trace_id': identifier, 'accepted': False, 'reason': 'missing_full_mei_trace', 'links': links[identifier]})
                continue
            actual_found.add(identifier)
            reason = None
            payload = record.get('output') or {}
            target = payload.get('parsed_json')
            if record.get('configured_model') != mei_model:
                reason = 'mei_version_mismatch'
            elif record.get('namespace') not in {'neos_web_mei', 'neos_image_mei', 'neos_reverse_mei'}:
                reason = 'not_a_mei_task'
            elif not isinstance(target, dict) or not payload.get('raw_text', '').strip():
                reason = 'invalid_or_empty_mei_output'
            elif record.get('actual_model') not in {None, mei_model}:
                reason = 'unexpected_mei_route'
            identity = None
            if reason is None:
                try:
                    identity = mei_io_identity(record, target)
                except ValueError:
                    reason = 'missing_captured_input'
            if reason is None and identity in already_trained:
                reason = 'duplicate_previously_trained_io'
            if reason is None and identity in dedup:
                reason = 'duplicate_exact_io'
            if reason is None:
                try:
                    messages, images = converter.convert_messages(record, target)
                    example = {'messages': messages}
                    if images:
                        example['images'] = images
                    json.dumps(example, ensure_ascii=False).encode('utf-8')
                    accepted.append(example)
                    accepted_ids.add(identifier)
                    dedup.add(identity)
                except (ValueError, RuntimeError, OSError, UnicodeError, KeyError, TypeError) as exc:
                    reason = f'unavailable_input:{type(exc).__name__}'
            audit.append({'trace_id': identifier, 'accepted': reason is None, 'io_identity': identity, 'reason': reason, 'links': links[identifier], 'example_id': record.get('example_id')})
    unresolved = set(links) - found
    for identifier in sorted(unresolved):
        audit.append({'trace_id': identifier, 'accepted': False, 'reason': 'missing_full_mei_trace', 'links': links[identifier]})
    dropped_missing = unresolved | missing_markers - actual_found
    train = output / 'train.jsonl'
    with train.open('w') as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    with (output / 'selection.jsonl').open('w') as handle:
        for row in audit:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    manifest = {'schema': 'neos-coev-mei-sft', 'first_policy_step': first_step, 'last_policy_step': last_step, 'source_mei_model': mei_model, 'requested_trace_ids': len(links), 'found_trace_ids': len(actual_found), 'not_yet_published_trace_ids': len(unresolved), 'dropped_missing_trace_ids': sorted(dropped_missing), 'dropped_missing_trace_count': len(dropped_missing), 'missing_trace_policy': 'exclude_incomplete_examples_do_not_block_update', 'train_rows': len(accepted), 'train_file': str(train), 'trace_reader': dict(read_audit.counts), 'quarantine_dir': str(read_audit.root), 'duplicate_trace_records_skipped': duplicate_trace_records, 'deduplication': 'captured messages including image assets + canonical parsed JSON; across outcome/process sources and prior successful SFT updates', 'duplicate_current_window_count': sum((r.get('reason') == 'duplicate_exact_io' for r in audit)), 'duplicate_previously_trained_count': sum((r.get('reason') == 'duplicate_previously_trained_io' for r in audit)), 'max_length': 32768, 'reasoning_in_targets': False, 'input_source': 'captured_exact_mei_wire_messages_and_pixels', 'selection_is_weak_supervision_not_independent_mei_ground_truth': True}
    atomic_json(output / 'manifest.json', manifest)
    return manifest
