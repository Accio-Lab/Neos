from __future__ import annotations
import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
SCHEMA_VERSION = 'neos-mei-sft'
ASSET_URI_PREFIX = 'mei-asset://images/'

def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f'non-object row at {path}:{line_number}')
            rows.append(value)
    return rows

def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 384)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

def latest_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get('replay_id') or '')
        if not key:
            continue
        if key not in latest or int(row.get('attempt') or 1) >= int(latest[key].get('attempt') or 1):
            latest[key] = row
    return latest

def _asset_map(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {ASSET_URI_PREFIX + str(asset['asset_id']): asset for asset in trace.get('assets') or []}

def convert_messages(trace: dict[str, Any], target: dict[str, Any], reasoning_content: str | None=None) -> tuple[list[dict[str, Any]], list[str]]:
    assets = _asset_map(trace)
    images: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in (trace.get('input') or {}).get('messages') or []:
        role = str(message.get('role') or '')
        content = message.get('content')
        if isinstance(content, str):
            rendered = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get('type') == 'text':
                    text = str(item.get('text') or '')
                    if text:
                        parts.append(text)
                    continue
                if item.get('type') != 'image_url':
                    continue
                image_url = item.get('image_url') or {}
                uri = str(image_url.get('url') if isinstance(image_url, dict) else image_url)
                asset = assets.get(uri)
                if asset is None:
                    raise RuntimeError(f'trace references unknown MEI asset: {uri}')
                path = Path(str(asset.get('shared_path') or ''))
                if not path.is_file():
                    raise FileNotFoundError(f'MEI SFT image is unavailable: {path}')
                parts.append('<image>')
                images.append(str(path.resolve()))
            rendered = '\n'.join((part for part in parts if part))
        else:
            rendered = str(content or '')
        messages.append({'role': role, 'content': rendered})
    assistant = {'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}
    if reasoning_content is not None:
        assistant['reasoning_content'] = reasoning_content
    messages.append(assistant)
    return (messages, images)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, action='append', required=True)
    parser.add_argument('--traces', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-rows', type=int, default=128)
    parser.add_argument('--allow-unjudged', action='store_true')
    parser.add_argument('--accepted-status', action='append', dest='accepted_statuses', help='Result status to compile; repeat for multiple statuses. Defaults to accepted (plus materialized_unjudged when requested).')
    parser.add_argument('--historical-judge-status', help='When set, require historical_agreement_judgment.status to equal this value (for example accepted).')
    parser.add_argument('--include-reasoning-content', action='store_true', help="Train on the Qwen teacher's persisted hidden reasoning as the assistant reasoning_content preceding the strict-JSON answer.")
    args = parser.parse_args()
    if args.validation_rows < 0:
        parser.error('validation-rows must be non-negative')
    result_paths = [path.expanduser().resolve() for path in args.results]
    trace_paths = [path.expanduser().resolve() for path in args.traces]
    for path in (*result_paths, *trace_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    results = latest_results([row for path in result_paths for row in read_jsonl(path)])
    accepted_status = set(args.accepted_statuses or ['accepted'])
    if args.allow_unjudged:
        accepted_status.add('materialized_unjudged')
    selected = [row for row in results.values() if row.get('status') in accepted_status and (args.historical_judge_status is None or (row.get('historical_agreement_judgment') or {}).get('status') == args.historical_judge_status) and row.get('mei_trace_id') and ((row.get('source') or {}).get('main_trajectory_correct') is True) and ((row.get('source') or {}).get('benchmark_test_trajectory') is False)]
    traces = {str(row['trace_id']): row for path in trace_paths for row in read_jsonl(path) if row.get('trace_id')}
    compiled: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    for result in selected:
        trace_id = str(result['mei_trace_id'])
        trace = traces.get(trace_id)
        if trace is None:
            counters['missing_trace'] += 1
            continue
        if not trace.get('output', {}).get('parsed'):
            counters['invalid_trace'] += 1
            continue
        target = result.get('new_output')
        if not isinstance(target, dict):
            counters['invalid_target'] += 1
            continue
        reasoning_content: str | None = None
        if args.include_reasoning_content:
            persisted_reasoning = result.get('reasoning_content')
            if not isinstance(persisted_reasoning, str) or not persisted_reasoning.strip():
                raise RuntimeError(f"selected MEI replay has no persisted reasoning_content: {result.get('replay_id')}")
            reasoning_content = persisted_reasoning
        messages, images = convert_messages(trace, target, reasoning_content)
        example_id = str(trace.get('example_id') or trace_id)
        training = {'messages': messages, 'images': images}
        audit = {'example_id': example_id, 'trace_id': trace_id, 'replay_id': result['replay_id'], 'task_type': result['task_type'], 'source': result['source'], 'teacher_model': result.get('mei_actual_model') or result.get('teacher_model'), 'consistency': result.get('consistency'), 'reasoning_content_characters': len(reasoning_content or ''), 'image_count': len(images)}
        prior = compiled.get(example_id)
        if prior is not None:
            if canonical(prior[0]) != canonical(training):
                raise RuntimeError(f'conflicting duplicate MEI example: {example_id}')
            counters['duplicates_removed'] += 1
            continue
        compiled[example_id] = (training, audit)
    ordered = [compiled[key] for key in sorted(compiled)]
    validation_rows = min(args.validation_rows, max(0, len(ordered) // 20))
    validation = ordered[:validation_rows]
    training = ordered[validation_rows:]
    output = args.output.expanduser().resolve()
    train_payload = b''.join((canonical(row[0]) + b'\n' for row in training))
    val_payload = b''.join((canonical(row[0]) + b'\n' for row in validation))
    index_payload = b''.join((canonical(row[1]) + b'\n' for row in ordered))
    atomic_bytes(output / 'train.jsonl', train_payload)
    atomic_bytes(output / 'validation.jsonl', val_payload)
    atomic_bytes(output / 'index.jsonl', index_payload)
    task_counts = Counter((row[1]['task_type'] for row in ordered))
    manifest = {'schema_version': SCHEMA_VERSION, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'results': [{'path': str(path)} for path in result_paths], 'traces': [{'path': str(path)} for path in trace_paths], 'selection': {'main_trajectory_correct': True, 'benchmark_test_trajectory': False, 'accepted_status': sorted(accepted_status), 'historical_judge_status': args.historical_judge_status, 'teacher_output_used': 'new_output', 'teacher_reasoning_content_used': args.include_reasoning_content, 'historical_output_used_as_target': False}, 'rows': len(ordered), 'train_rows': len(training), 'validation_rows': len(validation), 'by_task': dict(sorted(task_counts.items())), 'counters': dict(sorted(counters.items())), 'files': {'train': {'path': str(output / 'train.jsonl')}, 'validation': {'path': str(output / 'validation.jsonl')}, 'index': {'path': str(output / 'index.jsonl')}}}
    atomic_bytes(output / 'manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8') + b'\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
