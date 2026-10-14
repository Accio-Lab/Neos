
from __future__ import annotations
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from neos.workflow.eval import _summary

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 384)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

def compact_summary_row(row: dict) -> dict:
    pass
    keys = {'bench', 'sid', 'error', 'skipped', 'skip_reason', 'reference_missing', 'correct', 'mei_model', 'mei_fallback_models', 'mei_actual_models', 'mei_gateway_calls', 'mei_fallback_calls', 'mei_failed_gateway_calls', 'judge_model', 'termination_reason', 'format_adapted_turns', 'format_recovery_model_calls', 'n_turns', 'tool_call_counts', 'tool_calls_executed', 'model_calls', 'protocol_metrics'}
    return {key: row.get(key) for key in keys if key in row}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--unified-run', type=Path, required=True)
    parser.add_argument('--group-dir', type=Path, required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--expected-total', type=int, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    args = parser.parse_args()
    if args.expected_total <= 0:
        parser.error('--expected-total must be positive')
    unified = args.unified_run.expanduser().resolve()
    group = args.group_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    manifest_path = unified / 'run_manifest.json'
    repeat_summary_path = unified / 'unified_repeat_summary.json'
    trajectories_path = unified / 'trajectories.jsonl'
    for required in (manifest_path, repeat_summary_path, trajectories_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    repeat_summary = json.loads(repeat_summary_path.read_text(encoding='utf-8'))
    repeat_count = int(repeat_summary.get('repeat_count') or 0)
    if repeat_count <= 1:
        raise ValueError('unified run does not contain repeated evaluation')
    benchmark_summary = (repeat_summary.get('benchmarks') or {}).get(args.benchmark)
    if not isinstance(benchmark_summary, dict):
        raise ValueError(f'unified summary has no benchmark {args.benchmark!r}')
    group.mkdir(parents=True, exist_ok=True)
    run_dirs = {repeat: group.parent / f'{group.name}__repeat{repeat}' for repeat in range(1, repeat_count + 1)}
    temporary_paths = {}
    handles = {}
    compact_rows = {repeat: [] for repeat in run_dirs}
    sid_sets = {repeat: set() for repeat in run_dirs}
    try:
        for repeat, run_dir in run_dirs.items():
            run_dir.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=run_dir, prefix='.trajectories.jsonl.', suffix='.tmp')
            temporary_paths[repeat] = Path(temporary)
            handles[repeat] = os.fdopen(descriptor, 'w', encoding='utf-8')
        with trajectories_path.open(encoding='utf-8') as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('bench') != args.benchmark:
                    continue
                repeat = int(row.get('evaluation_repeat') or 1)
                if repeat not in handles:
                    raise ValueError(f'invalid evaluation_repeat at line {line_number}: {repeat}')
                sid = str(row.get('sid'))
                if sid in sid_sets[repeat]:
                    raise ValueError(f'duplicate repeat={repeat} sid={sid}')
                sid_sets[repeat].add(sid)
                materialized = {**row, 'evaluation_run_id': run_dirs[repeat].name, 'unified_parent_run': unified.name}
                handles[repeat].write(json.dumps(materialized, ensure_ascii=False) + '\n')
                compact_rows[repeat].append(compact_summary_row(materialized))
        for repeat, handle in handles.items():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            os.chmod(temporary_paths[repeat], 384)
            if len(sid_sets[repeat]) != args.expected_total:
                raise ValueError(f'repeat {repeat} has {len(sid_sets[repeat])} unique rows; expected {args.expected_total}')
        canonical = sid_sets[1]
        if any((values != canonical for values in sid_sets.values())):
            raise ValueError('repeat SID sets differ')
        runs = []
        for repeat, run_dir in run_dirs.items():
            os.replace(temporary_paths[repeat], run_dir / 'trajectories.jsonl')
            summary = _summary(compact_rows[repeat], (args.benchmark,))
            bench = summary['benchmarks'][args.benchmark]
            atomic_json(run_dir / 'summary.json', summary)
            run_manifest = {**manifest, 'schema_version': 'neos-materialized-repeat-run', 'status': 'completed' if bench['errors'] == 0 else 'completed_with_errors', 'evaluation_repeat': repeat, 'repeat_count': repeat_count, 'benchmark_selection': args.benchmark, 'benchmark_ids': [args.benchmark], 'unified_parent_run': str(unified), 'materialized_at_utc': utc_now(), 'summary_file': str((run_dir / 'summary.json').resolve()), 'trajectory_file': str((run_dir / 'trajectories.jsonl').resolve())}
            atomic_json(run_dir / 'run_manifest.json', run_manifest)
            runs.append({'repeat': repeat, 'run_dir': str(run_dir), 'correct': bench['correct'], 'expected_total': args.expected_total, 'fixed_denominator_accuracy': bench['correct'] / args.expected_total})
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
    average = benchmark_summary['average']
    passed = benchmark_summary['pass']
    group_payload = {'schema_version': 'neos-unified-event-average-pass', 'created_at_utc': utc_now(), 'model_path': str(model_path), 'benchmark': args.benchmark, 'expected_total_per_repeat': args.expected_total, 'protocol': {'repeat_execution': 'one-process-global-repeat-benchmark-queue', 'trajectory_routing': 'least-loaded-trajectory-sticky', 'execution_scheduler': 'turn-event-model-tool-judge-pools', 'source_run': str(unified)}, 'runs': runs, 'average4' if repeat_count == 4 else 'average': average, 'pass4' if repeat_count == 4 else 'pass': passed}
    atomic_json(group / 'average4_summary.json', group_payload)
    print(json.dumps(group_payload, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
