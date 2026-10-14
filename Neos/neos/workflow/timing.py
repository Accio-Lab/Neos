
from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Any, Iterable
TIMING_SUMMARY_SCHEMA_VERSION = 'neos-eval-timing-summary'

def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None

def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

def distribution(values: Iterable[Any]) -> dict[str, Any]:
    clean = sorted((number for value in values if (number := _number(value)) is not None))
    total = sum(clean)
    return {'count': len(clean), 'total_ms': round(total, 3), 'mean_ms': round(total / len(clean), 3) if clean else None, 'p50_ms': round(value, 3) if (value := _percentile(clean, 0.5)) is not None else None, 'p90_ms': round(value, 3) if (value := _percentile(clean, 0.9)) is not None else None, 'p95_ms': round(value, 3) if (value := _percentile(clean, 0.95)) is not None else None, 'max_ms': round(clean[-1], 3) if clean else None}

def build_timing_summary(rows: list[dict[str, Any]], persistence_events: list[dict[str, Any]], *, selected: tuple[str, ...], evaluator_wall_ms: float, phase_timings_ms: dict[str, float], invocations: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    if invocations:
        evaluator_wall_ms = sum((float(invocation.get('evaluator_wall_ms') or 0) for invocation in invocations))
        accumulated_phases: dict[str, float] = defaultdict(float)
        for invocation in invocations:
            for key, value in (invocation.get('phase_timings_ms') or {}).items():
                accumulated_phases[key] += float(value or 0)
        phase_timings_ms = dict(accumulated_phases)
        failed_attempt_wall_ms = sum((float(invocation.get('error_sample_worker_wall_ms') or 0) for invocation in invocations))
        failed_attempts = sum((int(invocation.get('error_sample_attempts') or 0) for invocation in invocations))
    else:
        failed_attempt_wall_ms = 0.0
        failed_attempts = 0
    timed_rows = [row for row in rows if isinstance(row.get('timing'), dict)]
    skipped_rows = [row for row in rows if row.get('skipped')]
    successful_rows = [row for row in rows if not row.get('error') and (not row.get('skipped'))]
    sample_wall = []
    agent_wall = []
    judge_latency = []
    model_calls = []
    model_calls_by_turn: dict[int, list[float]] = defaultdict(list)
    model_calls_by_prompt_bucket: dict[str, list[float]] = defaultdict(list)
    model_prompt_tokens = 0
    model_completion_tokens = 0
    model_cached_tokens = 0
    assistant_parse = []
    tool_runtime = []
    tool_postprocess = []
    input_preparation = []
    agent_unattributed = []
    tool_by_name: dict[str, list[float]] = defaultdict(list)
    internal_by_operation: dict[str, list[float]] = defaultdict(list)
    judge_skipped: Counter[str] = Counter()
    for row in successful_rows:
        evaluation_timing = row.get('evaluation_timing') or {}
        if (value := _number(evaluation_timing.get('sample_worker_wall_ms'))) is not None:
            sample_wall.append(value)
        judge_skipped_reason = evaluation_timing.get('judge_skipped_reason')
        if judge_skipped_reason:
            judge_skipped[str(judge_skipped_reason)] += 1
        elif (value := _number(evaluation_timing.get('judge_latency_ms'))) is not None:
            judge_latency.append(value)
        timing = row.get('timing') or {}
        if (value := _number(timing.get('agent_wall_ms'))) is not None:
            agent_wall.append(value)
        if (value := _number(timing.get('unattributed_agent_ms'))) is not None:
            agent_unattributed.append(value)
        input_preparation.append(sum((float(timing.get(field) or 0.0) for field in ('input_image_load_ms', 'input_image_prepare_ms', 'input_artifact_ms', 'initial_message_build_ms'))))
        for event in row.get('trajectory') or []:
            if event.get('role') == 'assistant':
                if (value := _number(event.get('model_latency_ms'))) is not None:
                    model_calls.append(value)
                    model_calls_by_turn[int(event.get('turn') or 0)].append(value)
                    usage = ((event.get('provider_metadata') or {}).get('response') or {}).get('usage') or {}
                    prompt_tokens = int(usage.get('prompt_tokens') or 0)
                    completion_tokens = int(usage.get('completion_tokens') or 0)
                    cached_tokens = int((usage.get('prompt_tokens_details') or {}).get('cached_tokens') or 0)
                    model_prompt_tokens += prompt_tokens
                    model_completion_tokens += completion_tokens
                    model_cached_tokens += cached_tokens
                    if prompt_tokens < 16384:
                        bucket = '000k-016k'
                    elif prompt_tokens < 32768:
                        bucket = '016k-032k'
                    elif prompt_tokens < 65536:
                        bucket = '032k-064k'
                    elif prompt_tokens < 98304:
                        bucket = '064k-096k'
                    elif prompt_tokens < 131072:
                        bucket = '096k-128k'
                    else:
                        bucket = '128k+'
                    if prompt_tokens:
                        model_calls_by_prompt_bucket[bucket].append(value)
                if (value := _number(event.get('assistant_parse_latency_ms'))) is not None:
                    assistant_parse.append(value)
            elif event.get('role') == 'tool':
                if (value := _number(event.get('latency_ms'))) is not None:
                    tool_runtime.append(value)
                    tool_by_name[str(event.get('name') or 'unknown')].append(value)
                if (value := _number(event.get('postprocess_latency_ms'))) is not None:
                    tool_postprocess.append(value)
                internal = (event.get('metadata') or {}).get('timing') or {}
                operations = internal.get('operations') or []
                for operation in operations:
                    if not isinstance(operation, dict):
                        continue
                    for key, raw_value in operation.items():
                        if key.endswith('_ms') and key not in {'worker_wall_ms'}:
                            if (value := _number(raw_value)) is not None:
                                internal_by_operation[key].append(value)
                if (value := _number(internal.get('python_sandbox_ms'))) is not None:
                    internal_by_operation['python_sandbox_ms'].append(value)
    components = {'main_model_inference': sum(model_calls), 'tool_runtime': sum(tool_runtime), 'tool_result_postprocess': sum(tool_postprocess), 'assistant_output_parse': sum(assistant_parse), 'input_image_and_message_preparation': sum(input_preparation), 'judge': sum(judge_latency), 'agent_unattributed': sum(agent_unattributed), 'failed_sample_attempts': failed_attempt_wall_ms}
    cumulative_sample_ms = sum(sample_wall) + failed_attempt_wall_ms
    ranking = []
    for name, total in sorted(components.items(), key=lambda item: item[1], reverse=True):
        ranking.append({'operation': name, 'total_ms': round(total, 3), 'share_of_cumulative_sample_wall': round(total / cumulative_sample_ms, 6) if cumulative_sample_ms > 0 else None})
    per_benchmark = {}
    for benchmark in selected:
        benchmark_rows = [row for row in successful_rows if row.get('bench') == benchmark]
        per_benchmark[benchmark] = {'samples': len(benchmark_rows), 'sample_worker_wall': distribution(((row.get('evaluation_timing') or {}).get('sample_worker_wall_ms') for row in benchmark_rows)), 'agent_wall': distribution(((row.get('timing') or {}).get('agent_wall_ms') for row in benchmark_rows)), 'judge': distribution(((row.get('evaluation_timing') or {}).get('judge_latency_ms') for row in benchmark_rows if not (row.get('evaluation_timing') or {}).get('judge_skipped_reason'))), 'turns': {'total': sum((int(row.get('n_turns') or 0) for row in benchmark_rows)), 'mean': round(sum((int(row.get('n_turns') or 0) for row in benchmark_rows)) / len(benchmark_rows), 3) if benchmark_rows else None}}
    persistence_fields = ('json_serialize_ms', 'write_lock_wait_ms', 'file_open_ms', 'trajectory_write_ms', 'trajectory_flush_ms', 'trajectory_fsync_ms', 'file_close_ms', 'persistence_wall_ms')
    persistence = {field: distribution((event.get(field) for event in persistence_events)) for field in persistence_fields}
    total_events = len(persistence_events)
    total_persistence_ms = sum((float(event.get('persistence_wall_ms') or 0.0) for event in persistence_events))
    return {'schema_version': TIMING_SUMMARY_SCHEMA_VERSION, 'rows': len(rows), 'successful_rows': len(successful_rows), 'skipped_rows': len(skipped_rows), 'skip_reasons': dict(sorted(Counter((row.get('skip_reason') or 'unknown' for row in skipped_rows)).items())), 'timed_rows': len(timed_rows), 'evaluator_wall_ms': round(evaluator_wall_ms, 3), 'evaluator_wall_hours': round(evaluator_wall_ms / 3600000, 6), 'evaluation_invocations': len(invocations or []) or 1, 'failed_sample_attempts': failed_attempts, 'failed_sample_attempt_wall_ms': round(failed_attempt_wall_ms, 3), 'phase_timings_ms': {key: round(float(value), 3) for key, value in sorted(phase_timings_ms.items())}, 'throughput': {'samples_per_hour': round(len(successful_rows) * 3600000 / evaluator_wall_ms, 3) if evaluator_wall_ms > 0 else None, 'cumulative_sample_wall_ms': round(cumulative_sample_ms, 3), 'effective_parallelism': round(cumulative_sample_ms / evaluator_wall_ms, 3) if evaluator_wall_ms > 0 else None}, 'distributions': {'sample_worker_wall': distribution(sample_wall), 'agent_wall': distribution(agent_wall), 'main_model_call': distribution(model_calls), 'assistant_output_parse': distribution(assistant_parse), 'tool_runtime': distribution(tool_runtime), 'tool_result_postprocess': distribution(tool_postprocess), 'judge': distribution(judge_latency), 'input_image_and_message_preparation': distribution(input_preparation), 'agent_unattributed': distribution(agent_unattributed)}, 'bottleneck_ranking': ranking, 'tool_runtime_by_name': {name: distribution(values) for name, values in sorted(tool_by_name.items())}, 'main_model': {'calls_by_turn': {str(turn): distribution(values) for turn, values in sorted(model_calls_by_turn.items())}, 'latency_by_prompt_token_bucket': {bucket: distribution(values) for bucket, values in sorted(model_calls_by_prompt_bucket.items())}, 'prompt_tokens': model_prompt_tokens, 'completion_tokens': model_completion_tokens, 'cached_prompt_tokens': model_cached_tokens, 'completion_tokens_per_second_of_model_call': round(model_completion_tokens * 1000 / sum(model_calls), 3) if sum(model_calls) > 0 else None}, 'tool_internal_operations': {name: distribution(values) for name, values in sorted(internal_by_operation.items())}, 'judge_skipped': dict(sorted(judge_skipped.items())), 'trajectory_persistence': {'events': total_events, 'total_persistence_wall_ms': round(total_persistence_ms, 3), 'share_of_evaluator_wall': round(total_persistence_ms / evaluator_wall_ms, 6) if evaluator_wall_ms > 0 else None, **persistence}, 'benchmarks': per_benchmark}
